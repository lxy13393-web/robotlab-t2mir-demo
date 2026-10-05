"""Shared multidimensional dynamics-profile utilities for G1 experiments."""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DynamicsProfile:
    task_id: int
    action_lag: float
    motor_strength: float
    payload_kg: float
    friction: float
    split: str = "train"

    def validate(self) -> None:
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if not 0.0 <= self.action_lag < 1.0:
            raise ValueError("action_lag must be in [0, 1)")
        if self.motor_strength <= 0.0:
            raise ValueError("motor_strength must be positive")
        if self.payload_kg < 0.0:
            raise ValueError("payload_kg must be non-negative")
        if self.friction <= 0.0:
            raise ValueError("friction must be positive")
        if self.split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")

    @property
    def tag(self) -> str:
        def number(value: float) -> str:
            return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")

        return (
            f"task{self.task_id:02d}_lag{number(self.action_lag)}"
            f"_motor{number(self.motor_strength)}_payload{number(self.payload_kg)}"
            f"_friction{number(self.friction)}"
        )

    def cli_args(self) -> list[str]:
        return [
            "--action_lag", str(self.action_lag),
            "--motor_strength", str(self.motor_strength),
            "--payload_kg", str(self.payload_kg),
            "--friction", str(self.friction),
            "--deterministic_dynamics",
        ]


def default_profiles() -> list[DynamicsProfile]:
    """Return the interpretable 4 x 3 x 2 x 2 G1 dynamics grid."""
    values = product(
        (0.20, 0.32, 0.40, 0.43),
        (0.80, 0.90, 1.00),
        (0.0, 3.0),
        (0.65, 1.00),
    )
    # Six spread-out combinations are withheld from T2MIR parameter updates.
    eval_ids = {5, 14, 23, 32, 41, 47}
    return [
        DynamicsProfile(task_id, lag, motor, payload, friction,
                        "eval" if task_id in eval_ids else "train")
        for task_id, (lag, motor, payload, friction) in enumerate(values)
    ]


def validate_profiles(profiles: Iterable[DynamicsProfile], expected_count: int | None = None) -> list[DynamicsProfile]:
    profiles = list(profiles)
    if expected_count is not None and len(profiles) != expected_count:
        raise ValueError(f"expected {expected_count} profiles, got {len(profiles)}")
    for profile in profiles:
        profile.validate()
    ids = [profile.task_id for profile in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs must be unique")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("task IDs must be contiguous from zero")
    tuples = [(p.action_lag, p.motor_strength, p.payload_kg, p.friction) for p in profiles]
    if len(tuples) != len(set(tuples)):
        raise ValueError("dynamics parameter tuples must be unique")
    return profiles


def manifest_payload(profiles: Iterable[DynamicsProfile]) -> dict:
    profiles = validate_profiles(profiles)
    task_rows = [asdict(profile) | {"tag": profile.tag} for profile in profiles]
    canonical = json.dumps(task_rows, sort_keys=True, separators=(",", ":"))
    return {
        "format_version": 1,
        "description": "RobotLab G1 4D hidden-dynamics benchmark",
        "parameter_semantics": {
            "action_lag": "first-order action filter coefficient",
            "motor_strength": "actuator effort-limit multiplier",
            "payload_kg": "mass added to torso_link",
            "friction": "fixed static and dynamic robot-body friction coefficient",
        },
        "task_count": len(task_rows),
        "training_tasks": sum(row["split"] == "train" for row in task_rows),
        "eval_tasks": [row["task_id"] for row in task_rows if row["split"] == "eval"],
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "tasks": task_rows,
    }


def write_manifest(path: Path, profiles: Iterable[DynamicsProfile] | None = None) -> None:
    payload = manifest_payload(default_profiles() if profiles is None else profiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_manifest(path: Path) -> tuple[dict, list[DynamicsProfile]]:
    payload = json.loads(path.read_text())
    profiles = validate_profiles(
        DynamicsProfile(
            task_id=int(row["task_id"]),
            action_lag=float(row["action_lag"]),
            motor_strength=float(row["motor_strength"]),
            payload_kg=float(row["payload_kg"]),
            friction=float(row["friction"]),
            split=str(row["split"]),
        )
        for row in payload["tasks"]
    )
    expected = manifest_payload(profiles)
    for key in ("task_count", "training_tasks", "eval_tasks", "sha256"):
        if payload.get(key) != expected[key]:
            raise ValueError(f"manifest {key} mismatch: stored={payload.get(key)!r}, expected={expected[key]!r}")
    return payload, profiles


def configure_env_dynamics(env_cfg, *, motor_strength: float, payload_kg: float,
                           friction: float | None, deterministic: bool,
                           payload_body: str = "torso_link") -> dict:
    """Apply one fixed dynamics profile before ``gym.make`` is called."""
    if motor_strength <= 0.0 or payload_kg < 0.0 or (friction is not None and friction <= 0.0):
        raise ValueError("invalid fixed dynamics profile")

    scaled_actuators = {}
    for name, actuator_cfg in env_cfg.scene.robot.actuators.items():
        if actuator_cfg.effort_limit is not None:
            actuator_cfg.effort_limit *= motor_strength
            scaled_actuators[name] = float(actuator_cfg.effort_limit)
    if motor_strength != 1.0 and not scaled_actuators:
        raise RuntimeError("selected robot has no configurable actuator effort limits")

    if deterministic:
        if hasattr(env_cfg.events, "randomize_actuator_gains"):
            env_cfg.events.randomize_actuator_gains = None
        if hasattr(env_cfg.events, "randomize_joint_parameters"):
            env_cfg.events.randomize_joint_parameters = None

    if friction is not None:
        material = env_cfg.events.physics_material
        if material is None:
            raise RuntimeError("environment has no physics_material event")
        material.params["static_friction_range"] = (friction, friction)
        material.params["dynamic_friction_range"] = (friction, friction)

    if payload_kg > 0.0:
        from omni.isaac.lab.managers import EventTermCfg as EventTerm
        import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

        from omni.isaac.lab.managers import SceneEntityCfg

        env_cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[payload_body]),
                "mass_distribution_params": (payload_kg, payload_kg),
                "operation": "add",
            },
        )
    elif deterministic and hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    return {
        "motor_strength": motor_strength,
        "payload_kg": payload_kg,
        "friction": friction,
        "deterministic_dynamics": deterministic,
        "payload_body": payload_body,
        "scaled_effort_limits": scaled_actuators,
    }
