"""Static deployment contract for the RobotLab Unitree G1 policies.

This module deliberately depends only on the Python standard library and
NumPy.  It is the single source of truth shared by the MuJoCo, ROS 2, PPO and
T2MIR adapters.  In particular, a simulator's native joint order must never be
assumed to match the order used during RobotLab training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np


# Isaac Lab exposes joints in PhysX articulation-tree order (not URDF/MJCF file
# order).  This exact sequence is the 37-DoF ``robot.data.joint_names`` order
# for the G1 minimal USD used by the training environment.  MuJoCo vectors are
# explicitly remapped to this order at the deployment boundary.
G1_POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
)


# Values are taken from G1_MINIMAL_CFG used by the recorded RobotLab run.
_DEFAULT_BY_NAME = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_pitch_joint": 0.87,
    "right_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}
G1_DEFAULT_JOINT_POSITIONS: tuple[float, ...] = tuple(
    _DEFAULT_BY_NAME.get(name, 0.0) for name in G1_POLICY_JOINT_NAMES
)

# Root pose from the exact RobotLab run recorded in
# ``logs/rsl_rl/unitree_g1_flat/2026-08-30_23-35-43/params/env.yaml``.  The
# published Unitree MJCF uses a slightly different 0.755 m spawn height, so the
# deployment boundary must not inherit its qpos0 silently.
G1_INITIAL_ROOT_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.74)
G1_INITIAL_ROOT_QUATERNION_WXYZ: tuple[float, float, float, float] = (
    1.0,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class ObservationTerm:
    """One contiguous term in the flattened policy observation."""

    name: str
    width: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation term name cannot be empty")
        if self.width <= 0:
            raise ValueError(f"observation term {self.name!r} needs a positive width")


G1_OBSERVATION_TERMS: tuple[ObservationTerm, ...] = (
    ObservationTerm("base_linear_velocity", 3),
    ObservationTerm("base_angular_velocity", 3),
    ObservationTerm("projected_gravity", 3),
    ObservationTerm("velocity_command", 3),
    ObservationTerm("joint_position_relative", 37),
    ObservationTerm("joint_velocity_relative", 37),
    ObservationTerm("previous_action", 37),
)


@dataclass(frozen=True)
class DeploymentContract:
    """Policy-facing dimensions, timing and joint semantics."""

    joint_names: tuple[str, ...] = G1_POLICY_JOINT_NAMES
    default_joint_positions: tuple[float, ...] = G1_DEFAULT_JOINT_POSITIONS
    observation_terms: tuple[ObservationTerm, ...] = G1_OBSERVATION_TERMS
    physics_dt: float = 0.005
    decimation: int = 4
    action_scale: float = 0.5
    initial_root_position: tuple[float, float, float] = G1_INITIAL_ROOT_POSITION
    initial_root_quaternion_wxyz: tuple[float, float, float, float] = (
        G1_INITIAL_ROOT_QUATERNION_WXYZ
    )

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ValueError("joint_names cannot be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if len(self.default_joint_positions) != len(self.joint_names):
            raise ValueError("one default joint position is required per joint")
        if len({term.name for term in self.observation_terms}) != len(self.observation_terms):
            raise ValueError("observation term names must be unique")
        if self.physics_dt <= 0.0 or not np.isfinite(self.physics_dt):
            raise ValueError("physics_dt must be finite and positive")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if self.action_scale <= 0.0 or not np.isfinite(self.action_scale):
            raise ValueError("action_scale must be finite and positive")
        defaults = np.asarray(self.default_joint_positions, dtype=np.float64)
        if not np.isfinite(defaults).all():
            raise ValueError("default_joint_positions must be finite")
        root_position = np.asarray(self.initial_root_position, dtype=np.float64)
        root_quaternion = np.asarray(
            self.initial_root_quaternion_wxyz, dtype=np.float64
        )
        if root_position.shape != (3,) or not np.isfinite(root_position).all():
            raise ValueError("initial_root_position must contain three finite values")
        if root_quaternion.shape != (4,) or not np.isfinite(root_quaternion).all():
            raise ValueError(
                "initial_root_quaternion_wxyz must contain four finite values"
            )
        if not np.isclose(
            np.linalg.norm(root_quaternion), 1.0, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("initial_root_quaternion_wxyz must be unit length")

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return sum(term.width for term in self.observation_terms)

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def policy_hz(self) -> float:
        return 1.0 / self.policy_dt

    @property
    def observation_slices(self) -> dict[str, slice]:
        offset = 0
        result: dict[str, slice] = {}
        for term in self.observation_terms:
            result[term.name] = slice(offset, offset + term.width)
            offset += term.width
        return result

    def as_dict(self) -> dict[str, object]:
        """Canonical, serialisable policy/simulator interface contract."""

        return {
            "schema": "robotlab-g1-deployment-contract-v2",
            "joint_names": list(self.joint_names),
            "default_joint_positions": list(self.default_joint_positions),
            "observation_terms": [
                {"name": term.name, "width": term.width} for term in self.observation_terms
            ],
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
            "action_scale": self.action_scale,
            "initial_root_position": list(self.initial_root_position),
            "initial_root_quaternion_wxyz": list(
                self.initial_root_quaternion_wxyz
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

    @property
    def sha256(self) -> str:
        """Stable fingerprint recorded beside every deployment result."""

        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_joint_names(self, names: Sequence[str], *, require_order: bool = True) -> None:
        """Validate a simulator joint list against the policy contract.

        ``require_order=False`` validates the name set only.  Call
        :meth:`reorder_indices` before consuming vectors in that case.
        """

        actual = tuple(str(name) for name in names)
        if len(set(actual)) != len(actual):
            duplicates = sorted({name for name in actual if actual.count(name) > 1})
            raise ValueError(f"simulator joint names contain duplicates: {duplicates}")
        expected_set = set(self.joint_names)
        actual_set = set(actual)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            raise ValueError(f"joint-name set mismatch; missing={missing}, extra={extra}")
        if require_order and actual != self.joint_names:
            mismatch = next(
                index
                for index, (expected, received) in enumerate(zip(self.joint_names, actual, strict=True))
                if expected != received
            )
            raise ValueError(
                "joint order mismatch at index "
                f"{mismatch}: expected {self.joint_names[mismatch]!r}, got {actual[mismatch]!r}"
            )

    def reorder_indices(self, source_names: Sequence[str]) -> np.ndarray:
        """Return indices that map a source joint vector into policy order."""

        self.validate_joint_names(source_names, require_order=False)
        source_by_name = {name: index for index, name in enumerate(source_names)}
        return np.asarray([source_by_name[name] for name in self.joint_names], dtype=np.int64)

    def reorder_joint_vector(
        self, values: np.ndarray | Sequence[float], source_names: Sequence[str]
    ) -> np.ndarray:
        """Copy a one-dimensional source vector into canonical policy order."""

        vector = np.asarray(values)
        if vector.shape != (len(source_names),):
            raise ValueError(
                f"joint vector shape {vector.shape} does not match {len(source_names)} source names"
            )
        return vector[self.reorder_indices(source_names)].copy()

    def action_to_joint_targets(self, action: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply RobotLab's default-offset joint-position action transform."""

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape[-1:] != (self.action_dim,):
            raise ValueError(
                f"action trailing dimension must be {self.action_dim}, got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("action contains NaN or Inf")
        defaults = np.asarray(self.default_joint_positions, dtype=np.float32)
        return defaults + np.float32(self.action_scale) * action_array


DEFAULT_CONTRACT = DeploymentContract()

# Fail at import time if a future edit accidentally changes the trained model
# interface while leaving this module's label unchanged.
if DEFAULT_CONTRACT.action_dim != 37 or DEFAULT_CONTRACT.state_dim != 123:
    raise RuntimeError("RobotLab G1 deployment contract must remain 123-state/37-action")


__all__ = [
    "DEFAULT_CONTRACT",
    "DeploymentContract",
    "G1_DEFAULT_JOINT_POSITIONS",
    "G1_INITIAL_ROOT_POSITION",
    "G1_INITIAL_ROOT_QUATERNION_WXYZ",
    "G1_OBSERVATION_TERMS",
    "G1_POLICY_JOINT_NAMES",
    "ObservationTerm",
]
