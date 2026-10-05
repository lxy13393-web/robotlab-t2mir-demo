"""CPU-only contract helpers for fair stationary PPO online evaluation.

Keep this module free of Isaac Sim imports.  Besides making protocol checks
cheap to test, that separation lets the executable reject an invalid training
profile or checkpoint hash before starting Kit.
"""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.protocols.dynamics_profile import DynamicsProfile, load_manifest

if TYPE_CHECKING:
    import torch


DEFAULT_EPISODES = 8
DEFAULT_NUM_ENVS = 32
POLICY_KIND = "ppo"
EPISODE_BOUNDARY_MECHANISM = "wrapper_initial_reset_then_forced_timeout_auto_reset"


@dataclass(frozen=True)
class StaticEvaluationContract:
    """Inputs whose identity can be established without starting Isaac Sim."""

    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest: dict[str, Any]
    profile: DynamicsProfile


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol_values(
    *,
    episodes: int,
    num_envs: int,
    goal_min_distance: float,
    goal_max_distance: float,
    goal_timeout: float,
    goal_hold_time: float,
) -> None:
    """Validate the simulator-independent portion of the CLI contract."""

    if episodes < 2:
        raise ValueError("--episodes must be at least 2 to preserve the 8-episode protocol shape")
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if not 0.0 < goal_min_distance < goal_max_distance:
        raise ValueError("goal distances must satisfy 0 < min < max")
    if goal_timeout <= 0.0 or goal_hold_time <= 0.0:
        raise ValueError("goal timeout and hold time must be positive")


def heldout_profile_from_manifest(path: Path, profile_id: int) -> tuple[dict, DynamicsProfile]:
    """Load exactly one held-out profile and reject all training profiles."""

    manifest, profiles = load_manifest(path)
    matches = [profile for profile in profiles if profile.task_id == profile_id]
    if len(matches) != 1:
        raise ValueError(f"profile {profile_id} is absent or duplicated in {path}")
    profile = matches[0]
    if profile.split != "eval":
        raise ValueError(
            f"profile {profile_id} split={profile.split!r}; official online evaluation "
            "requires split='eval'"
        )
    return manifest, profile


def prepare_static_contract(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str | None,
    manifest: Path,
    profile_id: int,
) -> StaticEvaluationContract:
    """Resolve and validate immutable evaluation inputs before Kit starts."""

    checkpoint_path = checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_checkpoint_sha256 is not None:
        expected = expected_checkpoint_sha256.strip().lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError("--checkpoint-sha256 must be a 64-character hexadecimal digest")
        if checkpoint_sha256 != expected:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: {checkpoint_sha256} != {expected}"
            )

    manifest_path = manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dynamics manifest does not exist: {manifest_path}")
    manifest_payload, profile = heldout_profile_from_manifest(manifest_path, profile_id)
    return StaticEvaluationContract(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        manifest_path=manifest_path,
        manifest=manifest_payload,
        profile=profile,
    )


def generate_t2mir_identical_scenarios(
    *,
    seed: int,
    episodes: int,
    num_envs: int,
    goal_min_distance: float,
    goal_max_distance: float,
) -> tuple["torch.Tensor", str]:
    """Generate goals bit-for-bit like ``evaluate_t2mir_online.py``.

    The three independent ``torch.rand(shape)`` calls and their order are part
    of the formal comparison contract.  Replacing them with one
    ``torch.rand((*shape, 3))`` call changes which random values are assigned
    to distance, bearing, and relative yaw.
    """

    # Imported lazily so preflight/profile validation remains safe before Kit
    # starts in the executable.
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (episodes, num_envs)
    distances = goal_min_distance + (goal_max_distance - goal_min_distance) * torch.rand(
        shape, generator=generator
    )
    bearings = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
    relative_yaws = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
    relative_x = distances * torch.cos(bearings)
    relative_y = distances * torch.sin(bearings)
    scenarios = torch.stack((relative_x, relative_y, relative_yaws), dim=-1)
    scenario_sha256 = sha256_bytes(scenarios.numpy().tobytes())
    return scenarios, scenario_sha256


def pose_sha256(x: float, y: float, yaw: float) -> str:
    """Return an architecture-stable hash for one float32 planar start pose."""

    return sha256_bytes(struct.pack("<fff", x, y, yaw))


def freeze_evaluation_reset(env_cfg: Any) -> dict[str, Any]:
    """Freeze reset state so action-dependent auto-resets cannot break pairing.

    A fall makes Isaac automatically reset that replica and consume reset RNG.
    Without this guard, two policies that fall at different times can start a
    later nominally paired episode from different random base poses.  Terrain
    origins remain replica-specific; only the root offset/yaw, velocity and
    joint-reset noise are removed.
    """

    events = getattr(env_cfg, "events", None)
    reset_base = getattr(events, "reset_base", None)
    if reset_base is None or not isinstance(getattr(reset_base, "params", None), dict):
        raise ValueError("environment has no configurable reset_base event")
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_joints = getattr(events, "reset_robot_joints", None)
    if reset_joints is not None and isinstance(getattr(reset_joints, "params", None), dict):
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)
    return {
        "root_pose_range": reset_base.params["pose_range"],
        "root_velocity_range": reset_base.params["velocity_range"],
        "joint_position_scale_range": (
            reset_joints.params.get("position_range") if reset_joints is not None else None
        ),
        "joint_velocity_range": (
            reset_joints.params.get("velocity_range") if reset_joints is not None else None
        ),
    }


def synchronize_vector_episode_boundary(
    *, env: Any, command_term: Any, episode_index: int
) -> dict[str, Any]:
    """Start a clean vector episode without repeatedly calling ``env.reset``.

    ``RslRlVecEnvWrapper`` already resets every replica in its constructor.
    Calling the wrapped Gym environment's global ``reset`` again at every
    evaluation episode is both redundant and, with the RobotLab/Isaac Sim 4.2
    stack used by this project, can make Kit close natively on a later reset.

    Episode zero therefore consumes the constructor reset. Later boundaries
    set every replica one step before its time limit and execute one zero-action
    step. ``ManagerBasedRLEnv.step`` then takes its normal timeout path and
    calls ``_reset_idx`` for every replica. The boundary transition is owned
    by the evaluator, not either policy: callers must invoke this before
    starting the policy episode and must not include it in returns or prompts.

    The postconditions are deliberately strict. A partial/non-timeout reset
    would invalidate paired A/D/PPO initial states and is therefore fatal.
    """

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")

    # Imported lazily so the module's static preflight helpers remain cheap.
    import torch

    episode_lengths = env.episode_length_buf
    if episode_index == 0:
        if not bool(torch.all(episode_lengths == 0).item()):
            raise RuntimeError(
                "RSL wrapper constructor did not leave every replica at episode length zero"
            )
        return {
            "mechanism": "rsl_wrapper_constructor_reset",
            "replicas_reset": int(env.num_envs),
            "boundary_step_excluded": True,
        }

    max_episode_length = int(env.max_episode_length)
    if max_episode_length <= 0:
        raise RuntimeError(f"invalid max_episode_length={max_episode_length}")

    command = getattr(command_term, "vel_command_b", None)
    if not torch.is_tensor(command):
        raise RuntimeError("base_velocity command term has no tensor vel_command_b")
    command.zero_()

    # ManagerBasedRLEnv.step increments this counter before checking timeout.
    episode_lengths.fill_(max_episode_length - 1)
    zero_actions = torch.zeros(
        (int(env.num_envs), int(env.num_actions)),
        dtype=torch.float32,
        device=episode_lengths.device,
    )
    _, _, dones, extras = env.step(zero_actions)
    time_outs = extras.get("time_outs")
    if time_outs is None:
        raise RuntimeError("forced episode boundary returned no time_outs audit tensor")
    if not bool(torch.all(dones.bool()).item()):
        count = int(dones.bool().sum().item())
        raise RuntimeError(f"forced episode boundary reset only {count}/{env.num_envs} replicas")
    if not bool(torch.all(time_outs.bool()).item()):
        count = int(time_outs.bool().sum().item())
        raise RuntimeError(
            f"forced episode boundary timed out only {count}/{env.num_envs} replicas"
        )
    if not bool(torch.all(env.episode_length_buf == 0).item()):
        remaining = env.episode_length_buf.detach().cpu().tolist()
        raise RuntimeError(
            "forced episode boundary did not reset episode counters to zero: "
            f"{remaining}"
        )
    return {
        "mechanism": "manager_based_rl_forced_timeout_auto_reset",
        "replicas_reset": int(env.num_envs),
        "boundary_step_excluded": True,
    }


def summarize(rows: list[dict[str, Any]], episodes: int) -> dict[str, Any]:
    """Produce the same aggregate fields as the T2MIR online evaluator."""

    per_episode = []
    for episode_index in range(episodes):
        selected = [row for row in rows if row["episode_index"] == episode_index]
        successes = sum(row["result"] == "success" for row in selected)
        falls = sum(row["result"] == "fall" for row in selected)
        per_episode.append(
            {
                "episode_index": episode_index,
                "episodes": len(selected),
                "successes": successes,
                "falls": falls,
                "timeouts": sum(row["result"] == "timeout" for row in selected),
                "success_rate": successes / max(len(selected), 1),
                "fall_rate": falls / max(len(selected), 1),
                "mean_return": sum(float(row["episode_return"]) for row in selected)
                / max(len(selected), 1),
                "mean_position_error": sum(float(row["position_error"]) for row in selected)
                / max(len(selected), 1),
                "mean_yaw_error": sum(float(row["yaw_error"]) for row in selected)
                / max(len(selected), 1),
            }
        )
    total_successes = sum(row["result"] == "success" for row in rows)
    total_falls = sum(row["result"] == "fall" for row in rows)
    return {
        "episodes_per_replica": episodes,
        "replicas": len(rows) // episodes,
        "total_episodes": len(rows),
        "success_rate": total_successes / max(len(rows), 1),
        "fall_rate": total_falls / max(len(rows), 1),
        "per_episode": per_episode,
        "adaptation_success_delta_last_minus_first": (
            per_episode[-1]["success_rate"] - per_episode[0]["success_rate"]
        ),
        "adaptation_return_delta_last_minus_first": (
            per_episode[-1]["mean_return"] - per_episode[0]["mean_return"]
        ),
    }


def no_context_prompt_protocol(
    *, policy_kind: str, action_semantics: str = "deterministic_actor_mean"
) -> dict[str, Any]:
    """Describe a deliberately non-adaptive baseline in report-compatible form."""

    if not policy_kind:
        raise ValueError("policy_kind must not be empty")
    return {
        "format_version": 1,
        "protocol": f"no-prompt-fixed-{policy_kind}",
        "prompt_horizon": 0,
        "window": None,
        "short_episode_policy": "not_applicable",
        "empty_prompt_policy": "always_zero_length",
        "vector_batching": "single_policy_batch",
        "action_semantics": action_semantics,
        "cross_reset_transitions_allowed": False,
        "online_prompt_updates_within_episode": False,
        "online_prompt_updates_between_episodes": False,
    }


def ppo_prompt_protocol() -> dict[str, Any]:
    """Backward-compatible PPO-specific prompt contract."""

    return no_context_prompt_protocol(policy_kind=POLICY_KIND)
