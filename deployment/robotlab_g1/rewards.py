"""Reward providers for RobotLab G1 deployment.

The formal T2MIR prompt contains rewards, while a PPO policy does not consume
them.  Deployment therefore treats reward construction as an explicit,
versioned boundary instead of hiding it inside the MuJoCo loop.  The
``ObservableVelocityReward`` below is intentionally named a *proxy*: it uses
only quantities that are available from MuJoCo or ROS 2 and must not be
silently presented as bitwise-equivalent to Isaac Lab's full reward manager.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Protocol

import numpy as np

from .contract import DEFAULT_CONTRACT


REWARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RewardInput:
    """Quantities needed by a deployment-time reward provider."""

    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray
    command: np.ndarray
    action: np.ndarray
    previous_action: np.ndarray
    joint_torque: np.ndarray
    terminated: bool = False

    def __post_init__(self) -> None:
        expected = {
            "base_linear_velocity": (3,),
            "base_angular_velocity": (3,),
            "projected_gravity": (3,),
            "command": (3,),
            "action": (37,),
            "previous_action": (37,),
            "joint_torque": (37,),
        }
        for field, shape in expected.items():
            value = np.asarray(getattr(self, field), dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{field} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{field} contains NaN or Inf")
            object.__setattr__(self, field, value)


class RewardProvider(Protocol):
    """Interface shared by MuJoCo-computed and externally supplied rewards."""

    @property
    def schema_id(self) -> str: ...

    def compute(self, value: RewardInput) -> float: ...


@dataclass(frozen=True)
class ObservableVelocityRewardConfig:
    """Observable subset of the RobotLab G1 velocity reward.

    The tracking and regularization weights mirror the current training task
    where the required signals are available.  Terms that require Isaac-only
    contact/sensor semantics are deliberately omitted and listed in
    ``omitted_terms`` for provenance.
    """

    policy_dt: float = 0.02
    linear_tracking_weight: float = 1.0
    yaw_tracking_weight: float = 1.0
    tracking_std: float = 0.5
    vertical_velocity_weight: float = -0.2
    flat_orientation_weight: float = -1.0
    action_rate_weight: float = -0.005
    hip_knee_torque_weight: float = -2.0e-6
    termination_weight: float = -200.0
    omitted_terms: tuple[str, ...] = (
        "feet_air_time",
        "feet_slide",
        "joint_limit",
        "joint_deviation",
        "dof_acceleration",
    )

    def __post_init__(self) -> None:
        if self.policy_dt <= 0 or self.tracking_std <= 0:
            raise ValueError("policy_dt and tracking_std must be positive")


class ObservableVelocityReward:
    """Deterministic reward proxy suitable for deployment plumbing tests."""

    def __init__(self, config: ObservableVelocityRewardConfig | None = None) -> None:
        self.config = config or ObservableVelocityRewardConfig()
        payload = {
            "format_version": REWARD_SCHEMA_VERSION,
            "name": type(self).__name__,
            "config": asdict(self.config),
            "semantics": "observable-proxy-not-bitwise-isaac",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self._schema_id = f"robotlab-g1-reward-v1:{hashlib.sha256(encoded).hexdigest()}"

    @property
    def schema_id(self) -> str:
        return self._schema_id

    def components(self, value: RewardInput) -> dict[str, float]:
        cfg = self.config
        linear_error = value.command[:2] - value.base_linear_velocity[:2]
        yaw_error = float(value.command[2] - value.base_angular_velocity[2])
        linear_tracking = np.exp(-float(linear_error @ linear_error) / cfg.tracking_std**2)
        yaw_tracking = np.exp(-(yaw_error * yaw_error) / cfg.tracking_std**2)
        vertical_velocity = float(value.base_linear_velocity[2] ** 2)
        flat_orientation = float(value.projected_gravity[:2] @ value.projected_gravity[:2])
        action_delta = value.action - value.previous_action
        action_rate = float(action_delta @ action_delta)
        hip_knee_indices = np.asarray(
            [
                index
                for index, name in enumerate(DEFAULT_CONTRACT.joint_names)
                if "_hip_" in name or "_knee_joint" in name
            ],
            dtype=np.int64,
        )
        selected_torque = value.joint_torque[hip_knee_indices]
        torque_cost = float(selected_torque @ selected_torque)
        return {
            "linear_tracking": cfg.policy_dt * cfg.linear_tracking_weight * linear_tracking,
            "yaw_tracking": cfg.policy_dt * cfg.yaw_tracking_weight * yaw_tracking,
            "vertical_velocity": cfg.policy_dt * cfg.vertical_velocity_weight * vertical_velocity,
            "flat_orientation": cfg.policy_dt * cfg.flat_orientation_weight * flat_orientation,
            "action_rate": cfg.policy_dt * cfg.action_rate_weight * action_rate,
            "hip_knee_torque": cfg.policy_dt * cfg.hip_knee_torque_weight * torque_cost,
            "termination": cfg.policy_dt * cfg.termination_weight * float(value.terminated),
        }

    def compute(self, value: RewardInput) -> float:
        reward = float(sum(self.components(value).values()))
        if not np.isfinite(reward):
            raise FloatingPointError("deployment reward is NaN or Inf")
        return reward


class ExternalReward:
    """One-shot externally supplied reward, useful for a ROS 2 supervisor."""

    def __init__(self, schema_id: str) -> None:
        if not schema_id:
            raise ValueError("schema_id must be non-empty")
        self._schema_id = schema_id
        self._pending: float | None = None

    @property
    def schema_id(self) -> str:
        return self._schema_id

    def update(self, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError("external reward must be finite")
        self._pending = float(reward)

    def compute(self, value: RewardInput) -> float:  # noqa: ARG002 - protocol compatibility
        if self._pending is None:
            raise RuntimeError("no external reward has been supplied for this transition")
        reward = self._pending
        self._pending = None
        return reward
