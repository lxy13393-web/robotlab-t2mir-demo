"""Pure NumPy actuator-side transforms for RobotLab G1 deployment.

The policy emits *raw* 37-dimensional actions in the PhysX policy order.  A
deployment target must keep three concepts separate:

* raw policy action -- stored in a T2MIR prompt;
* executed action -- after the optional first-order action-lag filter and used
  as the next observation's ``previous_action``;
* joint torque -- produced by the same default-offset position target and PD
  gains used by Isaac Lab's implicit actuators.

Keeping these stages explicit prevents a common but very damaging error where
the lagged action is written into the context as if it were the policy output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import DEFAULT_CONTRACT, DeploymentContract


def _finite_vector(
    value: np.ndarray | Sequence[float], width: int, name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


@dataclass(frozen=True)
class DynamicsParameters:
    """Deployment counterpart of one RobotLab hidden-dynamics profile."""

    action_lag: float = 0.0
    motor_strength: float = 1.0
    payload_kg: float = 0.0
    friction: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.action_lag, self.motor_strength, self.payload_kg, self.friction),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("dynamics parameters must be finite")
        if not 0.0 <= self.action_lag < 1.0:
            raise ValueError("action_lag must be in [0, 1)")
        if self.motor_strength <= 0.0:
            raise ValueError("motor_strength must be positive")
        if self.payload_kg < 0.0:
            raise ValueError("payload_kg must be non-negative")
        if self.friction <= 0.0:
            raise ValueError("friction must be positive")


class FirstOrderActionLag:
    """The exact filter used by ``pipeline/expert/play.py``.

    The first action of an episode is passed through unchanged.  Thereafter
    ``executed = (1 - lag) * raw + lag * previous_executed``.
    """

    def __init__(self, coefficient: float, action_dim: int = 37) -> None:
        if not np.isfinite(coefficient) or not 0.0 <= coefficient < 1.0:
            raise ValueError("coefficient must be finite and in [0, 1)")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.coefficient = float(coefficient)
        self.action_dim = int(action_dim)
        self._previous: np.ndarray | None = None

    @property
    def previous_executed_action(self) -> np.ndarray | None:
        return None if self._previous is None else self._previous.copy()

    def reset(self) -> None:
        self._previous = None

    def apply(self, raw_action: np.ndarray | Sequence[float]) -> np.ndarray:
        raw = _finite_vector(raw_action, self.action_dim, "raw_action")
        if self._previous is None:
            executed = raw.copy()
        else:
            executed = (
                np.float32(1.0 - self.coefficient) * raw
                + np.float32(self.coefficient) * self._previous
            )
        self._previous = np.ascontiguousarray(executed, dtype=np.float32)
        return self._previous.copy()


@dataclass(frozen=True)
class PDGains:
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("stiffness", "damping", "effort_limit"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim != 1 or not value.size:
                raise ValueError(f"{name} must be a non-empty vector")
            if not np.isfinite(value).all() or (value < 0.0).any():
                raise ValueError(f"{name} must contain finite non-negative values")
            arrays[name] = value.copy()
        if len({array.shape for array in arrays.values()}) != 1:
            raise ValueError("PD gain vectors must have identical shapes")
        object.__setattr__(self, "stiffness", arrays["stiffness"])
        object.__setattr__(self, "damping", arrays["damping"])
        object.__setattr__(self, "effort_limit", arrays["effort_limit"])


def robotlab_g1_pd_gains(
    contract: DeploymentContract = DEFAULT_CONTRACT,
) -> PDGains:
    """Return gains from the exact ``G1_MINIMAL_CFG`` training asset."""

    stiffness: list[float] = []
    damping: list[float] = []
    effort_limit: list[float] = []
    for name in contract.joint_names:
        if "ankle_" in name:
            stiffness.append(20.0)
            damping.append(2.0)
            effort_limit.append(20.0)
        elif name == "torso_joint" or "hip_pitch" in name or "knee_joint" in name:
            stiffness.append(200.0)
            damping.append(5.0)
            effort_limit.append(300.0)
        elif "hip_roll" in name or "hip_yaw" in name:
            stiffness.append(150.0)
            damping.append(5.0)
            effort_limit.append(300.0)
        else:
            stiffness.append(40.0)
            damping.append(10.0)
            effort_limit.append(300.0)
    return PDGains(
        np.asarray(stiffness, dtype=np.float32),
        np.asarray(damping, dtype=np.float32),
        np.asarray(effort_limit, dtype=np.float32),
    )


def robotlab_g1_joint_armatures(
    contract: DeploymentContract = DEFAULT_CONTRACT,
) -> np.ndarray:
    """Return the PhysX armatures from the recorded ``G1_MINIMAL_CFG`` run.

    Legs, feet, torso, shoulders and elbows use ``0.01``.  The fourteen hand
    joints use ``0.001``.  These values belong to the joint dynamics and are
    separate from the explicit position-PD stiffness/damping above.
    """

    finger_tokens = (
        "_zero_joint",
        "_one_joint",
        "_two_joint",
        "_three_joint",
        "_four_joint",
        "_five_joint",
        "_six_joint",
    )
    return np.asarray(
        [
            0.001 if any(token in name for token in finger_tokens) else 0.01
            for name in contract.joint_names
        ],
        # MuJoCo stores compiled joint dynamics as float64.  Keeping these
        # contract constants in float32 introduces a ~2e-10 representation
        # error (for example 0.01 -> 0.009999999776...), which makes the strict
        # parity gate reject an otherwise exact MJCF value.  Actions and PD
        # arithmetic are still converted to the runtime dtype by their
        # callers; the contract itself should preserve the decimal values.
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PDOutput:
    joint_targets: np.ndarray
    torques: np.ndarray


class PositionPDController:
    """Convert an executed RobotLab action into clipped joint torques."""

    def __init__(
        self,
        contract: DeploymentContract = DEFAULT_CONTRACT,
        gains: PDGains | None = None,
        *,
        motor_strength: float = 1.0,
        actuator_limits: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        if not np.isfinite(motor_strength) or motor_strength <= 0.0:
            raise ValueError("motor_strength must be finite and positive")
        self.contract = contract
        self.gains = gains or robotlab_g1_pd_gains(contract)
        expected = (contract.action_dim,)
        if self.gains.stiffness.shape != expected:
            raise ValueError(f"gain shape must be {expected}, got {self.gains.stiffness.shape}")

        # RobotLab's motor-strength profile scales effort limits, not gains or
        # the requested torque.  Preserve that exact semantic here.
        limits = self.gains.effort_limit * np.float32(motor_strength)
        if actuator_limits is not None:
            model_limits = np.asarray(actuator_limits, dtype=np.float32)
            if model_limits.shape != (contract.action_dim,):
                raise ValueError(
                    f"actuator_limits must have shape ({contract.action_dim},), "
                    f"got {model_limits.shape}"
                )
            if np.isnan(model_limits).any() or (model_limits <= 0.0).any():
                raise ValueError("actuator_limits must be positive or +Inf")
            limits = np.minimum(limits, model_limits)
        self.effective_effort_limit = np.asarray(limits, dtype=np.float32)

    def compute(
        self,
        executed_action: np.ndarray | Sequence[float],
        joint_position: np.ndarray | Sequence[float],
        joint_velocity: np.ndarray | Sequence[float],
    ) -> PDOutput:
        action = _finite_vector(executed_action, self.contract.action_dim, "executed_action")
        position = _finite_vector(joint_position, self.contract.action_dim, "joint_position")
        velocity = _finite_vector(joint_velocity, self.contract.action_dim, "joint_velocity")
        targets = self.contract.action_to_joint_targets(action)
        requested = self.gains.stiffness * (targets - position) - self.gains.damping * velocity
        torques = np.clip(
            requested, -self.effective_effort_limit, self.effective_effort_limit
        ).astype(np.float32, copy=False)
        return PDOutput(
            joint_targets=np.ascontiguousarray(targets, dtype=np.float32),
            torques=np.ascontiguousarray(torques, dtype=np.float32),
        )


__all__ = [
    "DynamicsParameters",
    "FirstOrderActionLag",
    "PDGains",
    "PDOutput",
    "PositionPDController",
    "robotlab_g1_joint_armatures",
    "robotlab_g1_pd_gains",
]
