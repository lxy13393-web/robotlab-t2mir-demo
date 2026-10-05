"""Pure NumPy construction of the 123-dimensional RobotLab observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import DEFAULT_CONTRACT, DeploymentContract


def _vector(value: np.ndarray | Sequence[float], width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class RobotState:
    """One policy-time state sample.

    Base velocities and gravity are expressed in the base/body frame, matching
    Isaac Lab's ``base_lin_vel``, ``base_ang_vel`` and ``projected_gravity``
    observation terms.  Joint vectors may use any order when ``joint_names`` is
    supplied; otherwise they are required to already use policy order.
    """

    base_linear_velocity: np.ndarray | Sequence[float]
    base_angular_velocity: np.ndarray | Sequence[float]
    projected_gravity: np.ndarray | Sequence[float]
    velocity_command: np.ndarray | Sequence[float]
    joint_position: np.ndarray | Sequence[float]
    joint_velocity: np.ndarray | Sequence[float]
    previous_action: np.ndarray | Sequence[float]
    joint_names: Sequence[str] | None = None


class ObservationBuilder:
    """Build deterministic, noise-free deployment observations."""

    def __init__(self, contract: DeploymentContract = DEFAULT_CONTRACT) -> None:
        self.contract = contract

    def build(self, state: RobotState) -> np.ndarray:
        action_dim = self.contract.action_dim
        base_linear_velocity = _vector(state.base_linear_velocity, 3, "base_linear_velocity")
        base_angular_velocity = _vector(state.base_angular_velocity, 3, "base_angular_velocity")
        projected_gravity = _vector(state.projected_gravity, 3, "projected_gravity")
        velocity_command = _vector(state.velocity_command, 3, "velocity_command")
        joint_position = _vector(state.joint_position, action_dim, "joint_position")
        joint_velocity = _vector(state.joint_velocity, action_dim, "joint_velocity")
        previous_action = _vector(state.previous_action, action_dim, "previous_action")

        if state.joint_names is not None:
            indices = self.contract.reorder_indices(state.joint_names)
            joint_position = joint_position[indices]
            joint_velocity = joint_velocity[indices]

        defaults = np.asarray(self.contract.default_joint_positions, dtype=np.float32)
        result = np.concatenate(
            (
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                velocity_command,
                joint_position - defaults,
                joint_velocity,  # default joint velocity is zero in G1_MINIMAL_CFG
                previous_action,
            ),
            dtype=np.float32,
        )
        if result.shape != (self.contract.state_dim,):
            raise RuntimeError(
                f"constructed observation has shape {result.shape}; expected {(self.contract.state_dim,)}"
            )
        return result


def rotate_world_to_body(
    vector_world: np.ndarray | Sequence[float], quaternion_wxyz: np.ndarray | Sequence[float]
) -> np.ndarray:
    """Rotate a world-frame vector into the body frame.

    The quaternion is the body orientation in the world frame and uses the
    MuJoCo/Isaac ``(w, x, y, z)`` convention.
    """

    vector = _vector(vector_world, 3, "vector_world").astype(np.float64)
    quaternion = _vector(quaternion_wxyz, 4, "quaternion_wxyz").astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = quaternion / norm
    # World-from-body rotation.  Its transpose maps world vectors to body.
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return (rotation.T @ vector).astype(np.float32)


def projected_gravity_from_quaternion(
    quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return Isaac Lab's unit gravity direction in the base frame."""

    return rotate_world_to_body((0.0, 0.0, -1.0), quaternion_wxyz)


__all__ = [
    "ObservationBuilder",
    "RobotState",
    "projected_gravity_from_quaternion",
    "rotate_world_to_body",
]
