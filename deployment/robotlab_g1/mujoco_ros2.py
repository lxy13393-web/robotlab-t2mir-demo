"""In-process bridge between the ROS-independent policy core and MuJoCo.

``Ros2PolicyCore`` owns policy/context timing.  ``MujocoRos2Plant`` owns only
physics and actuator transforms.  The small handler returned by
``make_mujoco_step_handler`` feeds post-step reward, termination and the lagged
executed action back to the core before the next 50 Hz policy tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import DEFAULT_CONTRACT, DeploymentContract
from .control import (
    DynamicsParameters,
    FirstOrderActionLag,
    PDGains,
    PositionPDController,
    robotlab_g1_pd_gains,
)
from .mujoco_runner import MujocoBindings, MujocoEpisodeConfig
from .observation import RobotState, projected_gravity_from_quaternion
from .rewards import ObservableVelocityReward, RewardInput, RewardProvider
from .ros2_node import PolicyStep, Ros2PolicyCore


@dataclass(frozen=True)
class MujocoPlantStep:
    reward: float
    terminated: bool
    truncated: bool
    termination_reason: str
    executed_action: np.ndarray
    joint_torque: np.ndarray


class MujocoRos2Plant:
    """A single 37-DoF MuJoCo robot stepped once per ROS policy callback."""

    def __init__(
        self,
        mujoco_module: Any,
        model: Any,
        data: Any,
        *,
        contract: DeploymentContract = DEFAULT_CONTRACT,
        dynamics: DynamicsParameters | None = None,
        reward_provider: RewardProvider | None = None,
        episode_config: MujocoEpisodeConfig | None = None,
        integration_dt: float | None = None,
        stiffness_scale: float = 1.0,
        damping_scale: float = 1.0,
        actuator_mode: str = "explicit_pd",
    ) -> None:
        self.mj = mujoco_module
        self.model = model
        self.data = data
        self.contract = contract
        self.dynamics = dynamics or DynamicsParameters()
        self.reward_provider = reward_provider or ObservableVelocityReward()
        self.episode_config = episode_config or MujocoEpisodeConfig()
        self.bindings = MujocoBindings(
            mujoco_module, model, contract=contract, integration_dt=integration_dt
        )
        self.action_filter = FirstOrderActionLag(
            self.dynamics.action_lag, contract.action_dim
        )
        if not np.isfinite(stiffness_scale) or stiffness_scale <= 0.0:
            raise ValueError("stiffness_scale must be finite and positive")
        if not np.isfinite(damping_scale) or damping_scale <= 0.0:
            raise ValueError("damping_scale must be finite and positive")
        base_gains = robotlab_g1_pd_gains(contract)
        gains = PDGains(
            base_gains.stiffness * np.float32(stiffness_scale),
            base_gains.damping * np.float32(damping_scale),
            base_gains.effort_limit,
        )
        self.stiffness_scale = float(stiffness_scale)
        self.damping_scale = float(damping_scale)
        self.controller = PositionPDController(
            contract, gains=gains, motor_strength=self.dynamics.motor_strength
        )
        if actuator_mode not in {"explicit_pd", "implicit_pd"}:
            raise ValueError("actuator_mode must be 'explicit_pd' or 'implicit_pd'")
        self.actuator_mode = actuator_mode
        if actuator_mode == "implicit_pd":
            self.bindings.configure_implicit_position_pd(
                self.controller.gains, self.controller.effective_effort_limit
            )
        else:
            self.bindings.set_robotlab_effort_limits(self.controller.effective_effort_limit)
        self.previous_executed_action = np.zeros(contract.action_dim, dtype=np.float32)
        self.policy_steps = 0
        self.reset()

    @classmethod
    def from_xml(cls, model_path: str, **kwargs: Any) -> "MujocoRos2Plant":
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - deployment-host dependency
            raise RuntimeError(
                "MuJoCo ROS bridge requires the official 'mujoco' Python package"
            ) from exc
        model = mujoco.MjModel.from_xml_path(str(model_path))
        return cls(mujoco, model, mujoco.MjData(model), **kwargs)

    def reset(self) -> None:
        self.bindings.reset_to_policy_pose(self.data)
        self.bindings.apply_dynamics(self.data, self.dynamics)
        self.action_filter.reset()
        self.previous_executed_action.fill(0.0)
        self.policy_steps = 0

    def state(self) -> RobotState:
        linear, angular, quaternion = self.bindings.base_kinematics(self.data)
        return RobotState(
            base_linear_velocity=linear,
            base_angular_velocity=angular,
            projected_gravity=projected_gravity_from_quaternion(quaternion),
            # Ros2PolicyCore owns and replaces these two fields.
            velocity_command=np.zeros(3, dtype=np.float32),
            joint_position=self.bindings.joint_position(self.data),
            joint_velocity=self.bindings.joint_velocity(self.data),
            previous_action=self.previous_executed_action.copy(),
            joint_names=self.contract.joint_names,
        )

    def _termination(self, projected_gravity: np.ndarray) -> tuple[bool, str]:
        if self.bindings.base_height(self.data) < self.episode_config.fall_height:
            return True, "base_height"
        if float(projected_gravity[2]) > self.episode_config.fall_projected_gravity_z:
            return True, "orientation"
        return False, "running"

    def apply_policy_step(self, step: PolicyStep) -> MujocoPlantStep:
        raw_action = np.asarray(step.policy_action, dtype=np.float32)
        executed_action = self.action_filter.apply(raw_action)
        previous_executed = self.previous_executed_action.copy()
        torque = np.zeros(self.contract.action_dim, dtype=np.float32)
        targets = None
        if self.actuator_mode == "implicit_pd":
            targets = self.contract.action_to_joint_targets(executed_action)
        for _ in range(self.bindings.metadata.physics_steps_per_policy_step):
            if targets is not None:
                self.bindings.write_joint_targets(self.data, targets)
            else:
                control = self.controller.compute(
                    executed_action,
                    self.bindings.joint_position(self.data),
                    self.bindings.joint_velocity(self.data),
                )
                torque = control.torques
                self.bindings.write_torques(self.data, torque)
            self.mj.mj_step(self.model, self.data)
            if targets is not None:
                torque = self.bindings.actuator_forces(self.data)

        post_state = self.state()
        projected_gravity = np.asarray(post_state.projected_gravity, dtype=np.float32)
        terminated, reason = self._termination(projected_gravity)
        self.policy_steps += 1
        truncated = (
            not terminated and self.policy_steps >= self.episode_config.max_policy_steps
        )
        if truncated:
            reason = "time_limit"
        reward = self.reward_provider.compute(
            RewardInput(
                base_linear_velocity=np.asarray(post_state.base_linear_velocity),
                base_angular_velocity=np.asarray(post_state.base_angular_velocity),
                projected_gravity=projected_gravity,
                command=step.velocity_command,
                action=executed_action,
                previous_action=previous_executed,
                joint_torque=torque,
                terminated=terminated,
            )
        )
        self.previous_executed_action = executed_action.copy()
        return MujocoPlantStep(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            executed_action=executed_action.copy(),
            joint_torque=torque.copy(),
        )


def make_mujoco_step_handler(
    core: Ros2PolicyCore,
    plant: MujocoRos2Plant,
):
    """Return the callback that closes the ROS2→MuJoCo→context loop."""

    def handle(step: PolicyStep) -> MujocoPlantStep:
        result = plant.apply_policy_step(step)
        core.update_executed_action(result.executed_action)
        core.update_reward(result.reward)
        if result.terminated or result.truncated:
            core.notify_done(True, f"mujoco:{result.termination_reason}")
            # The boundary is applied by the core at the next tick, after the
            # just-computed reward is recorded.  Reset physics immediately so
            # that the next state sample belongs to the new episode.
            plant.reset()
        return result

    return handle


__all__ = [
    "MujocoPlantStep",
    "MujocoRos2Plant",
    "make_mujoco_step_handler",
]
