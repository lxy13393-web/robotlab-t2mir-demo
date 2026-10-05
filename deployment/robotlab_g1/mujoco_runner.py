"""Minimal, model-strict MuJoCo runner for RobotLab's 37-DoF G1 policy.

This module imports MuJoCo lazily, so contracts, ROS plumbing and unit tests can
be developed on machines without the Python ``mujoco`` package.  At runtime it
refuses 23/29-DoF G1 assets and constructs explicit joint/actuator address maps;
no simulator-native ordering is trusted.

The runner is intentionally small.  It proves the deployment boundary and
supports repeatable velocity-command episodes; it is not yet a claim of
Isaac-to-MuJoCo performance parity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .contract import DEFAULT_CONTRACT, DeploymentContract
from .control import (
    DynamicsParameters,
    FirstOrderActionLag,
    PDGains,
    PositionPDController,
    robotlab_g1_pd_gains,
    robotlab_g1_joint_armatures,
)
from .observation import ObservationBuilder, RobotState, projected_gravity_from_quaternion
from .policy_backends import PolicyBackend
from .rewards import ObservableVelocityReward, RewardInput, RewardProvider


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_command(value: np.ndarray | Sequence[float]) -> np.ndarray:
    command = np.asarray(value, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"velocity command must have shape (3,), got {command.shape}")
    if not np.isfinite(command).all():
        raise ValueError("velocity command contains NaN or Inf")
    return command


@dataclass(frozen=True)
class MujocoBindingMetadata:
    joint_ids: tuple[int, ...]
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    actuator_limits: tuple[float, ...]
    base_body_id: int
    payload_body_id: int
    free_joint_id: int
    free_qpos_address: int
    original_timestep: float
    deployed_timestep: float
    physics_steps_per_policy_step: int


class MujocoBindings:
    """Validated address map from an ``MjModel`` into policy order."""

    def __init__(
        self,
        mujoco_module: Any,
        model: Any,
        *,
        contract: DeploymentContract = DEFAULT_CONTRACT,
        payload_body_name: str = "torso_link",
        synchronize_timestep: bool = True,
        integration_dt: float | None = None,
    ) -> None:
        self.mj = mujoco_module
        self.model = model
        self.contract = contract

        joint_ids = tuple(self._required_id(self.mj.mjtObj.mjOBJ_JOINT, name) for name in contract.joint_names)
        hinge_type = int(self.mj.mjtJoint.mjJNT_HINGE)
        non_hinge = [
            name
            for name, joint_id in zip(contract.joint_names, joint_ids, strict=True)
            if int(model.jnt_type[joint_id]) != hinge_type
        ]
        if non_hinge:
            raise ValueError(f"policy joints must be one-DoF hinge joints: {non_hinge}")

        free_type = int(self.mj.mjtJoint.mjJNT_FREE)
        free_joint_ids = [
            index for index in range(int(model.njnt)) if int(model.jnt_type[index]) == free_type
        ]
        if len(free_joint_ids) != 1:
            raise ValueError(f"expected exactly one free joint, found {len(free_joint_ids)}")
        free_joint_id = free_joint_ids[0]
        base_body_id = int(model.jnt_bodyid[free_joint_id])
        payload_body_id = self._required_id(self.mj.mjtObj.mjOBJ_BODY, payload_body_name)

        qpos_addresses = tuple(int(model.jnt_qposadr[index]) for index in joint_ids)
        dof_addresses = tuple(int(model.jnt_dofadr[index]) for index in joint_ids)
        self._validate_joint_dynamics(dof_addresses)
        actuator_ids: list[int] = []
        for name, joint_id in zip(contract.joint_names, joint_ids, strict=True):
            matches = np.flatnonzero(np.asarray(model.actuator_trnid)[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(
                    f"joint {name!r} must have exactly one direct actuator, found {len(matches)}"
                )
            actuator_ids.append(int(matches[0]))
        if len(set(actuator_ids)) != contract.action_dim:
            raise ValueError("policy joints do not map one-to-one onto MuJoCo actuators")

        # ``data.ctrl`` is interpreted very differently by a position servo
        # and a torque motor.  The deployment controller writes torque, so a
        # name/direct-joint match is not sufficient: require MuJoCo's exact
        # unit-gain, unit-gear, stateless torque-motor semantics.
        self._validate_torque_motor_semantics(tuple(actuator_ids))

        actuator_limits: list[float] = []
        for actuator_id in actuator_ids:
            limited = bool(model.actuator_ctrllimited[actuator_id])
            if not limited:
                actuator_limits.append(float("inf"))
                continue
            low, high = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64)
            limit = min(abs(float(low)), abs(float(high)))
            if not np.isfinite(limit) or limit <= 0.0:
                raise ValueError(f"actuator {actuator_id} has invalid ctrlrange [{low}, {high}]")
            actuator_limits.append(limit)

        original_timestep = float(model.opt.timestep)
        if not np.isfinite(original_timestep) or original_timestep <= 0.0:
            raise ValueError(f"MuJoCo model timestep is invalid: {original_timestep}")
        target_timestep = contract.physics_dt if integration_dt is None else float(integration_dt)
        if not np.isfinite(target_timestep) or target_timestep <= 0.0:
            raise ValueError("integration_dt must be finite and positive")
        ratio = contract.policy_dt / target_timestep
        physics_steps_per_policy_step = int(round(ratio))
        if physics_steps_per_policy_step <= 0 or not np.isclose(
            ratio, physics_steps_per_policy_step, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "integration_dt must divide the policy period exactly: "
                f"policy_dt={contract.policy_dt}, integration_dt={target_timestep}"
            )
        if synchronize_timestep:
            model.opt.timestep = target_timestep
        elif not np.isclose(original_timestep, contract.physics_dt, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"MuJoCo timestep={original_timestep} differs from training "
                f"timestep={contract.physics_dt}"
            )

        self._baseline_body_mass = np.asarray(model.body_mass, dtype=np.float64).copy()
        self._baseline_body_inertia = np.asarray(model.body_inertia, dtype=np.float64).copy()
        self._baseline_geom_friction = np.asarray(model.geom_friction, dtype=np.float64).copy()
        self.metadata = MujocoBindingMetadata(
            joint_ids=joint_ids,
            qpos_addresses=qpos_addresses,
            dof_addresses=dof_addresses,
            actuator_ids=tuple(actuator_ids),
            actuator_limits=tuple(actuator_limits),
            base_body_id=base_body_id,
            payload_body_id=payload_body_id,
            free_joint_id=free_joint_id,
            free_qpos_address=int(model.jnt_qposadr[free_joint_id]),
            original_timestep=original_timestep,
            deployed_timestep=float(model.opt.timestep),
            physics_steps_per_policy_step=physics_steps_per_policy_step,
        )

    def _enum(self, namespace: str, member: str) -> int:
        enum_type = getattr(self.mj, namespace, None)
        value = None if enum_type is None else getattr(enum_type, member, None)
        if value is None:
            raise ValueError(
                f"MuJoCo bindings do not expose {namespace}.{member}; "
                "cannot prove torque-motor semantics"
            )
        return int(value)

    def _validate_torque_motor_semantics(self, actuator_ids: tuple[int, ...]) -> None:
        required_arrays = (
            "actuator_trntype",
            "actuator_dyntype",
            "actuator_gaintype",
            "actuator_biastype",
            "actuator_gainprm",
            "actuator_gear",
            "actuator_forcelimited",
        )
        missing = tuple(name for name in required_arrays if not hasattr(self.model, name))
        if missing:
            raise ValueError(
                f"MuJoCo model lacks actuator semantic arrays {list(missing)}; "
                "refusing unverified torque control"
            )
        expected = {
            "actuator_trntype": self._enum("mjtTrn", "mjTRN_JOINT"),
            "actuator_dyntype": self._enum("mjtDyn", "mjDYN_NONE"),
            "actuator_gaintype": self._enum("mjtGain", "mjGAIN_FIXED"),
            "actuator_biastype": self._enum("mjtBias", "mjBIAS_NONE"),
        }
        errors: list[str] = []
        for actuator_id in actuator_ids:
            for array_name, expected_value in expected.items():
                actual = int(getattr(self.model, array_name)[actuator_id])
                if actual != expected_value:
                    errors.append(
                        f"actuator {actuator_id} {array_name}={actual}, expected {expected_value}"
                    )
            gain = np.asarray(self.model.actuator_gainprm[actuator_id], dtype=np.float64)
            gear = np.asarray(self.model.actuator_gear[actuator_id], dtype=np.float64)
            if gain.size < 1 or not np.isclose(gain[0], 1.0, rtol=0.0, atol=1.0e-12):
                errors.append(f"actuator {actuator_id} has non-unit fixed gain")
            expected_gear = np.zeros_like(gear)
            if expected_gear.size:
                expected_gear[0] = 1.0
            if gear.size < 1 or not np.allclose(gear, expected_gear, rtol=0.0, atol=1.0e-12):
                errors.append(f"actuator {actuator_id} has non-unit direct gear {gear.tolist()}")
            if bool(self.model.actuator_forcelimited[actuator_id]):
                errors.append(
                    f"actuator {actuator_id} has an extra force clamp; torque parity is ambiguous"
                )
        if hasattr(self.model, "jnt_actfrclimited"):
            for joint_id in self.metadata_joint_ids_for_validation(actuator_ids):
                if bool(self.model.jnt_actfrclimited[joint_id]):
                    errors.append(
                        f"joint {joint_id} has an extra actuator-force clamp; "
                        "torque parity is ambiguous"
                    )
        if errors:
            preview = "; ".join(errors[:8])
            remainder = "" if len(errors) <= 8 else f"; ... {len(errors) - 8} more"
            raise ValueError(
                "policy actuators are not direct unit-gear torque motors: "
                f"{preview}{remainder}"
            )

    def metadata_joint_ids_for_validation(
        self, actuator_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Return direct transmission joint ids before metadata is constructed."""

        return tuple(int(self.model.actuator_trnid[index, 0]) for index in actuator_ids)

    def _validate_joint_dynamics(self, dof_addresses: tuple[int, ...]) -> None:
        required_arrays = ("dof_armature", "dof_damping", "dof_frictionloss")
        missing = tuple(name for name in required_arrays if not hasattr(self.model, name))
        if missing:
            raise ValueError(
                f"MuJoCo model lacks joint-dynamics arrays {list(missing)}; "
                "cannot prove RobotLab actuator parity"
            )
        indices = np.asarray(dof_addresses, dtype=np.int64)
        actual_armature = np.asarray(self.model.dof_armature, dtype=np.float64)[indices]
        expected_armature = robotlab_g1_joint_armatures(self.contract).astype(np.float64)
        actual_damping = np.asarray(self.model.dof_damping, dtype=np.float64)[indices]
        actual_friction = np.asarray(self.model.dof_frictionloss, dtype=np.float64)[indices]
        errors: list[str] = []
        if not np.allclose(
            actual_armature, expected_armature, rtol=0.0, atol=1.0e-12
        ):
            errors.append("armature differs from recorded G1_MINIMAL_CFG")
        if not np.allclose(actual_damping, 0.0, rtol=0.0, atol=1.0e-12):
            errors.append("passive joint damping is nonzero (external PD would double it)")
        if not np.allclose(actual_friction, 0.0, rtol=0.0, atol=1.0e-12):
            errors.append("joint frictionloss is nonzero")
        if errors:
            raise ValueError("joint dynamics do not match RobotLab: " + "; ".join(errors))

    def _required_id(self, object_type: Any, name: str) -> int:
        object_id = int(self.mj.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing required object {name!r}")
        return object_id

    @property
    def qpos_addresses(self) -> np.ndarray:
        return np.asarray(self.metadata.qpos_addresses, dtype=np.int64)

    @property
    def dof_addresses(self) -> np.ndarray:
        return np.asarray(self.metadata.dof_addresses, dtype=np.int64)

    @property
    def actuator_ids(self) -> np.ndarray:
        return np.asarray(self.metadata.actuator_ids, dtype=np.int64)

    @property
    def actuator_limits(self) -> np.ndarray:
        return np.asarray(self.metadata.actuator_limits, dtype=np.float32)

    def joint_position(self, data: Any) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_addresses], dtype=np.float32).copy()

    def joint_velocity(self, data: Any) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_addresses], dtype=np.float32).copy()

    def base_kinematics(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return actor-body-frame COM velocities and world orientation.

        ``mjOBJ_BODY`` local velocity is expressed in the inertial/COM frame,
        whose orientation may differ from the actor body frame used by
        ``xquat`` and projected gravity.  Read COM velocity in world axes and
        rotate it with the actor body's ``xmat`` so all observation terms use
        one frame.
        """

        velocity = np.zeros(6, dtype=np.float64)
        self.mj.mj_objectVelocity(
            self.model,
            data,
            self.mj.mjtObj.mjOBJ_BODY,
            self.metadata.base_body_id,
            velocity,
            0,  # world coordinates at the body's COM
        )
        quaternion = np.asarray(data.xquat[self.metadata.base_body_id], dtype=np.float32).copy()
        body_to_world = np.asarray(
            data.xmat[self.metadata.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        if not np.isfinite(body_to_world).all():
            raise ValueError("MuJoCo body rotation contains NaN or Inf")
        # mj_objectVelocity uses [angular, linear] spatial-vector order.
        angular_body = body_to_world.T @ velocity[:3]
        linear_body = body_to_world.T @ velocity[3:]
        return linear_body.astype(np.float32), angular_body.astype(np.float32), quaternion

    def base_height(self, data: Any) -> float:
        return float(data.xpos[self.metadata.base_body_id, 2])

    def reset_to_policy_pose(self, data: Any) -> None:
        self.mj.mj_resetData(self.model, data)
        free_qpos = self.metadata.free_qpos_address
        data.qpos[free_qpos : free_qpos + 3] = np.asarray(
            self.contract.initial_root_position, dtype=np.float64
        )
        data.qpos[free_qpos + 3 : free_qpos + 7] = np.asarray(
            self.contract.initial_root_quaternion_wxyz, dtype=np.float64
        )
        data.qpos[self.qpos_addresses] = self.policy_reset_joint_positions()
        data.qvel[self.dof_addresses] = 0.0
        self.mj.mj_forward(self.model, data)

    def policy_reset_joint_positions(self) -> np.ndarray:
        """Return RobotLab's realized reset pose in policy joint order.

        RobotLab resets joints to the configured default and then clamps that
        value to the articulation's *soft* position limits.  G1's default is
        zero for four hand joints, while the 0.9 soft-limit interval excludes
        zero; PhysX therefore starts those joints at +/-0.092 rad.  MuJoCo's
        hard ranges include zero, so copying only the configured defaults
        creates an observation and policy-action mismatch at step zero.

        The policy observation/action reference remains the configured
        default.  This method affects only the realized episode reset pose.
        """

        positions = np.asarray(
            self.contract.default_joint_positions, dtype=np.float64
        ).copy()
        if not hasattr(self.model, "jnt_limited") or not hasattr(self.model, "jnt_range"):
            return positions
        soft_factor = 0.9
        for policy_index, joint_id in enumerate(self.metadata.joint_ids):
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            low, high = np.asarray(self.model.jnt_range[joint_id], dtype=np.float64)
            center = 0.5 * (low + high)
            half_range = 0.5 * (high - low) * soft_factor
            positions[policy_index] = np.clip(
                positions[policy_index], center - half_range, center + half_range
            )
        return positions

    def apply_dynamics(self, data: Any, dynamics: DynamicsParameters) -> None:
        """Apply the four RobotLab profile parameters without accumulating edits."""

        self.model.body_mass[:] = self._baseline_body_mass
        self.model.body_inertia[:] = self._baseline_body_inertia
        self.model.geom_friction[:] = self._baseline_geom_friction
        payload_body = self.metadata.payload_body_id
        baseline_mass = float(self._baseline_body_mass[payload_body])
        if baseline_mass <= 0.0:
            raise ValueError("payload body must have positive baseline mass")
        deployed_mass = baseline_mass + dynamics.payload_kg
        self.model.body_mass[payload_body] = deployed_mass
        # Isaac Lab's randomize_rigid_body_mass(recompute_inertia=True) scales
        # the default inertia with the new/default mass ratio.  Match it here.
        self.model.body_inertia[payload_body] = (
            self._baseline_body_inertia[payload_body] * (deployed_mass / baseline_mass)
        )
        # MuJoCo dynamically combines equal-priority geom friction using the
        # element-wise maximum.  Setting only the robot to 0.65 while the floor
        # remains 1.0 would therefore leave the *actual* contact at 1.0 and make
        # the benchmark knob a no-op.  The minimum deployment scene is a flat
        # robot/floor world, so set the sliding coefficient on every geom.  The
        # torsional/rolling coefficients retain the pinned asset values.
        self.model.geom_friction[:, 0] = dynamics.friction
        # ``mj_setConst`` recomputes reference-configuration constants and
        # therefore must see ``model.qpos0``.  ``data`` is normally already at
        # RobotLab's policy reset pose here, which is not guaranteed to equal
        # the source MJCF reference pose.  Passing that live episode data can
        # silently bake the policy pose into MuJoCo's kinematic constants and
        # change contacts even for a nominal no-op dynamics profile.  Use a
        # fresh data object (initialized at qpos0) for the constant update and
        # keep the live episode state untouched.
        reference_data = self.mj.MjData(self.model)
        self.mj.mj_setConst(self.model, reference_data)
        self.mj.mj_forward(self.model, data)

    def set_robotlab_effort_limits(
        self, effort_limits_policy_order: np.ndarray | Sequence[float]
    ) -> None:
        """Install Isaac-training effort limits on the MuJoCo torque motors.

        The published Unitree MJCF contains physical ``ctrlrange`` values that
        are often much lower than ``G1_MINIMAL_CFG``.  Leaving those ranges in
        place would silently erase the 0.8/0.9/1.0 motor-strength benchmark
        (for example every hip would clamp at the same 88 Nm).  This runner is
        a RobotLab sim-to-sim benchmark, so its actuator gate must preserve the
        training configuration.  Real-hardware safety limits remain the
        responsibility of the hardware driver.
        """

        limits = np.asarray(effort_limits_policy_order, dtype=np.float64)
        if limits.shape != (self.contract.action_dim,):
            raise ValueError(
                f"effort limits must have shape ({self.contract.action_dim},), got {limits.shape}"
            )
        if not np.isfinite(limits).all() or (limits <= 0.0).any():
            raise ValueError("effort limits must be finite and positive")
        ids = self.actuator_ids
        self.model.actuator_ctrllimited[ids] = True
        self.model.actuator_ctrlrange[ids, 0] = -limits
        self.model.actuator_ctrlrange[ids, 1] = limits

    def configure_joint_limit_solver(self, time_constant: float) -> None:
        """Make MuJoCo joint stops approximate PhysX's rigid articulation limits."""

        value = float(time_constant)
        minimum = 2.0 * float(self.model.opt.timestep)
        if not np.isfinite(value) or value < minimum:
            raise ValueError(
                "joint limit time constant must be finite and at least twice "
                f"the integration step ({minimum})"
            )
        if not hasattr(self.model, "jnt_solref") or not hasattr(self.model, "jnt_limited"):
            raise ValueError("MuJoCo model lacks joint-limit solver parameters")
        for joint_id in self.metadata.joint_ids:
            if bool(self.model.jnt_limited[joint_id]):
                self.model.jnt_solref[joint_id, 0] = value

    def configure_implicit_position_pd(
        self,
        gains: PDGains,
        effort_limits: np.ndarray | Sequence[float],
    ) -> None:
        """Configure native implicit position drives matching PhysX Kp/Kd.

        The source asset remains a strict direct-torque asset and is validated
        before this runtime conversion.  MuJoCo can then linearize the affine
        position/velocity actuator inside ``implicitfast``; this avoids the
        unstable explicit integration of the 0.001-armature hand drives.
        """

        limits = np.asarray(effort_limits, dtype=np.float64)
        expected = (self.contract.action_dim,)
        if gains.stiffness.shape != expected or limits.shape != expected:
            raise ValueError("implicit PD gains/limits do not match the policy action dimension")
        if not np.isfinite(limits).all() or (limits <= 0.0).any():
            raise ValueError("implicit PD effort limits must be finite and positive")
        required = ("actuator_biasprm", "actuator_forcerange")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(f"MuJoCo model lacks implicit actuator arrays: {missing}")
        ids = self.actuator_ids
        self.model.opt.integrator = self.mj.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.actuator_gaintype[ids] = self.mj.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[ids] = self.mj.mjtBias.mjBIAS_AFFINE
        self.model.actuator_gainprm[ids] = 0.0
        self.model.actuator_gainprm[ids, 0] = gains.stiffness
        self.model.actuator_biasprm[ids] = 0.0
        self.model.actuator_biasprm[ids, 1] = -gains.stiffness
        self.model.actuator_biasprm[ids, 2] = -gains.damping
        # Control is now a joint angle target, so torque clipping belongs on
        # actuator force rather than on ctrl.
        self.model.actuator_ctrllimited[ids] = False
        self.model.actuator_forcelimited[ids] = True
        self.model.actuator_forcerange[ids, 0] = -limits
        self.model.actuator_forcerange[ids, 1] = limits

    def write_joint_targets(self, data: Any, targets_policy_order: np.ndarray) -> None:
        targets = np.asarray(targets_policy_order, dtype=np.float32)
        if targets.shape != (self.contract.action_dim,) or not np.isfinite(targets).all():
            raise ValueError("joint targets must be a finite policy-order vector")
        data.ctrl[self.actuator_ids] = targets

    def actuator_forces(self, data: Any) -> np.ndarray:
        return np.asarray(data.actuator_force[self.actuator_ids], dtype=np.float32).copy()

    def write_torques(self, data: Any, torques_policy_order: np.ndarray) -> None:
        torques = np.asarray(torques_policy_order, dtype=np.float32)
        if torques.shape != (self.contract.action_dim,) or not np.isfinite(torques).all():
            raise ValueError("torques must be a finite policy-order action vector")
        data.ctrl[self.actuator_ids] = torques


@dataclass(frozen=True)
class MujocoEpisodeConfig:
    max_policy_steps: int = 500
    fall_height: float = 0.45
    fall_projected_gravity_z: float = -0.35
    controller_mode: int = 1

    def __post_init__(self) -> None:
        if self.max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be positive")
        if not np.isfinite(self.fall_height) or self.fall_height <= 0.0:
            raise ValueError("fall_height must be finite and positive")
        if not -1.0 <= self.fall_projected_gravity_z <= 1.0:
            raise ValueError("fall_projected_gravity_z must be in [-1, 1]")


@dataclass(frozen=True)
class MujocoEpisodeResult:
    episode_index: int
    policy_steps: int
    sim_time: float
    episode_return: float
    terminated: bool
    termination_reason: str
    mean_linear_tracking_error: float
    mean_yaw_tracking_error: float


class MujocoDeploymentRunner:
    """Run one policy backend through the strict 37-DoF MuJoCo boundary."""

    def __init__(
        self,
        mujoco_module: Any,
        model: Any,
        data: Any,
        policy: PolicyBackend,
        *,
        contract: DeploymentContract = DEFAULT_CONTRACT,
        dynamics: DynamicsParameters | None = None,
        reward_provider: RewardProvider | None = None,
        episode_config: MujocoEpisodeConfig | None = None,
        integration_dt: float | None = None,
        stiffness_scale: float = 1.0,
        damping_scale: float = 1.0,
        actuator_mode: str = "explicit_pd",
        joint_limit_time_constant: float | None = None,
    ) -> None:
        self.mj = mujoco_module
        self.model = model
        self.data = data
        self.policy = policy
        self.contract = contract
        self.dynamics = dynamics or DynamicsParameters()
        self.reward_provider = reward_provider or ObservableVelocityReward()
        self.episode_config = episode_config or MujocoEpisodeConfig()
        if policy.observation_dim != contract.state_dim or policy.action_dim != contract.action_dim:
            raise ValueError(
                f"policy is {policy.observation_dim}->{policy.action_dim}, expected "
                f"{contract.state_dim}->{contract.action_dim}"
            )
        if policy.batch_size != 1:
            raise ValueError("minimal MuJoCo runner currently requires policy batch_size=1")

        self.bindings = MujocoBindings(
            mujoco_module, model, contract=contract, integration_dt=integration_dt
        )
        if joint_limit_time_constant is None:
            joint_limit_time_constant = max(
                0.002, 2.0 * self.bindings.metadata.deployed_timestep
            )
        self.bindings.configure_joint_limit_solver(joint_limit_time_constant)
        self.joint_limit_time_constant = float(joint_limit_time_constant)
        self.observation_builder = ObservationBuilder(contract)
        self.action_filter = FirstOrderActionLag(
            self.dynamics.action_lag, contract.action_dim
        )
        if not np.isfinite(stiffness_scale) or stiffness_scale <= 0.0:
            raise ValueError("stiffness_scale must be finite and positive")
        if not np.isfinite(damping_scale) or damping_scale <= 0.0:
            raise ValueError("damping_scale must be finite and positive")
        base_gains = robotlab_g1_pd_gains(contract)
        deployed_gains = PDGains(
            base_gains.stiffness * np.float32(stiffness_scale),
            base_gains.damping * np.float32(damping_scale),
            base_gains.effort_limit,
        )
        self.stiffness_scale = float(stiffness_scale)
        self.damping_scale = float(damping_scale)
        self.controller = PositionPDController(
            contract, gains=deployed_gains, motor_strength=self.dynamics.motor_strength
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

    @classmethod
    def from_xml(
        cls,
        model_path: str | Path,
        policy: PolicyBackend,
        **kwargs: Any,
    ) -> "MujocoDeploymentRunner":
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on deployment host
            raise RuntimeError(
                "MuJoCo deployment requires the official 'mujoco' Python package"
            ) from exc
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo XML does not exist: {path}")
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        runner = cls(mujoco, model, data, policy, **kwargs)
        runner.model_path = path
        runner.model_sha256 = _sha256_file(path)
        return runner

    def _robot_state(self, command: np.ndarray) -> tuple[RobotState, np.ndarray]:
        linear_velocity, angular_velocity, quaternion = self.bindings.base_kinematics(self.data)
        projected_gravity = projected_gravity_from_quaternion(quaternion)
        state = RobotState(
            base_linear_velocity=linear_velocity,
            base_angular_velocity=angular_velocity,
            projected_gravity=projected_gravity,
            velocity_command=command,
            joint_position=self.bindings.joint_position(self.data),
            joint_velocity=self.bindings.joint_velocity(self.data),
            previous_action=self.previous_executed_action,
        )
        return state, projected_gravity

    def _terminated(self, projected_gravity: np.ndarray) -> tuple[bool, str]:
        if self.bindings.base_height(self.data) < self.episode_config.fall_height:
            return True, "base_height"
        if float(projected_gravity[2]) > self.episode_config.fall_projected_gravity_z:
            return True, "orientation"
        return False, "time_limit"

    def step_physics(self, executed_action: np.ndarray) -> np.ndarray:
        """Advance one 20 ms policy interval and return applied joint forces."""

        last_torque = np.zeros(self.contract.action_dim, dtype=np.float32)
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
                last_torque = control.torques
                self.bindings.write_torques(self.data, last_torque)
            self.mj.mj_step(self.model, self.data)
            if targets is not None:
                last_torque = self.bindings.actuator_forces(self.data)
        return last_torque

    def run_episode(
        self,
        command: np.ndarray | Sequence[float],
        *,
        episode_index: int,
        frame_callback: Callable[[int, int], None] | None = None,
    ) -> MujocoEpisodeResult:
        command_array = _finite_command(command)
        self.bindings.reset_to_policy_pose(self.data)
        self.bindings.apply_dynamics(self.data, self.dynamics)
        self.action_filter.reset()
        self.previous_executed_action.fill(0.0)
        self.policy.start_episode(episode_index)
        if frame_callback is not None:
            frame_callback(int(episode_index), -1)

        episode_return = 0.0
        linear_errors: list[float] = []
        yaw_errors: list[float] = []
        terminated = False
        reason = "time_limit"
        completed_steps = 0
        try:
            for step in range(self.episode_config.max_policy_steps):
                pre_state, _ = self._robot_state(command_array)
                observation = self.observation_builder.build(pre_state)
                raw_action = np.asarray(self.policy.predict(observation), dtype=np.float32)
                executed_action = self.action_filter.apply(raw_action)
                previous_executed = self.previous_executed_action.copy()

                last_torque = self.step_physics(executed_action)
                if frame_callback is not None:
                    frame_callback(int(episode_index), int(step))

                post_state, projected_gravity = self._robot_state(command_array)
                terminated, reason = self._terminated(projected_gravity)
                reward = self.reward_provider.compute(
                    RewardInput(
                        base_linear_velocity=np.asarray(post_state.base_linear_velocity),
                        base_angular_velocity=np.asarray(post_state.base_angular_velocity),
                        projected_gravity=projected_gravity,
                        command=command_array,
                        action=executed_action,
                        previous_action=previous_executed,
                        joint_torque=last_torque,
                        terminated=terminated,
                    )
                )
                # Official online semantics: pre-step state + raw policy action
                # + post-step reward.  The lagged action remains inside the
                # physical environment and next observation only.
                self.policy.record_transition(
                    observation,
                    raw_action,
                    reward,
                    controller_modes=self.episode_config.controller_mode,
                )
                self.previous_executed_action = executed_action
                episode_return += reward
                linear_errors.append(
                    float(
                        np.linalg.norm(
                            command_array[:2]
                            - np.asarray(post_state.base_linear_velocity, dtype=np.float32)[:2]
                        )
                    )
                )
                yaw_errors.append(
                    abs(
                        float(
                            command_array[2]
                            - np.asarray(post_state.base_angular_velocity, dtype=np.float32)[2]
                        )
                    )
                )
                completed_steps = step + 1
                if terminated:
                    break
        finally:
            self.policy.finish_episode()

        return MujocoEpisodeResult(
            episode_index=int(episode_index),
            policy_steps=completed_steps,
            sim_time=completed_steps * self.contract.policy_dt,
            episode_return=float(episode_return),
            terminated=terminated,
            termination_reason=reason,
            mean_linear_tracking_error=float(np.mean(linear_errors)) if linear_errors else float("nan"),
            mean_yaw_tracking_error=float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        )

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "claim_scope": "minimal-interface-smoke-not-sim2sim-parity",
            "policy": self.policy.provenance,
            "contract": {
                "state_dim": self.contract.state_dim,
                "action_dim": self.contract.action_dim,
                "physics_dt": self.contract.physics_dt,
                "decimation": self.contract.decimation,
                "policy_hz": self.contract.policy_hz,
                "action_scale": self.contract.action_scale,
                "joint_names": list(self.contract.joint_names),
                "sha256": self.contract.sha256,
            },
            "bindings": asdict(self.bindings.metadata),
            "dynamics": asdict(self.dynamics),
            "reward_schema": self.reward_provider.schema_id,
            "friction_semantics": (
                "fixed-mujoco-sliding-coefficient-on-all-minimum-scene-geoms; "
                "required-because-equal-priority-contact-mixing-uses-max"
            ),
            "effort_limit_semantics": "robotlab-g1-minimal-cfg-scaled-by-motor-strength",
            "effective_effort_limits": self.controller.effective_effort_limit.tolist(),
            "pd_stiffness": self.controller.gains.stiffness.astype(float).tolist(),
            "pd_damping": self.controller.gains.damping.astype(float).tolist(),
            "pd_stiffness_scale": self.stiffness_scale,
            "pd_damping_scale": self.damping_scale,
            "actuator_mode": self.actuator_mode,
            "mujoco_integrator": int(self.model.opt.integrator),
            "joint_armature": robotlab_g1_joint_armatures(self.contract).astype(float).tolist(),
            "realized_reset_joint_positions": self.bindings.policy_reset_joint_positions()
            .astype(float)
            .tolist(),
            "reset_soft_joint_position_limit_factor": 0.9,
            "joint_limit_time_constant": self.joint_limit_time_constant,
        }
        if hasattr(self, "model_path"):
            payload["model"] = {
                "path": str(self.model_path),
                "sha256": self.model_sha256,
            }
        if hasattr(self, "robot_asset_report"):
            payload["robot_asset_contract"] = self.robot_asset_report
        if hasattr(self, "scene_binding_report"):
            payload["scene_robot_binding"] = self.scene_binding_report
        return payload

    def write_report(
        self,
        path: str | Path,
        results: Sequence[MujocoEpisodeResult],
    ) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.provenance() | {
            "episodes": [asdict(result) for result in results],
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return output


__all__ = [
    "MujocoBindingMetadata",
    "MujocoBindings",
    "MujocoDeploymentRunner",
    "MujocoEpisodeConfig",
    "MujocoEpisodeResult",
]
