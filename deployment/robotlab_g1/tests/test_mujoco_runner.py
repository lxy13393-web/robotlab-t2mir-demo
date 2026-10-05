from __future__ import annotations

import types
import unittest

import numpy as np

from deployment.robotlab_g1.contract import DEFAULT_CONTRACT
from deployment.robotlab_g1.control import (
    DynamicsParameters,
    robotlab_g1_joint_armatures,
)
from deployment.robotlab_g1.mujoco_runner import (
    MujocoBindings,
    MujocoDeploymentRunner,
    MujocoEpisodeConfig,
)
from deployment.robotlab_g1.policy_backends import PolicyBackend


class _FakeModel:
    def __init__(self):
        self.opt = types.SimpleNamespace(timestep=0.002, integrator=0)
        self._joint_names = ["floating_base_joint"] + list(
            reversed(DEFAULT_CONTRACT.joint_names)
        )
        self._body_names = ["world", "pelvis", "torso_link"]
        self.njnt = len(self._joint_names)
        self.nq = 7 + 37
        self.nv = 6 + 37
        self.nu = 37
        self.jnt_type = np.array([0] + [3] * 37)
        self.jnt_qposadr = np.array([0] + list(range(7, 44)))
        self.jnt_dofadr = np.array([0] + list(range(6, 43)))
        self.jnt_bodyid = np.array([1] + [2] * 37)
        self.jnt_actfrclimited = np.zeros(self.njnt, dtype=bool)
        self.jnt_limited = np.zeros(self.njnt, dtype=bool)
        self.jnt_range = np.tile([-2.0, 2.0], (self.njnt, 1))
        self.jnt_solref = np.tile([0.02, 1.0], (self.njnt, 1))
        self.dof_armature = np.zeros(self.nv, dtype=float)
        self.dof_damping = np.zeros(self.nv, dtype=float)
        self.dof_frictionloss = np.zeros(self.nv, dtype=float)
        for policy_index, name in enumerate(DEFAULT_CONTRACT.joint_names):
            native_joint = self._joint_names.index(name)
            dof_address = int(self.jnt_dofadr[native_joint])
            self.dof_armature[dof_address] = robotlab_g1_joint_armatures()[policy_index]
        # Actuator native order follows the reversed MJCF joint order.
        self.actuator_trnid = np.column_stack(
            (np.arange(1, 38, dtype=np.int64), np.full(37, -1, dtype=np.int64))
        )
        self.actuator_ctrllimited = np.ones(37, dtype=bool)
        self.actuator_ctrlrange = np.tile([-100.0, 100.0], (37, 1))
        self.actuator_trntype = np.zeros(37, dtype=np.int32)
        self.actuator_dyntype = np.zeros(37, dtype=np.int32)
        self.actuator_gaintype = np.zeros(37, dtype=np.int32)
        self.actuator_biastype = np.zeros(37, dtype=np.int32)
        self.actuator_gainprm = np.zeros((37, 10), dtype=float)
        self.actuator_gainprm[:, 0] = 1.0
        self.actuator_biasprm = np.zeros((37, 10), dtype=float)
        self.actuator_gear = np.zeros((37, 6), dtype=float)
        self.actuator_gear[:, 0] = 1.0
        self.actuator_forcelimited = np.zeros(37, dtype=bool)
        self.actuator_forcerange = np.tile([-100.0, 100.0], (37, 1))
        self.body_mass = np.array([0.0, 10.0, 20.0])
        self.body_inertia = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=float
        )
        self.geom_friction = np.tile([0.8, 0.1, 0.01], (3, 1)).astype(float)
        self.geom_bodyid = np.array([0, 1, 2])
        self.qpos0 = np.zeros(self.nq)
        self.qpos0[2] = 0.8
        self.qpos0[3] = 1.0


class _FakeData:
    def __init__(self, model):
        self.qpos = model.qpos0.copy()
        self.qvel = np.zeros(model.nv)
        self.ctrl = np.zeros(model.nu)
        self.actuator_force = np.zeros(model.nu)
        self.xquat = np.zeros((3, 4))
        self.xquat[:, 0] = 1.0
        self.xmat = np.tile(np.eye(3).reshape(1, 9), (3, 1))
        self.xpos = np.zeros((3, 3))
        self.xpos[1, 2] = 0.8
        self.object_velocity_world = np.zeros(6)
        self.time = 0.0


class _FakeMujoco:
    MjData = _FakeData
    mjtObj = types.SimpleNamespace(mjOBJ_JOINT=0, mjOBJ_BODY=1)
    mjtJoint = types.SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
    mjtTrn = types.SimpleNamespace(mjTRN_JOINT=0)
    mjtDyn = types.SimpleNamespace(mjDYN_NONE=0)
    mjtGain = types.SimpleNamespace(mjGAIN_FIXED=0)
    mjtBias = types.SimpleNamespace(mjBIAS_NONE=0, mjBIAS_AFFINE=1)
    mjtIntegrator = types.SimpleNamespace(mjINT_IMPLICITFAST=3)
    last_setconst_qpos = None

    @staticmethod
    def mj_name2id(model, object_type, name):
        names = model._joint_names if object_type == 0 else model._body_names
        try:
            return names.index(name)
        except ValueError:
            return -1

    @staticmethod
    def mj_resetData(model, data):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.time = 0.0
        _FakeMujoco.mj_forward(model, data)

    @staticmethod
    def mj_forward(model, data):
        data.xpos[1, 2] = data.qpos[2]
        data.xquat[1] = data.qpos[3:7]

    @staticmethod
    def mj_setConst(model, data):
        _FakeMujoco.last_setconst_qpos = data.qpos.copy()
        del model

    @staticmethod
    def mj_objectVelocity(model, data, object_type, object_id, result, local):
        del model, object_type, object_id
        if local != 0:
            raise AssertionError("deployment must request world-frame COM velocity")
        result[:] = data.object_velocity_world

    @staticmethod
    def mj_step(model, data):
        data.time += model.opt.timestep


class _ZeroPolicy(PolicyBackend):
    observation_dim = 123
    action_dim = 37
    batch_size = 1
    provenance = {"backend": "zero-test"}

    def __init__(self):
        self.records = []
        self.active = False

    def start_episode(self, episode_index=None):
        self.active = True

    def predict(self, observations):
        self.assert_shape = np.asarray(observations).shape
        return np.zeros(37, dtype=np.float32)

    def record_transition(self, states, policy_actions, rewards, active=None, controller_modes=None):
        self.records.append(
            (np.asarray(states).copy(), np.asarray(policy_actions).copy(), rewards, controller_modes)
        )

    def finish_episode(self):
        self.active = False


class MujocoBindingTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.data = _FakeData(self.model)

    def test_reversed_native_order_is_mapped_to_policy_order(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        native_values = np.arange(37, dtype=float)
        self.data.qpos[7:] = native_values
        np.testing.assert_array_equal(bindings.joint_position(self.data), native_values[::-1])
        self.assertAlmostEqual(self.model.opt.timestep, 0.005)

    def test_dynamics_do_not_accumulate(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        deployed_limits = np.linspace(10.0, 46.0, 37)
        bindings.set_robotlab_effort_limits(deployed_limits)
        np.testing.assert_allclose(
            self.model.actuator_ctrlrange[bindings.actuator_ids, 1], deployed_limits
        )
        np.testing.assert_allclose(
            self.model.actuator_ctrlrange[bindings.actuator_ids, 0], -deployed_limits
        )
        bindings.apply_dynamics(
            self.data,
            DynamicsParameters(payload_kg=3.0, friction=0.65),
        )
        self.assertEqual(self.model.body_mass[2], 23.0)
        np.testing.assert_allclose(self.model.body_inertia[2], [2.3, 4.6, 6.9])
        np.testing.assert_array_equal(
            self.model.geom_friction[:, 0], [0.65, 0.65, 0.65]
        )
        bindings.apply_dynamics(self.data, DynamicsParameters(payload_kg=0.0, friction=1.0))
        self.assertEqual(self.model.body_mass[2], 20.0)
        np.testing.assert_array_equal(self.model.body_inertia[2], [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(self.model.geom_friction[:, 0], [1.0, 1.0, 1.0])

    def test_dynamics_recompute_constants_at_model_reference_pose(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        live_qpos = np.linspace(-0.5, 0.5, self.model.nq)
        self.data.qpos[:] = live_qpos
        bindings.apply_dynamics(self.data, DynamicsParameters())
        np.testing.assert_array_equal(_FakeMujoco.last_setconst_qpos, self.model.qpos0)
        np.testing.assert_array_equal(self.data.qpos, live_qpos)

    def test_implicit_position_pd_runtime_conversion_preserves_limits(self):
        policy = _ZeroPolicy()
        runner = MujocoDeploymentRunner(
            _FakeMujoco,
            self.model,
            self.data,
            policy,
            actuator_mode="implicit_pd",
        )
        ids = runner.bindings.actuator_ids
        self.assertEqual(self.model.opt.integrator, 3)
        self.assertTrue(self.model.actuator_forcelimited[ids].all())
        self.assertFalse(self.model.actuator_ctrllimited[ids].any())
        np.testing.assert_allclose(
            self.model.actuator_forcerange[ids, 1],
            runner.controller.effective_effort_limit,
        )
        np.testing.assert_allclose(
            self.model.actuator_biasprm[ids, 1], -runner.controller.gains.stiffness
        )
        np.testing.assert_allclose(
            self.model.actuator_biasprm[ids, 2], -runner.controller.gains.damping
        )

    def test_reset_uses_recorded_robotlab_root_pose(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        self.data.qpos[:7] = (9.0, 8.0, 7.0, 0.0, 1.0, 0.0, 0.0)
        bindings.reset_to_policy_pose(self.data)
        np.testing.assert_allclose(
            self.data.qpos[:3], DEFAULT_CONTRACT.initial_root_position
        )
        np.testing.assert_allclose(
            self.data.qpos[3:7], DEFAULT_CONTRACT.initial_root_quaternion_wxyz
        )

    def test_reset_clamps_default_to_robotlab_soft_joint_limits(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        name = "left_six_joint"
        joint_id = self.model._joint_names.index(name)
        self.model.jnt_limited[joint_id] = True
        self.model.jnt_range[joint_id] = (-1.84, 0.0)
        bindings.reset_to_policy_pose(self.data)
        policy_index = DEFAULT_CONTRACT.joint_names.index(name)
        self.assertAlmostEqual(
            float(self.data.qpos[bindings.qpos_addresses[policy_index]]), -0.092
        )
        # The action/observation reference is intentionally still zero.
        self.assertEqual(DEFAULT_CONTRACT.default_joint_positions[policy_index], 0.0)

    def test_joint_limit_solver_is_applied_in_policy_order(self):
        bindings = MujocoBindings(_FakeMujoco, self.model, integration_dt=0.001)
        joint_id = self.model._joint_names.index("right_four_joint")
        self.model.jnt_limited[joint_id] = True
        bindings.configure_joint_limit_solver(0.005)
        self.assertEqual(float(self.model.jnt_solref[joint_id, 0]), 0.005)

    def test_joint_dynamics_must_match_recorded_asset(self):
        mutations = {
            "armature": lambda model: model.dof_armature.__setitem__(6, 0.02),
            "damping": lambda model: model.dof_damping.__setitem__(6, 0.5),
            "friction": lambda model: model.dof_frictionloss.__setitem__(6, 0.1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                model = _FakeModel()
                mutate(model)
                with self.assertRaisesRegex(ValueError, "joint dynamics do not match"):
                    MujocoBindings(_FakeMujoco, model)

    def test_29_dof_or_missing_joint_asset_is_rejected(self):
        self.model._joint_names.remove("left_six_joint")
        with self.assertRaisesRegex(ValueError, "missing required object"):
            MujocoBindings(_FakeMujoco, self.model)

    def test_position_servo_semantics_are_rejected(self):
        # MuJoCo position actuators use affine bias; direct torque motors do not.
        self.model.actuator_biastype[0] = 1
        with self.assertRaisesRegex(ValueError, "not direct unit-gear torque motors"):
            MujocoBindings(_FakeMujoco, self.model)

    def test_scaled_or_stateful_actuators_are_rejected(self):
        mutations = {
            "transmission": lambda model: model.actuator_trntype.__setitem__(0, 1),
            "dynamics": lambda model: model.actuator_dyntype.__setitem__(0, 1),
            "gain type": lambda model: model.actuator_gaintype.__setitem__(0, 1),
            "gain scale": lambda model: model.actuator_gainprm.__setitem__((0, 0), 2.0),
            "gear": lambda model: model.actuator_gear.__setitem__((0, 0), 2.0),
            "force clamp": lambda model: model.actuator_forcelimited.__setitem__(0, True),
            "joint force clamp": lambda model: model.jnt_actfrclimited.__setitem__(1, True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                model = _FakeModel()
                mutate(model)
                with self.assertRaisesRegex(ValueError, "not direct unit-gear torque motors"):
                    MujocoBindings(_FakeMujoco, model)

    def test_base_com_velocity_is_rotated_from_world_to_actor_body_frame(self):
        bindings = MujocoBindings(_FakeMujoco, self.model)
        # Actor body is rotated +90 degrees around world z.  A +world-x
        # velocity is therefore -body-y.
        self.data.xmat[1] = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        ).reshape(9)
        self.data.object_velocity_world[:] = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        linear, angular, _ = bindings.base_kinematics(self.data)
        np.testing.assert_allclose(linear, (0.0, -1.0, 0.0), atol=1e-7)
        np.testing.assert_allclose(angular, (1.0, 0.0, 0.0), atol=1e-7)


class MujocoRunnerTest(unittest.TestCase):
    def test_cpu_fake_episode_preserves_raw_action_prompt_semantics(self):
        model = _FakeModel()
        data = _FakeData(model)
        policy = _ZeroPolicy()
        runner = MujocoDeploymentRunner(
            _FakeMujoco,
            model,
            data,
            policy,
            dynamics=DynamicsParameters(action_lag=0.4, motor_strength=0.8),
            episode_config=MujocoEpisodeConfig(max_policy_steps=3),
        )
        frames = []
        result = runner.run_episode(
            [0.0, 0.0, 0.0],
            episode_index=0,
            frame_callback=lambda episode, step: frames.append((episode, step)),
        )
        self.assertEqual(result.policy_steps, 3)
        self.assertFalse(result.terminated)
        self.assertEqual(policy.assert_shape, (123,))
        self.assertEqual(len(policy.records), 3)
        self.assertEqual(policy.records[0][0].shape, (123,))
        self.assertEqual(policy.records[0][1].shape, (37,))
        self.assertEqual(policy.records[0][3], 1)
        self.assertFalse(policy.active)
        self.assertEqual(frames, [(0, -1), (0, 0), (0, 1), (0, 2)])
        self.assertEqual(runner.provenance()["claim_scope"], "minimal-interface-smoke-not-sim2sim-parity")


if __name__ == "__main__":
    unittest.main()
