"""CPU-only tests for the RobotLab G1 deployment boundary."""

from __future__ import annotations

import unittest

import numpy as np

from deployment.robotlab_g1.context import (
    EpisodeContextBuffer,
    LEGACY_CONTEXT_PROTOCOL,
    OFFICIAL_CONTEXT_PROTOCOL,
)
from deployment.robotlab_g1.contract import (
    DEFAULT_CONTRACT,
    G1_DEFAULT_JOINT_POSITIONS,
    G1_POLICY_JOINT_NAMES,
)
from deployment.robotlab_g1.observation import (
    ObservationBuilder,
    RobotState,
    projected_gravity_from_quaternion,
)


class DeploymentContractTest(unittest.TestCase):
    def test_dimensions_timing_and_layout_are_frozen(self):
        contract = DEFAULT_CONTRACT
        self.assertEqual(contract.state_dim, 123)
        self.assertEqual(contract.action_dim, 37)
        self.assertEqual(len(G1_POLICY_JOINT_NAMES), 37)
        self.assertEqual(len(set(G1_POLICY_JOINT_NAMES)), 37)
        self.assertEqual(len(G1_DEFAULT_JOINT_POSITIONS), 37)
        self.assertAlmostEqual(contract.physics_dt, 0.005)
        self.assertEqual(contract.decimation, 4)
        self.assertAlmostEqual(contract.policy_dt, 0.02)
        self.assertAlmostEqual(contract.policy_hz, 50.0)
        self.assertEqual(len(contract.sha256), 64)
        self.assertEqual(contract.as_dict()["state_dim"], 123)
        self.assertEqual(
            [(term.name, term.width) for term in contract.observation_terms],
            [
                ("base_linear_velocity", 3),
                ("base_angular_velocity", 3),
                ("projected_gravity", 3),
                ("velocity_command", 3),
                ("joint_position_relative", 37),
                ("joint_velocity_relative", 37),
                ("previous_action", 37),
            ],
        )

    def test_physx_policy_joint_order_is_exact(self):
        self.assertEqual(
            G1_POLICY_JOINT_NAMES,
            (
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
            ),
        )

    def test_joint_order_requires_explicit_remap(self):
        source_names = tuple(reversed(G1_POLICY_JOINT_NAMES))
        with self.assertRaisesRegex(ValueError, "joint order mismatch"):
            DEFAULT_CONTRACT.validate_joint_names(source_names)
        DEFAULT_CONTRACT.validate_joint_names(source_names, require_order=False)

        source_values = np.arange(37, dtype=np.float32)[::-1]
        mapped = DEFAULT_CONTRACT.reorder_joint_vector(source_values, source_names)
        np.testing.assert_array_equal(mapped, np.arange(37, dtype=np.float32))

    def test_missing_duplicate_and_extra_joints_are_rejected(self):
        missing = G1_POLICY_JOINT_NAMES[:-1]
        with self.assertRaisesRegex(ValueError, "missing"):
            DEFAULT_CONTRACT.validate_joint_names(missing, require_order=False)
        duplicate = list(G1_POLICY_JOINT_NAMES)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            DEFAULT_CONTRACT.validate_joint_names(duplicate, require_order=False)

    def test_action_target_transform(self):
        action = np.linspace(-1.0, 1.0, 37, dtype=np.float32)
        expected = np.asarray(G1_DEFAULT_JOINT_POSITIONS, dtype=np.float32) + 0.5 * action
        actual = DEFAULT_CONTRACT.action_to_joint_targets(action)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        with self.assertRaisesRegex(ValueError, "trailing dimension"):
            DEFAULT_CONTRACT.action_to_joint_targets(np.zeros(36, dtype=np.float32))


class ObservationBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = ObservationBuilder()
        self.defaults = np.asarray(G1_DEFAULT_JOINT_POSITIONS, dtype=np.float32)

    def make_state(self, **overrides) -> RobotState:
        values = {
            "base_linear_velocity": [1.0, 2.0, 3.0],
            "base_angular_velocity": [4.0, 5.0, 6.0],
            "projected_gravity": [7.0, 8.0, 9.0],
            "velocity_command": [10.0, 11.0, 12.0],
            "joint_position": self.defaults + np.arange(37, dtype=np.float32),
            "joint_velocity": 100.0 + np.arange(37, dtype=np.float32),
            "previous_action": 200.0 + np.arange(37, dtype=np.float32),
        }
        values.update(overrides)
        return RobotState(**values)

    def test_exact_123_element_concatenation(self):
        observation = self.builder.build(self.make_state())
        self.assertEqual(observation.shape, (123,))
        self.assertEqual(observation.dtype, np.float32)
        slices = DEFAULT_CONTRACT.observation_slices
        np.testing.assert_array_equal(observation[slices["base_linear_velocity"]], [1, 2, 3])
        np.testing.assert_array_equal(observation[slices["base_angular_velocity"]], [4, 5, 6])
        np.testing.assert_array_equal(observation[slices["projected_gravity"]], [7, 8, 9])
        np.testing.assert_array_equal(observation[slices["velocity_command"]], [10, 11, 12])
        np.testing.assert_allclose(
            observation[slices["joint_position_relative"]],
            np.arange(37, dtype=np.float32),
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_array_equal(
            observation[slices["joint_velocity_relative"]], 100 + np.arange(37, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            observation[slices["previous_action"]], 200 + np.arange(37, dtype=np.float32)
        )

    def test_source_joint_order_is_remapped_before_relative_position(self):
        names = tuple(reversed(G1_POLICY_JOINT_NAMES))
        canonical_position = self.defaults + np.arange(37, dtype=np.float32)
        canonical_velocity = np.arange(37, dtype=np.float32) * 2.0
        observation = self.builder.build(
            self.make_state(
                joint_names=names,
                joint_position=canonical_position[::-1],
                joint_velocity=canonical_velocity[::-1],
            )
        )
        slices = DEFAULT_CONTRACT.observation_slices
        np.testing.assert_allclose(
            observation[slices["joint_position_relative"]],
            np.arange(37, dtype=np.float32),
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_array_equal(
            observation[slices["joint_velocity_relative"]], canonical_velocity
        )

    def test_wrong_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "joint_position"):
            self.builder.build(self.make_state(joint_position=np.zeros(36)))
        base_velocity = np.zeros(3, dtype=np.float32)
        base_velocity[1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.builder.build(self.make_state(base_linear_velocity=base_velocity))

    def test_projected_gravity_quaternion_convention(self):
        np.testing.assert_allclose(
            projected_gravity_from_quaternion([1.0, 0.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            rtol=0.0,
            atol=1.0e-7,
        )
        # +90 degrees about world Y: world down appears as +X in body.
        half = np.sqrt(0.5)
        np.testing.assert_allclose(
            projected_gravity_from_quaternion([half, 0.0, half, 0.0]),
            [1.0, 0.0, 0.0],
            rtol=0.0,
            atol=1.0e-6,
        )


class EpisodeContextBufferTest(unittest.TestCase):
    @staticmethod
    def transition(value: float) -> tuple[np.ndarray, np.ndarray, float]:
        return (
            np.full(123, value, dtype=np.float32),
            np.full(37, -value, dtype=np.float32),
            value + 0.25,
        )

    def append_episode(self, buffer: EpisodeContextBuffer, index: int, length: int) -> None:
        buffer.start_episode(index)
        for step in range(length):
            buffer.append(*self.transition(index * 1000 + step))
        buffer.finish_episode()

    def test_official_prompt_is_empty_then_previous_episode_only_and_frozen(self):
        buffer = EpisodeContextBuffer(OFFICIAL_CONTEXT_PROTOCOL)
        initial = buffer.start_episode(0)
        self.assertEqual(initial.states.shape, (0, 123))
        self.assertEqual(initial.actions.shape, (0, 37))
        self.assertEqual(initial.rewards.shape, (0, 1))
        for step in range(70):
            buffer.append(*self.transition(step))
        # Current transitions never leak into the current prompt.
        self.assertEqual(buffer.prompt().length, 0)
        buffer.finish_episode()

        prompt = buffer.start_episode(1)
        self.assertEqual(prompt.length, 64)
        self.assertEqual(prompt.episode_indices, (0,))
        self.assertEqual(prompt.episode_lengths, (64,))
        self.assertEqual(prompt.states[:, 0].tolist(), list(map(float, range(6, 70))))
        buffer.append(*self.transition(999.0))
        self.assertEqual(buffer.prompt().states[-1, 0], 69.0)
        buffer.finish_episode()

        prompt = buffer.start_episode(2)
        self.assertEqual(prompt.episode_indices, (1,))
        self.assertEqual(prompt.length, 1)
        self.assertEqual(prompt.states[0, 0], 999.0)

    def test_legacy_protocol_keeps_four_completed_episodes(self):
        buffer = EpisodeContextBuffer(LEGACY_CONTEXT_PROTOCOL)
        for episode_index in range(5):
            self.append_episode(buffer, episode_index, 64)
        prompt = buffer.start_episode(5)
        self.assertEqual(prompt.length, 256)
        self.assertEqual(prompt.episode_indices, (1, 2, 3, 4))
        self.assertEqual(prompt.episode_lengths, (64, 64, 64, 64))
        self.assertEqual(prompt.states[0, 0], 1000.0)
        self.assertEqual(prompt.states[-1, 0], 4063.0)
        batch = prompt.as_batch()
        self.assertEqual([item.shape for item in batch], [(1, 256, 123), (1, 256, 37), (1, 256, 1), (1, 256)])

    def test_context_validation_and_clear(self):
        buffer = EpisodeContextBuffer()
        with self.assertRaises(RuntimeError):
            buffer.append(*self.transition(0.0))
        buffer.start_episode(0)
        with self.assertRaisesRegex(ValueError, "state shape"):
            buffer.append(np.zeros(122), np.zeros(37), 0.0)
        invalid = np.zeros(37, dtype=np.float32)
        invalid[0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            buffer.append(np.zeros(123), invalid, 0.0)
        buffer.clear()
        self.assertFalse(buffer.episode_active)
        self.assertEqual(buffer.completed_episode_indices, ())
        self.assertEqual(buffer.prompt().length, 0)


if __name__ == "__main__":
    unittest.main()
