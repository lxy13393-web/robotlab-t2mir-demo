"""Pure CPU tests for the ROS-independent deployment policy core."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.robotlab_g1.contract import DEFAULT_CONTRACT  # noqa: E402
from deployment.robotlab_g1.observation import RobotState  # noqa: E402
from deployment.robotlab_g1.ros2_node import (  # noqa: E402
    PolicyLoopConfig,
    PolicyTimingStats,
    Ros2PolicyCore,
    _diagnostic_level_byte,
    _load_ros2,
    ros2_environment_report,
)


class FakeBackend:
    observation_dim = DEFAULT_CONTRACT.state_dim
    action_dim = DEFAULT_CONTRACT.action_dim
    batch_size = 1
    provenance = {"backend": "fake"}

    def __init__(self) -> None:
        self.starts: list[int | None] = []
        self.predictions: list[np.ndarray] = []
        self.transitions: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
        self.finishes = 0

    def start_episode(self, episode_index: int | None = None):
        self.starts.append(episode_index)

    def predict(self, observation):
        self.predictions.append(np.asarray(observation, dtype=np.float32).copy())
        offset = np.float32(len(self.predictions) - 1)
        return np.linspace(-1.0, 1.0, self.action_dim, dtype=np.float32) + offset

    def record_transition(
        self,
        states,
        policy_actions,
        rewards,
        active=None,
        controller_modes=None,
    ) -> None:
        self.transitions.append(
            (
                np.asarray(states, dtype=np.float32).copy(),
                np.asarray(policy_actions, dtype=np.float32).copy(),
                float(np.asarray(rewards)),
                bool(np.asarray(active)),
            )
        )

    def finish_episode(self):
        self.finishes += 1


def make_state() -> RobotState:
    action_dim = DEFAULT_CONTRACT.action_dim
    return RobotState(
        base_linear_velocity=np.asarray((0.1, 0.2, 0.3), dtype=np.float32),
        base_angular_velocity=np.asarray((0.4, 0.5, 0.6), dtype=np.float32),
        projected_gravity=np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        # These two fields are intentionally wrong: the policy core owns them.
        velocity_command=np.full(3, 99.0, dtype=np.float32),
        joint_position=np.asarray(DEFAULT_CONTRACT.default_joint_positions, dtype=np.float32),
        joint_velocity=np.zeros(action_dim, dtype=np.float32),
        previous_action=np.full(action_dim, 99.0, dtype=np.float32),
        joint_names=DEFAULT_CONTRACT.joint_names,
    )


class Ros2PolicyCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.core = Ros2PolicyCore(
            self.backend,
            config=PolicyLoopConfig(command_timeout_s=0.5, require_fresh_reward=False),
        )

    def test_command_previous_action_and_action_transform(self) -> None:
        self.core.update_command(0.7, -0.2, 0.3, now_s=10.0)
        first = self.core.step(make_state(), now_s=10.1)

        slices = DEFAULT_CONTRACT.observation_slices
        np.testing.assert_allclose(
            first.observation[slices["velocity_command"]],
            (0.7, -0.2, 0.3),
        )
        np.testing.assert_array_equal(
            first.observation[slices["previous_action"]],
            np.zeros(DEFAULT_CONTRACT.action_dim, dtype=np.float32),
        )
        np.testing.assert_allclose(
            first.joint_targets,
            DEFAULT_CONTRACT.action_to_joint_targets(first.policy_action),
        )
        self.assertEqual(self.backend.starts, [0])
        self.assertEqual(self.backend.transitions, [])
        self.assertFalse(first.command_stale)

        self.core.update_reward(1.25)
        second = self.core.step(make_state(), now_s=10.2)
        self.assertEqual(len(self.backend.transitions), 1)
        recorded_state, recorded_action, recorded_reward, active = self.backend.transitions[0]
        np.testing.assert_array_equal(recorded_state, first.observation)
        np.testing.assert_array_equal(recorded_action, first.policy_action)
        self.assertEqual(recorded_reward, 1.25)
        self.assertTrue(active)
        self.assertEqual(second.recorded_reward, 1.25)
        np.testing.assert_array_equal(
            second.observation[slices["previous_action"]],
            first.policy_action,
        )

    def test_command_timeout_applies_safe_zero(self) -> None:
        self.core.update_command(1.0, 2.0, 3.0, now_s=1.0)
        result = self.core.step(make_state(), now_s=1.51)
        command_slice = DEFAULT_CONTRACT.observation_slices["velocity_command"]
        np.testing.assert_array_equal(result.observation[command_slice], np.zeros(3))
        self.assertTrue(result.command_stale)
        self.assertEqual(self.core.diagnostics(now_s=1.51)["command_stale"], "true")

    def test_external_reward_is_consumed_once(self) -> None:
        self.core.step(make_state(), now_s=0.0)
        self.core.update_reward(4.0)
        self.core.step(make_state(), now_s=0.02)
        self.core.step(make_state(), now_s=0.04)
        self.assertEqual([item[2] for item in self.backend.transitions], [4.0, 0.0])
        diagnostics = self.core.diagnostics(now_s=0.04)
        self.assertEqual(diagnostics["missing_reward_feedback"], "1")
        self.assertEqual(diagnostics["reward_pending"], "false")

    def test_reward_without_action_and_duplicate_reward_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without a pending"):
            self.core.update_reward(1.0)
        self.core.step(make_state(), now_s=0.0)
        self.core.update_reward(1.0)
        with self.assertRaisesRegex(RuntimeError, "duplicate reward"):
            self.core.update_reward(2.0)

    def test_executed_action_feedback_changes_observation_not_prompt_action(self) -> None:
        first = self.core.step(make_state(), now_s=0.0)
        executed = np.linspace(0.25, -0.25, 37, dtype=np.float32)
        self.core.update_executed_action(executed)
        self.core.update_reward(1.0)
        second = self.core.step(make_state(), now_s=0.02)
        previous_slice = DEFAULT_CONTRACT.observation_slices["previous_action"]
        np.testing.assert_array_equal(second.observation[previous_slice], executed)
        # Context must still contain the raw action emitted by the policy.
        np.testing.assert_array_equal(self.backend.transitions[0][1], first.policy_action)

    def test_strict_reward_mode_waits_for_feedback(self) -> None:
        backend = FakeBackend()
        core = Ros2PolicyCore(
            backend,
            config=PolicyLoopConfig(require_fresh_reward=True),
        )
        core.step(make_state(), now_s=0.0)
        with self.assertRaisesRegex(RuntimeError, "fresh reward"):
            core.step(make_state(), now_s=0.02)
        core.update_reward(0.75)
        result = core.step(make_state(), now_s=0.04)
        self.assertEqual(result.recorded_reward, 0.75)
        self.assertEqual(len(backend.transitions), 1)

    def test_fault_can_discard_unrewarded_transition_and_start_clean_episode(self) -> None:
        backend = FakeBackend()
        core = Ros2PolicyCore(
            backend,
            config=PolicyLoopConfig(require_fresh_reward=True),
        )
        core.step(make_state(), now_s=0.0)
        self.assertTrue(core.abort_pending_transition("plant_fault"))
        recovered = core.step(make_state(), now_s=0.02)
        self.assertEqual(backend.transitions, [])
        self.assertEqual(backend.finishes, 1)
        self.assertEqual(backend.starts, [0, 1])
        self.assertEqual(recovered.applied_boundary, "plant_fault")
        self.assertEqual(recovered.episode_index, 1)
        self.assertEqual(
            core.diagnostics(now_s=0.02)["discarded_transitions"], "1"
        )

    def test_fresh_reward_is_the_safe_default(self) -> None:
        self.assertTrue(PolicyLoopConfig().require_fresh_reward)

    def test_done_records_feedback_then_starts_clean_episode(self) -> None:
        first = self.core.step(make_state(), now_s=0.0)
        self.core.update_reward(-2.0)
        self.core.notify_done()
        second = self.core.step(make_state(), now_s=0.02)

        self.assertEqual(self.backend.starts, [0, 1])
        self.assertEqual(self.backend.finishes, 1)
        self.assertEqual(len(self.backend.transitions), 1)
        np.testing.assert_array_equal(self.backend.transitions[0][0], first.observation)
        self.assertEqual(self.backend.transitions[0][2], -2.0)
        self.assertEqual(second.episode_index, 1)
        self.assertEqual(second.episode_step, 0)
        self.assertEqual(second.applied_boundary, "done")
        previous_slice = DEFAULT_CONTRACT.observation_slices["previous_action"]
        np.testing.assert_array_equal(second.observation[previous_slice], np.zeros(37))

    def test_reset_before_first_tick_does_not_fabricate_episode(self) -> None:
        self.core.request_reset("operator")
        result = self.core.step(make_state(), now_s=0.0)
        self.assertEqual(result.episode_index, 0)
        self.assertIsNone(result.applied_boundary)
        self.assertEqual(self.backend.starts, [0])
        self.assertEqual(self.backend.finishes, 0)

    def test_repeated_boundaries_are_coalesced(self) -> None:
        self.core.step(make_state(), now_s=0.0)
        self.core.notify_done(reason="fall")
        self.core.request_reset(reason="operator")
        result = self.core.step(make_state(), now_s=0.02)
        self.assertEqual(result.applied_boundary, "fall")
        self.assertEqual(self.backend.starts, [0, 1])
        self.assertEqual(self.backend.finishes, 1)

    def test_finalize_flushes_last_transition_once(self) -> None:
        result = self.core.step(make_state(), now_s=0.0)
        self.core.update_reward(3.5)
        self.core.finalize()
        self.core.finalize()
        self.assertEqual(len(self.backend.transitions), 1)
        np.testing.assert_array_equal(self.backend.transitions[0][1], result.policy_action)
        self.assertEqual(self.backend.transitions[0][2], 3.5)
        self.assertEqual(self.backend.finishes, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.core.step(make_state(), now_s=0.02)

    def test_backend_contract_is_checked_at_construction(self) -> None:
        backend = FakeBackend()
        backend.action_dim = 29
        with self.assertRaisesRegex(ValueError, "action_dim"):
            Ros2PolicyCore(backend)

    def test_non_finite_action_is_rejected(self) -> None:
        self.backend.predict = lambda _observation: np.full(37, np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.core.step(make_state(), now_s=0.0)


class Ros2OptionalDependencyTest(unittest.TestCase):
    def test_humble_diagnostic_level_is_encoded_as_one_byte(self) -> None:
        self.assertEqual(_diagnostic_level_byte(0), b"\x00")
        self.assertEqual(_diagnostic_level_byte(2), b"\x02")
        with self.assertRaises(ValueError):
            _diagnostic_level_byte(256)

    def test_environment_report_never_requires_a_running_graph(self) -> None:
        report = ros2_environment_report()
        self.assertIn("available", report)
        self.assertIn("ros_distro", report)
        self.assertEqual(
            set(report["packages"]),
            {"rclpy", "geometry_msgs", "sensor_msgs", "std_msgs", "diagnostic_msgs"},
        )

    def test_missing_ros_has_an_actionable_lazy_import_error(self) -> None:
        with mock.patch(
            "deployment.robotlab_g1.ros2_node.importlib.import_module",
            side_effect=ModuleNotFoundError("rclpy"),
        ):
            with self.assertRaisesRegex(RuntimeError, "source /opt/ros/humble/setup.bash"):
                _load_ros2()


class PolicyTimingStatsTest(unittest.TestCase):
    def test_deadline_and_percentile_metrics(self) -> None:
        stats = PolicyTimingStats(0.02)
        for elapsed in (0.005, 0.010, 0.015, 0.025):
            stats.observe(elapsed)
        report = stats.as_dict()
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["deadline_misses"], 1)
        self.assertAlmostEqual(report["deadline_miss_rate"], 0.25)
        self.assertAlmostEqual(report["mean_ms"], 13.75)
        self.assertGreaterEqual(report["p95_ms"], 15.0)
        self.assertEqual(stats.diagnostics()["timing_deadline_misses"], "1")

    def test_invalid_timing_samples_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyTimingStats(0.0)
        stats = PolicyTimingStats(0.02)
        with self.assertRaises(ValueError):
            stats.observe(-0.001)


if __name__ == "__main__":
    unittest.main()
