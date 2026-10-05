from __future__ import annotations

import unittest

from deployment.robotlab_g1.evaluate_mujoco import (
    GoalEvaluationConfig,
    GoalScenario,
    MujocoGoalEvaluator,
    generate_scenarios,
    scenario_sha256,
    summarize,
    validate_reward_claim,
)
from deployment.robotlab_g1.mujoco_runner import (
    MujocoDeploymentRunner,
    MujocoEpisodeConfig,
)
from deployment.robotlab_g1.tests.test_mujoco_runner import (
    _FakeData,
    _FakeModel,
    _FakeMujoco,
    _ZeroPolicy,
)
from pipeline.protocols.dynamics_profile import DynamicsProfile


class MujocoFormalEvaluationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = [
            DynamicsProfile(5, 0.2, 0.9, 0.0, 1.0, "eval"),
            DynamicsProfile(14, 0.32, 0.8, 3.0, 0.65, "eval"),
        ]

    def test_scenarios_are_repeatable_and_subset_invariant(self) -> None:
        full = generate_scenarios(
            self.profiles, episodes=3, seed=42, min_distance=1.0, max_distance=4.0
        )
        repeated = generate_scenarios(
            self.profiles, episodes=3, seed=42, min_distance=1.0, max_distance=4.0
        )
        subset = generate_scenarios(
            [self.profiles[1]], episodes=3, seed=42, min_distance=1.0, max_distance=4.0
        )
        self.assertEqual(full, repeated)
        self.assertEqual([row for row in full if row.profile_id == 14], subset)
        self.assertEqual(scenario_sha256(full), scenario_sha256(repeated))

    def test_reward_claim_blocks_formal_t2mir_proxy(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-parity"):
            validate_reward_claim({"backend": "t2mir"}, False)
        self.assertIn("preliminary", validate_reward_claim({"backend": "t2mir"}, True))
        self.assertIn("reward-independent", validate_reward_claim({"backend": "ppo"}, False))

    def test_summary_uses_success_as_primary_and_reports_adaptation(self) -> None:
        rows = [
            {"profile_id": 5, "episode_index": 0, "success": 0, "fall": 1, "timeout": 0,
             "episode_return": 100.0, "position_error": 1.0, "yaw_error": 0.5},
            {"profile_id": 5, "episode_index": 1, "success": 1, "fall": 0, "timeout": 0,
             "episode_return": 1.0, "position_error": 0.1, "yaw_error": 0.05},
        ]
        report = summarize(rows, [5], 2)
        self.assertEqual(report["overall"]["success_rate"], 0.5)
        self.assertEqual(report["overall"]["adaptation_success_delta_last_minus_first"], 1.0)
        self.assertEqual(report["per_episode"][0]["success_rate"], 0.0)
        self.assertEqual(report["per_episode"][1]["success_rate"], 1.0)

    def test_goal_loop_records_official_transition_and_success_contract(self) -> None:
        model = _FakeModel()
        data = _FakeData(model)
        policy = _ZeroPolicy()
        runner = MujocoDeploymentRunner(
            _FakeMujoco,
            model,
            data,
            policy,
            episode_config=MujocoEpisodeConfig(max_policy_steps=5),
        )
        evaluator = MujocoGoalEvaluator(
            runner,
            GoalEvaluationConfig(timeout_seconds=0.1, hold_seconds=0.04),
        )
        result = evaluator.run(GoalScenario(5, 0, 0.0, 0.0, 0.0))
        self.assertEqual(result["result"], "success")
        self.assertEqual(result["policy_steps"], 3)
        self.assertEqual(result["recorded_episode_length"], 3)
        self.assertEqual(len(policy.records), 3)
        self.assertFalse(policy.active)
        self.assertEqual(policy.records[-1][3], 3)  # HOLD controller mode


if __name__ == "__main__":
    unittest.main()
