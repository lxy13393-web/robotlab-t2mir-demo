"""CPU-only tests for the fair stationary PPO evaluation contract."""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.protocols.ppo_online_protocol import (  # noqa: E402
    DEFAULT_EPISODES,
    DEFAULT_NUM_ENVS,
    freeze_evaluation_reset,
    generate_t2mir_identical_scenarios,
    no_context_prompt_protocol,
    ppo_prompt_protocol,
    prepare_static_contract,
    summarize,
    synchronize_vector_episode_boundary,
    validate_protocol_values,
)


class ScenarioParityTest(unittest.TestCase):
    def test_formal_scenario_is_bit_identical_to_t2mir_algorithm(self):
        actual, actual_sha = generate_t2mir_identical_scenarios(
            seed=42,
            episodes=DEFAULT_EPISODES,
            num_envs=DEFAULT_NUM_ENVS,
            goal_min_distance=1.0,
            goal_max_distance=4.0,
        )

        # Independent transcription of evaluate_t2mir_online.py.  In
        # particular, this must remain three sequential draws of shape
        # (episodes, num_envs), not one draw with a trailing dimension of 3.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        shape = (8, 32)
        distances = 1.0 + (4.0 - 1.0) * torch.rand(shape, generator=generator)
        bearings = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
        relative_yaws = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
        expected = torch.stack(
            (distances * torch.cos(bearings), distances * torch.sin(bearings), relative_yaws),
            dim=-1,
        )
        expected_bytes = expected.numpy().tobytes()

        self.assertEqual(tuple(actual.shape), (8, 32, 3))
        self.assertEqual(actual.numpy().tobytes(), expected_bytes)
        self.assertEqual(actual_sha, hashlib.sha256(expected_bytes).hexdigest())
        # Locks the formal seed/shape/range contract against coordinated edits
        # to both the helper and the reference transcription above.
        self.assertEqual(
            actual_sha,
            "d8eff987ba530d532c1d365d910787e90f49b756281b72bac695eaf15bff4e98",
        )


class CliGuardTest(unittest.TestCase):
    def valid_values(self) -> dict:
        return {
            "episodes": 8,
            "num_envs": 32,
            "goal_min_distance": 1.0,
            "goal_max_distance": 4.0,
            "goal_timeout": 40.0,
            "goal_hold_time": 2.0,
        }

    def test_invalid_scalar_contracts_are_rejected(self):
        mutations = (
            ("episodes", 1),
            ("num_envs", 0),
            ("goal_min_distance", 0.0),
            ("goal_max_distance", 1.0),
            ("goal_timeout", 0.0),
            ("goal_hold_time", -1.0),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                arguments = self.valid_values()
                arguments[key] = value
                with self.assertRaises(ValueError):
                    validate_protocol_values(**arguments)

    def test_training_profile_is_rejected_before_simulator_launch(self):
        manifest = REPO / "configs/g1_dynamics_48.json"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint identity")
            with self.assertRaisesRegex(ValueError, "requires split='eval'"):
                prepare_static_contract(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=None,
                    manifest=manifest,
                    profile_id=0,
                )

    def test_heldout_profile_and_checkpoint_digest_are_frozen(self):
        manifest = REPO / "configs/g1_dynamics_48.json"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint identity")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            contract = prepare_static_contract(
                checkpoint=checkpoint,
                expected_checkpoint_sha256=digest.upper(),
                manifest=manifest,
                profile_id=5,
            )
            self.assertEqual(contract.profile.split, "eval")
            self.assertEqual(contract.profile.task_id, 5)
            self.assertEqual(contract.checkpoint_sha256, digest)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_static_contract(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256="0" * 64,
                    manifest=manifest,
                    profile_id=5,
                )

    def test_ppo_protocol_never_updates_a_prompt(self):
        protocol = ppo_prompt_protocol()
        self.assertEqual(protocol["prompt_horizon"], 0)
        self.assertFalse(protocol["cross_reset_transitions_allowed"])
        self.assertFalse(protocol["online_prompt_updates_within_episode"])
        self.assertFalse(protocol["online_prompt_updates_between_episodes"])

    def test_query_only_protocol_never_receives_context(self):
        protocol = no_context_prompt_protocol(
            policy_kind="query_only",
            action_semantics="deterministic_supervised_actor_mean",
        )
        self.assertEqual(protocol["protocol"], "no-prompt-fixed-query_only")
        self.assertEqual(protocol["prompt_horizon"], 0)
        self.assertEqual(
            protocol["action_semantics"], "deterministic_supervised_actor_mean"
        )
        self.assertFalse(protocol["cross_reset_transitions_allowed"])
        self.assertFalse(protocol["online_prompt_updates_within_episode"])
        self.assertFalse(protocol["online_prompt_updates_between_episodes"])

    def test_reset_contract_removes_action_dependent_rng_drift(self):
        class Term:
            def __init__(self, params):
                self.params = params

        class Events:
            reset_base = Term(
                {
                    "pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {"x": (-0.5, 0.5)},
                }
            )
            reset_robot_joints = Term(
                {"position_range": (0.5, 1.5), "velocity_range": (-1.0, 1.0)}
            )

        class Cfg:
            events = Events()

        contract = freeze_evaluation_reset(Cfg())
        self.assertEqual(contract["root_pose_range"]["yaw"], (0.0, 0.0))
        self.assertEqual(contract["root_velocity_range"]["pitch"], (0.0, 0.0))
        self.assertEqual(contract["joint_position_scale_range"], (1.0, 1.0))
        self.assertEqual(contract["joint_velocity_range"], (0.0, 0.0))


class EpisodeBoundaryTest(unittest.TestCase):
    class CommandTerm:
        def __init__(self, num_envs: int):
            self.vel_command_b = torch.ones(num_envs, 3)

    class FakeEnv:
        def __init__(self, num_envs: int = 3, num_actions: int = 2):
            self.num_envs = num_envs
            self.num_actions = num_actions
            self.max_episode_length = 10
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.step_calls = 0
            self.last_actions = None

        def step(self, actions):
            self.step_calls += 1
            self.last_actions = actions.clone()
            self.episode_length_buf += 1
            time_outs = self.episode_length_buf >= self.max_episode_length
            dones = time_outs.to(dtype=torch.long)
            self.episode_length_buf[time_outs] = 0
            observations = torch.zeros(self.num_envs, 4)
            rewards = torch.zeros(self.num_envs)
            return observations, rewards, dones, {"time_outs": time_outs}

    def test_episode_zero_consumes_wrapper_constructor_reset_without_step(self):
        env = self.FakeEnv()
        command = self.CommandTerm(env.num_envs)
        audit = synchronize_vector_episode_boundary(
            env=env, command_term=command, episode_index=0
        )
        self.assertEqual(audit["mechanism"], "rsl_wrapper_constructor_reset")
        self.assertEqual(env.step_calls, 0)

    def test_later_episode_uses_one_excluded_timeout_step(self):
        env = self.FakeEnv()
        env.episode_length_buf[:] = torch.tensor([2, 5, 8])
        command = self.CommandTerm(env.num_envs)
        audit = synchronize_vector_episode_boundary(
            env=env, command_term=command, episode_index=1
        )
        self.assertEqual(
            audit["mechanism"], "manager_based_rl_forced_timeout_auto_reset"
        )
        self.assertEqual(env.step_calls, 1)
        self.assertTrue(
            torch.equal(command.vel_command_b, torch.zeros_like(command.vel_command_b))
        )
        self.assertTrue(torch.equal(env.last_actions, torch.zeros_like(env.last_actions)))
        self.assertTrue(
            torch.equal(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))
        )


class SummaryParityTest(unittest.TestCase):
    def test_summary_matches_t2mir_schema_and_episode_math(self):
        rows = [
            {
                "episode_index": 0,
                "result": "success",
                "episode_return": 10.0,
                "position_error": 0.1,
                "yaw_error": 0.2,
            },
            {
                "episode_index": 0,
                "result": "fall",
                "episode_return": -2.0,
                "position_error": 2.1,
                "yaw_error": 1.2,
            },
            {
                "episode_index": 1,
                "result": "success",
                "episode_return": 12.0,
                "position_error": 0.2,
                "yaw_error": 0.1,
            },
            {
                "episode_index": 1,
                "result": "timeout",
                "episode_return": 4.0,
                "position_error": 1.0,
                "yaw_error": 0.5,
            },
        ]
        result = summarize(rows, episodes=2)

        self.assertEqual(result["episodes_per_replica"], 2)
        self.assertEqual(result["replicas"], 2)
        self.assertEqual(result["total_episodes"], 4)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["fall_rate"], 0.25)
        self.assertEqual(result["per_episode"][0]["falls"], 1)
        self.assertEqual(result["per_episode"][1]["timeouts"], 1)
        self.assertEqual(result["per_episode"][0]["mean_return"], 4.0)
        self.assertEqual(result["per_episode"][1]["mean_return"], 8.0)
        self.assertEqual(result["adaptation_success_delta_last_minus_first"], 0.0)
        self.assertEqual(result["adaptation_return_delta_last_minus_first"], 4.0)


if __name__ == "__main__":
    unittest.main()
