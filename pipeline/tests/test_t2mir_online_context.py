"""CPU-only protocol tests for the RobotLab online DPT prompt adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.protocols.t2mir_online_context import (  # noqa: E402
    OnlineEpisodePromptBuffer,
    validate_checkpoint_contract,
)


class CheckpointSupervisionContractTest(unittest.TestCase):
    @staticmethod
    def checkpoint(supervision_mode: str) -> dict:
        return {
            "policy": {},
            "state_mean": torch.zeros(3),
            "state_std": torch.ones(3),
            "state_dim": 3,
            "action_dim": 2,
            "config": {
                "prompt_horizon": 64,
                "max_episode_steps": 64,
                "prompt_episode_horizon": 1,
                "supervision_mode": supervision_mode,
                "action_tanh": False,
                "moe_config": {
                    "num_experts": 4,
                    "num_selects": 2,
                    "num_experts_contrastive": 4,
                    "num_selects_contrastive": 2,
                },
            },
        }

    def validate(self, checkpoint: dict, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint identity")
            return validate_checkpoint_contract(path, checkpoint, **kwargs)

    def test_default_remains_strictly_all_tokens(self):
        provenance = self.validate(self.checkpoint("all_tokens"))
        self.assertEqual(provenance["supervision_mode"], "all_tokens")
        with self.assertRaisesRegex(ValueError, "expected one of"):
            self.validate(self.checkpoint("final_token"))

    def test_calibrated_evaluator_can_explicitly_accept_final_token(self):
        provenance = self.validate(
            self.checkpoint("final_token"),
            expected_supervision_modes=("all_tokens", "final_token"),
        )
        self.assertEqual(provenance["supervision_mode"], "final_token")

    def test_unknown_supervision_mode_is_never_accepted(self):
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint"):
            self.validate(
                self.checkpoint("unknown"),
                expected_supervision_modes=("all_tokens", "final_token"),
            )


class OnlineEpisodePromptBufferTest(unittest.TestCase):
    def make_buffer(
        self,
        *,
        num_envs: int = 2,
        horizon: int = 4,
        update_mode: str = "episode",
    ):
        return OnlineEpisodePromptBuffer(
            num_envs=num_envs,
            state_dim=3,
            action_dim=2,
            prompt_horizon=horizon,
            window="last",
            update_mode=update_mode,
            update_interval=horizon,
        )

    def append_step(self, buffer, step, active=(True, True), modes=(0, 1)):
        states = torch.tensor(
            [[step, 10 + step, 20 + step], [100 + step, 110 + step, 120 + step]],
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [[step, -step], [100 + step, -100 - step]], dtype=torch.float32
        )
        rewards = torch.tensor([step + 0.25, 100 + step + 0.25])
        buffer.append(states, actions, rewards, active, modes)

    def test_episode_zero_prompt_is_truly_empty(self):
        buffer = self.make_buffer()
        buffer.start_episode(0)
        group = buffer.prompt_batch()
        self.assertEqual(group.states.shape, (2, 0, 3))
        self.assertEqual(group.actions.shape, (2, 0, 2))
        self.assertEqual(group.rewards.shape, (2, 0, 1))
        self.assertEqual(group.source_episode_indices, (None, None))

    def test_prompt_is_fixed_until_finish_and_never_crosses_reset(self):
        buffer = self.make_buffer()
        buffer.start_episode(0)
        self.append_step(buffer, 0)
        self.append_step(buffer, 1)
        # Current transitions must not leak into the current episode prompt.
        self.assertEqual(buffer.prompt_batch().states.shape[1], 0)
        first = buffer.finish_episode()
        self.assertEqual([row.length for row in first], [2, 2])

        buffer.start_episode(1)
        prompt = buffer.prompt_batch()
        self.assertEqual(prompt.states.shape, (2, 2, 3))
        self.assertTrue(torch.equal(prompt.states[0, :, 0], torch.tensor([0.0, 1.0])))
        self.append_step(buffer, 9)
        # Still the previous episode, not a previous+current concatenation.
        self.assertTrue(
            torch.equal(buffer.prompt_batch().states[0, :, 0], torch.tensor([0.0, 1.0]))
        )
        second = buffer.finish_episode()
        self.assertEqual([row.length for row in second], [1, 1])

        buffer.start_episode(2)
        prompt = buffer.prompt_batch()
        self.assertEqual(prompt.states.shape[1], 1)
        self.assertEqual(float(prompt.states[0, 0, 0]), 9.0)

    def test_recent_window_and_controller_histogram(self):
        buffer = self.make_buffer(horizon=4)
        buffer.start_episode(0)
        for step in range(7):
            self.append_step(buffer, step, modes=(step % 2, 3))
        trajectories = buffer.finish_episode()
        self.assertEqual([row.length for row in trajectories], [7, 7])
        buffer.start_episode(1)
        prompt = buffer.prompt_batch()
        self.assertTrue(
            torch.equal(prompt.states[0, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
        )
        self.assertEqual(prompt.source_lengths, (7, 7))
        self.assertEqual(prompt.controller_mode_histograms[0], {0: 2, 1: 2})
        self.assertEqual(prompt.controller_mode_histograms[1], {3: 4})

    def test_parallel_envs_are_isolated_and_grouped_by_real_length(self):
        buffer = self.make_buffer()
        buffer.start_episode(0)
        self.append_step(buffer, 0, active=(True, True))
        self.append_step(buffer, 1, active=(True, False))
        self.append_step(buffer, 2, active=(True, False))
        buffer.finish_episode()
        buffer.start_episode(1)
        groups = buffer.prompt_groups()
        self.assertEqual([group.states.shape[1] for group in groups], [1, 3])
        self.assertEqual(groups[0].env_ids, (1,))
        self.assertEqual(groups[1].env_ids, (0,))
        self.assertEqual(float(groups[0].states[0, 0, 0]), 100.0)
        self.assertTrue(
            torch.equal(groups[1].states[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
        )

    def test_shape_and_nonfinite_inputs_are_rejected(self):
        buffer = self.make_buffer()
        buffer.start_episode(0)
        with self.assertRaises(ValueError):
            buffer.append(torch.zeros(1, 3), torch.zeros(2, 2), torch.zeros(2), [True, True])
        states = torch.zeros(2, 3)
        states[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            buffer.append(states, torch.zeros(2, 2), torch.zeros(2), [True, True])

    def test_block_mode_promotes_only_complete_causal_context(self):
        buffer = self.make_buffer(horizon=4, update_mode="block")
        buffer.start_episode(0)
        for step in range(3):
            self.append_step(buffer, step)
        before = buffer.prompt_groups(current_if_complete=True)
        self.assertEqual(before[0].states.shape[1], 0)

        self.append_step(buffer, 3)
        complete = buffer.prompt_groups(current_if_complete=True)
        self.assertEqual(complete[0].states.shape, (2, 4, 3))
        self.assertTrue(
            torch.equal(complete[0].states[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
        )
        self.assertEqual(complete[0].source_episode_indices, (0, 0))

        self.append_step(buffer, 4)
        self.append_step(buffer, 5)
        rolling = buffer.prompt_groups(current_if_complete=True)
        self.assertTrue(
            torch.equal(rolling[0].states[0, :, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
        )

    def test_block_mode_preserves_last_full_prompt_after_short_episode(self):
        buffer = self.make_buffer(horizon=4, update_mode="block")
        buffer.start_episode(0)
        for step in range(4):
            self.append_step(buffer, step)
        buffer.finish_episode()

        buffer.start_episode(1)
        self.append_step(buffer, 9)
        buffer.finish_episode()

        buffer.start_episode(2)
        prompt = buffer.prompt_batch()
        self.assertEqual(prompt.source_episode_indices, (0, 0))
        self.assertTrue(
            torch.equal(prompt.states[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
        )

    def test_explicit_task_boundary_discards_previous_prompt(self):
        buffer = self.make_buffer(horizon=4, update_mode="block")
        buffer.start_episode(0)
        for step in range(4):
            self.append_step(buffer, step)
        buffer.finish_episode()
        self.assertEqual(buffer.previous_trajectories[0].length, 4)

        buffer.reset()
        self.assertEqual(buffer.previous_trajectories, (None, None))
        buffer.start_episode(1)
        prompt = buffer.prompt_batch()
        self.assertEqual(prompt.states.shape, (2, 0, 3))
        buffer.finish_episode()

    def test_task_boundary_cannot_interrupt_active_episode(self):
        buffer = self.make_buffer()
        buffer.start_episode(0)
        with self.assertRaisesRegex(RuntimeError, "active episode"):
            buffer.reset()
        buffer.finish_episode()

    def test_invalid_update_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            OnlineEpisodePromptBuffer(2, 3, 2, update_mode="unknown")
        with self.assertRaises(ValueError):
            OnlineEpisodePromptBuffer(2, 3, 2, update_interval=0)


if __name__ == "__main__":
    unittest.main()
