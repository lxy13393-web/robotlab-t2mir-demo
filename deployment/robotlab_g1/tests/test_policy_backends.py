"""CPU-only tests for the RobotLab G1 policy backend boundary."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.robotlab_g1.policy_backends import (  # noqa: E402
    G1_ACTION_DIM,
    G1_OBSERVATION_DIM,
    PPOOnnxBackend,
    T2MIRTorchBackend,
)


PPO_MODEL = (
    REPOSITORY_ROOT
    / "logs/rsl_rl/unitree_g1_flat/2026-08-30_23-35-43/exported/policy.onnx"
)
PILOT3_CHECKPOINT = (
    REPOSITORY_ROOT
    / "methods/t2mir/runs/RobotLab-G1-MultiDynamics"
    / "official_mixed_v1_pilot3_smoke/best.pt"
)
FINAL_TOKEN_D_CHECKPOINT = (
    REPOSITORY_ROOT
    / "methods/t2mir/runs/RobotLab-G1-MultiDynamics"
    / "actor_mean_control/D_seed42/best.pt"
)
T2MIR_ROOT = REPOSITORY_ROOT / "methods/t2mir"
HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_ONNXRUNTIME and PPO_MODEL.is_file(), "PPO ONNX fixture unavailable")
class PPOOnnxBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = PPOOnnxBackend(PPO_MODEL)

    def test_real_model_cpu_smoke_and_rank_preservation(self) -> None:
        vector_actions = self.backend.predict(np.zeros(G1_OBSERVATION_DIM, dtype=np.float32))
        batch_actions = self.backend.predict(np.zeros((1, G1_OBSERVATION_DIM), dtype=np.float64))

        self.assertEqual(vector_actions.shape, (G1_ACTION_DIM,))
        self.assertEqual(batch_actions.shape, (1, G1_ACTION_DIM))
        self.assertEqual(vector_actions.dtype, np.float32)
        self.assertTrue(np.isfinite(vector_actions).all())
        np.testing.assert_allclose(vector_actions, batch_actions[0], rtol=0.0, atol=0.0)
        self.assertEqual(self.backend.provenance["provider"], "CPUExecutionProvider")
        self.assertEqual(len(self.backend.provenance["model_sha256"]), 64)

    def test_bad_observation_is_rejected_before_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self.backend.predict(np.zeros(G1_OBSERVATION_DIM - 1, dtype=np.float32))
        observation = np.zeros(G1_OBSERVATION_DIM, dtype=np.float32)
        observation[7] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.backend.predict(observation)

    def test_expected_hash_is_an_enforced_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            PPOOnnxBackend(PPO_MODEL, expected_sha256="0" * 64)


@unittest.skipUnless(
    HAS_TORCH and PILOT3_CHECKPOINT.is_file() and T2MIR_ROOT.is_dir(),
    "pilot3 T2MIR fixture unavailable",
)
class T2MIRTorchBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = T2MIRTorchBackend(
            PILOT3_CHECKPOINT,
            t2mir_root=T2MIR_ROOT,
            device="cpu",
        )

    def test_a_pilot3_checkpoint_online_episode_protocol(self) -> None:
        backend = self.backend
        prompts = backend.start_episode(0)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(tuple(prompts[0].states.shape), (1, 0, G1_OBSERVATION_DIM))

        state = np.zeros(G1_OBSERVATION_DIM, dtype=np.float32)
        action = backend.predict(state)
        self.assertEqual(action.shape, (G1_ACTION_DIM,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.isfinite(action).all())
        backend.record_transition(state, action, 0.25, controller_modes=2)
        trajectories = backend.finish_episode()
        self.assertEqual(trajectories[0].length, 1)

        prompts = backend.start_episode()
        self.assertEqual(tuple(prompts[0].states.shape), (1, 1, G1_OBSERVATION_DIM))
        self.assertEqual(prompts[0].source_episode_indices, (0,))
        next_action = backend.predict(state)
        self.assertEqual(next_action.shape, (G1_ACTION_DIM,))
        self.assertTrue(np.isfinite(next_action).all())
        backend.finish_episode()

        protocol = backend.provenance["prompt_protocol"]
        self.assertEqual(protocol["protocol"], "previous-own-episode-fixed-prompt")
        self.assertEqual(protocol["prompt_horizon"], 64)
        self.assertFalse(protocol["cross_reset_transitions_allowed"])

    def test_b_start_and_input_guards(self) -> None:
        backend = self.backend
        with self.assertRaisesRegex(RuntimeError, "start_episode"):
            backend.predict(np.zeros(G1_OBSERVATION_DIM, dtype=np.float32))

        backend.start_episode(2)
        with self.assertRaisesRegex(ValueError, "shape"):
            backend.predict(np.zeros(G1_OBSERVATION_DIM - 1, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "policy_actions shape"):
            backend.record_transition(
                np.zeros(G1_OBSERVATION_DIM, dtype=np.float32),
                np.zeros(G1_ACTION_DIM - 1, dtype=np.float32),
                0.0,
            )
        backend.finish_episode()

    def test_c_block_protocol_refreshes_after_recorded_boundary(self) -> None:
        backend = T2MIRTorchBackend(
            PILOT3_CHECKPOINT,
            t2mir_root=T2MIR_ROOT,
            device="cpu",
            update_mode="block",
            update_interval=64,
        )
        prompts = backend.start_episode(0)
        self.assertEqual(tuple(prompts[0].states.shape), (1, 0, G1_OBSERVATION_DIM))
        state = np.zeros(G1_OBSERVATION_DIM, dtype=np.float32)
        action = backend.predict(state)
        for _ in range(64):
            backend.record_transition(state, action, 0.25, controller_modes=2)
        refreshed = backend._policy._episode_prompts
        self.assertEqual(tuple(refreshed[0].states.shape), (1, 64, G1_OBSERVATION_DIM))
        protocol = backend.provenance["prompt_protocol"]
        self.assertEqual(protocol["protocol"], "causal-own-rollout-block-prompt")
        self.assertEqual(protocol["update_interval"], 64)
        backend.finish_episode()


@unittest.skipUnless(
    HAS_TORCH and FINAL_TOKEN_D_CHECKPOINT.is_file() and T2MIR_ROOT.is_dir(),
    "final-token D T2MIR fixture unavailable",
)
class FinalTokenT2MIRTorchBackendTest(unittest.TestCase):
    def test_formal_d_checkpoint_cpu_smoke(self) -> None:
        backend = T2MIRTorchBackend(
            FINAL_TOKEN_D_CHECKPOINT,
            t2mir_root=T2MIR_ROOT,
            device="cpu",
        )
        backend.start_episode(0)
        action = backend.predict(np.zeros(G1_OBSERVATION_DIM, dtype=np.float32))
        self.assertEqual(action.shape, (G1_ACTION_DIM,))
        self.assertTrue(np.isfinite(action).all())
        self.assertEqual(backend.provenance["supervision_mode"], "final_token")
        backend.finish_episode()


if __name__ == "__main__":
    unittest.main()
