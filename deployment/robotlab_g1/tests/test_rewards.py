from __future__ import annotations

import unittest

import numpy as np

from deployment.robotlab_g1.rewards import (
    ExternalReward,
    ObservableVelocityReward,
    RewardInput,
)


def sample(**overrides) -> RewardInput:
    values = {
        "base_linear_velocity": np.array([0.5, 0.0, 0.0], dtype=np.float32),
        "base_angular_velocity": np.array([0.0, 0.0, 0.2], dtype=np.float32),
        "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "command": np.array([0.5, 0.0, 0.2], dtype=np.float32),
        "action": np.zeros(37, dtype=np.float32),
        "previous_action": np.zeros(37, dtype=np.float32),
        "joint_torque": np.zeros(37, dtype=np.float32),
    }
    values.update(overrides)
    return RewardInput(**values)


class RewardTest(unittest.TestCase):
    def test_tracking_reward_is_deterministic_and_versioned(self):
        provider = ObservableVelocityReward()
        self.assertAlmostEqual(provider.compute(sample()), 0.04, places=6)
        self.assertTrue(provider.schema_id.startswith("robotlab-g1-reward-v1:"))
        self.assertEqual(provider.schema_id, ObservableVelocityReward().schema_id)

    def test_fall_and_action_change_reduce_reward(self):
        provider = ObservableVelocityReward()
        nominal = provider.compute(sample())
        degraded = provider.compute(
            sample(action=np.ones(37, dtype=np.float32), terminated=True)
        )
        self.assertLess(degraded, nominal)

    def test_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            sample(action=np.zeros(29, dtype=np.float32))
        bad = np.zeros(37, dtype=np.float32)
        bad[0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            sample(action=bad)

    def test_external_reward_is_consumed_once(self):
        provider = ExternalReward("test/reward/v1")
        provider.update(1.25)
        self.assertEqual(provider.compute(sample()), 1.25)
        with self.assertRaisesRegex(RuntimeError, "no external reward"):
            provider.compute(sample())


if __name__ == "__main__":
    unittest.main()
