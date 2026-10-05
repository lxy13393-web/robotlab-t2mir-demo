from __future__ import annotations

import unittest

import numpy as np

from deployment.robotlab_g1.contract import DEFAULT_CONTRACT
from deployment.robotlab_g1.control import (
    DynamicsParameters,
    FirstOrderActionLag,
    PositionPDController,
    robotlab_g1_pd_gains,
)


class ActionLagTest(unittest.TestCase):
    def test_filter_matches_robotlab_and_resets(self):
        lag = FirstOrderActionLag(0.4, action_dim=2)
        np.testing.assert_array_equal(lag.apply([1.0, -1.0]), [1.0, -1.0])
        np.testing.assert_allclose(lag.apply([0.0, 1.0]), [0.4, 0.2], atol=1e-7)
        lag.reset()
        np.testing.assert_array_equal(lag.apply([0.0, 1.0]), [0.0, 1.0])

    def test_invalid_dynamics_are_rejected(self):
        for kwargs in (
            {"action_lag": 1.0},
            {"motor_strength": 0.0},
            {"payload_kg": -1.0},
            {"friction": 0.0},
        ):
            with self.assertRaises(ValueError):
                DynamicsParameters(**kwargs)


class PDControllerTest(unittest.TestCase):
    def test_gain_groups_match_training_asset(self):
        gains = robotlab_g1_pd_gains()
        by_name = {
            name: (gains.stiffness[i], gains.damping[i], gains.effort_limit[i])
            for i, name in enumerate(DEFAULT_CONTRACT.joint_names)
        }
        self.assertEqual(by_name["left_ankle_pitch_joint"], (20.0, 2.0, 20.0))
        self.assertEqual(by_name["left_hip_roll_joint"], (150.0, 5.0, 300.0))
        self.assertEqual(by_name["left_knee_joint"], (200.0, 5.0, 300.0))
        self.assertEqual(by_name["left_five_joint"], (40.0, 10.0, 300.0))

    def test_zero_action_at_default_is_zero_torque(self):
        controller = PositionPDController()
        result = controller.compute(
            np.zeros(37),
            np.asarray(DEFAULT_CONTRACT.default_joint_positions),
            np.zeros(37),
        )
        np.testing.assert_array_equal(result.torques, np.zeros(37))
        np.testing.assert_allclose(
            result.joint_targets, DEFAULT_CONTRACT.default_joint_positions, atol=1e-7
        )

    def test_effort_limits_apply_motor_strength_and_model_limit(self):
        model_limits = np.full(37, 50.0, dtype=np.float32)
        controller = PositionPDController(
            motor_strength=0.8, actuator_limits=model_limits
        )
        result = controller.compute(np.full(37, 100.0), np.zeros(37), np.zeros(37))
        self.assertTrue((np.abs(result.torques) <= 50.0).all())
        ankle = DEFAULT_CONTRACT.joint_names.index("left_ankle_pitch_joint")
        self.assertAlmostEqual(abs(float(result.torques[ankle])), 16.0)


if __name__ == "__main__":
    unittest.main()
