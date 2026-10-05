import unittest

from pipeline.protocols.goal_controller import GoalController, GoalControllerCfg


class GoalControllerTurnEnvelopeTests(unittest.TestCase):
    def test_optional_arc_speed_is_symmetric(self):
        cfg = GoalControllerCfg(turn_forward_speed=0.4, align_forward_speed=0.2)
        left = GoalController(0.0, 2.0, 0.0, cfg)
        right = GoalController(0.0, -2.0, 0.0, cfg)

        left_command, _ = left.compute(0.0, 0.0, 0.0)
        right_command, _ = right.compute(0.0, 0.0, 0.0)

        self.assertEqual(left.state, GoalController.TURN_TO_GOAL)
        self.assertEqual(right.state, GoalController.TURN_TO_GOAL)
        self.assertAlmostEqual(left_command[0], 0.4)
        self.assertAlmostEqual(right_command[0], 0.4)
        self.assertGreater(left_command[2], 0.0)
        self.assertLess(right_command[2], 0.0)

    def test_default_preserves_in_place_turn(self):
        controller = GoalController(0.0, 2.0, 0.0)
        command, _ = controller.compute(0.0, 0.0, 0.0)
        self.assertEqual(command[0], 0.0)

    def test_final_alignment_uses_smaller_independent_arc(self):
        cfg = GoalControllerCfg(turn_forward_speed=0.4, align_forward_speed=0.2)
        controller = GoalController(0.0, 0.0, 1.0, cfg)
        command, _ = controller.compute(0.0, 0.0, 0.0)
        self.assertEqual(controller.state, GoalController.ALIGN_FINAL_YAW)
        self.assertAlmostEqual(command[0], 0.2)
        self.assertGreater(command[2], 0.0)


if __name__ == "__main__":
    unittest.main()
