from __future__ import annotations

from pathlib import Path
from unittest import mock
import tempfile
import unittest

from deployment.robotlab_g1.preflight import build_report, main
from deployment.robotlab_g1.tests.test_assets import make_mjcf


class PreflightTest(unittest.TestCase):
    def test_scene_is_required_before_real_mujoco_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "robot.xml"
            robot.write_text(make_mjcf())
            report = build_report(robot)
        self.assertTrue(report["robot_mjcf"]["valid"])
        self.assertFalse(report["scene_mjcf"]["provided"])
        self.assertFalse(report["compiled_mujoco_contract"]["attempted"])
        self.assertFalse(report["ready_for_real_mujoco_smoke"])

    def test_scene_without_robot_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            scene.write_text('<mujoco><worldbody/></mujoco>')
            report = build_report(scene_mjcf=scene)
        self.assertFalse(report["scene_mjcf"]["valid"])
        self.assertIn("requires --robot-mjcf", report["scene_mjcf"]["error"])

    def test_strict_mode_returns_nonzero_when_real_smoke_is_not_ready(self) -> None:
        with mock.patch(
            "deployment.robotlab_g1.preflight.build_report",
            return_value={"ready_for_real_mujoco_smoke": False},
        ):
            self.assertEqual(main(["--strict"]), 2)

    def test_non_strict_mode_remains_a_read_only_report(self) -> None:
        with mock.patch(
            "deployment.robotlab_g1.preflight.build_report",
            return_value={"ready_for_real_mujoco_smoke": False},
        ):
            self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
