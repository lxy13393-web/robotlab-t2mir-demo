from __future__ import annotations

import unittest

from deployment.robotlab_g1.run_mujoco import (
    _validate_policy_source,
    _validate_reward_claim,
    build_parser,
)


class DeploymentCliGuardTest(unittest.TestCase):
    def _args(self, *extra: str):
        return build_parser().parse_args(
            ["--mjcf", "scene.xml", "--robot-mjcf", "robot.xml", *extra]
        )

    def test_ppo_does_not_need_proxy_reward_acknowledgement(self) -> None:
        _validate_reward_claim(self._args("--backend", "ppo"))

    def test_t2mir_requires_explicit_proxy_reward_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "T2MIR consumes reward"):
            _validate_reward_claim(self._args("--backend", "t2mir"))
        _validate_reward_claim(
            self._args("--backend", "t2mir", "--allow-proxy-reward")
        )

    def test_registered_model_source_cannot_mix_with_direct_flags(self) -> None:
        args = self._args(
            "--model-manifest", "active.json", "--backend", "ppo"
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_policy_source(args)

    def test_video_cli_defaults_and_overrides(self) -> None:
        defaults = self._args("--video")
        self.assertTrue(defaults.video)
        self.assertEqual(defaults.video_fps, 30.0)
        self.assertEqual((defaults.video_width, defaults.video_height), (1280, 720))
        custom = self._args(
            "--video",
            "--video-output",
            "demo.mp4",
            "--video-fps",
            "25",
            "--video-width",
            "960",
            "--video-height",
            "540",
        )
        self.assertEqual(str(custom.video_output), "demo.mp4")
        self.assertEqual(custom.video_fps, 25.0)
        self.assertEqual((custom.video_width, custom.video_height), (960, 540))


if __name__ == "__main__":
    unittest.main()
