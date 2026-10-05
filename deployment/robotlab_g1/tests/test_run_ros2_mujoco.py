from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from deployment.robotlab_g1.contract import DEFAULT_CONTRACT
from deployment.robotlab_g1.control import DynamicsParameters
from deployment.robotlab_g1.run_ros2_mujoco import (
    build_parser,
    build_runtime_report,
    write_runtime_report,
)


@dataclass(frozen=True)
class FakeBindingsMetadata:
    base_body_id: int = 1


class FakeReport:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def as_dict(self):
        return {"kind": self.kind, "valid": True}


class Ros2MujocoRuntimeReportTest(unittest.TestCase):
    def test_cli_has_a_stable_default_output(self) -> None:
        args = build_parser().parse_args(
            ["--mjcf", "scene.xml", "--robot-mjcf", "robot.xml"]
        )
        self.assertEqual(
            args.output, Path("outputs/deployment/ros2_mujoco_runtime.json")
        )
        self.assertEqual(args.max_runtime_seconds, 0.0)

    def test_cli_accepts_finite_runtime_smoke(self) -> None:
        args = build_parser().parse_args(
            [
                "--mjcf",
                "scene.xml",
                "--robot-mjcf",
                "robot.xml",
                "--max-runtime-seconds",
                "2.5",
            ]
        )
        self.assertEqual(args.max_runtime_seconds, 2.5)

    def test_report_records_contract_model_scene_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            scene.write_text("<mujoco/>")
            plant = SimpleNamespace(
                contract=DEFAULT_CONTRACT,
                dynamics=DynamicsParameters(),
                reward_provider=SimpleNamespace(schema_id="proxy-v1"),
                bindings=SimpleNamespace(metadata=FakeBindingsMetadata()),
            )
            backend = SimpleNamespace(
                provenance={"backend": "fake", "model_sha256": "abc"}
            )
            payload = build_runtime_report(
                backend=backend,
                plant=plant,
                runtime_snapshot={"timing": {"p95_ms": 4.0}},
                robot_asset_report=FakeReport("robot"),
                scene_binding_report=FakeReport("scene"),
                scene_path=scene,
                shutdown_reason="test",
            )
            output = write_runtime_report(Path(directory) / "report.json", payload)
            self.assertTrue(output.is_file())
            self.assertEqual(payload["contract"]["sha256"], DEFAULT_CONTRACT.sha256)
            self.assertEqual(payload["policy"]["model_sha256"], "abc")
            self.assertEqual(payload["runtime"]["timing"]["p95_ms"], 4.0)
            self.assertEqual(payload["shutdown_reason"], "test")


if __name__ == "__main__":
    unittest.main()
