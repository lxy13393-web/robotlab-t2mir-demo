"""Read-only deployment readiness report; never starts ROS or MuJoCo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from typing import Any

from .assets import (
    MJCFAssetError,
    validate_g1_mjcf,
    validate_scene_robot_binding,
)
from .mujoco_runner import MujocoBindings
from .control import PositionPDController
from .ros2_node import ros2_environment_report
from .run_mujoco import (
    DEFAULT_PILOT3_T2MIR,
    DEFAULT_PILOT3_T2MIR_SHA256,
    DEFAULT_PPO_ONNX,
    DEFAULT_PPO_ONNX_SHA256,
)


def _model_status(path: Path, expected_sha256: str) -> dict[str, Any]:
    exists = path.is_file()
    actual = None
    if exists:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "sha256_matches": exists and actual == expected_sha256,
    }


def _compiled_mujoco_status(
    scene_mjcf: Path,
) -> dict[str, Any]:
    """Compile the exact scene and run all runtime semantic gates, without stepping."""

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_mjcf))
        bindings = MujocoBindings(mujoco, model, integration_dt=0.001)
        controller = PositionPDController()
        bindings.configure_implicit_position_pd(
            controller.gains, controller.effective_effort_limit
        )
    except Exception as exc:  # readiness report must return an actionable result
        return {
            "attempted": True,
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "attempted": True,
        "valid": True,
        "bindings": asdict(bindings.metadata),
        "runtime_actuator_mode": "implicit_pd",
        "mujoco_integrator": int(model.opt.integrator),
    }


def build_report(
    robot_mjcf: Path | None = None,
    scene_mjcf: Path | None = None,
) -> dict[str, Any]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "onnxruntime", "torch", "mujoco")
    }
    models = {
        "ppo_onnx": _model_status(DEFAULT_PPO_ONNX, DEFAULT_PPO_ONNX_SHA256),
        "t2mir_pilot3": _model_status(
            DEFAULT_PILOT3_T2MIR, DEFAULT_PILOT3_T2MIR_SHA256
        )
        | {"claim_scope": "interface-smoke-only"},
    }
    asset: dict[str, Any] = {"provided": robot_mjcf is not None}
    if robot_mjcf is not None:
        try:
            report = validate_g1_mjcf(robot_mjcf)
        except (MJCFAssetError, OSError) as exc:
            asset.update({"valid": False, "error": str(exc)})
        else:
            asset.update({"valid": True, **report.as_dict()})
    scene: dict[str, Any] = {"provided": scene_mjcf is not None}
    if scene_mjcf is not None:
        if robot_mjcf is None:
            scene.update(
                {
                    "valid": False,
                    "error": "--scene-mjcf requires --robot-mjcf",
                }
            )
        else:
            try:
                binding = validate_scene_robot_binding(scene_mjcf, robot_mjcf)
            except (MJCFAssetError, OSError) as exc:
                scene.update({"valid": False, "error": str(exc)})
            else:
                scene.update({"valid": True, **binding.as_dict()})
    compiled: dict[str, Any] = {"attempted": False, "valid": False}
    if (
        dependencies["mujoco"]
        and bool(asset.get("valid", False))
        and bool(scene.get("valid", False))
        and scene_mjcf is not None
    ):
        compiled = _compiled_mujoco_status(scene_mjcf.expanduser().resolve())
    return {
        "format_version": 1,
        "dependencies": dependencies,
        "ros2": ros2_environment_report(),
        "models": models,
        "robot_mjcf": asset,
        "scene_mjcf": scene,
        "compiled_mujoco_contract": compiled,
        "ready_for_policy_cpu_smoke": (
            dependencies["numpy"]
            and dependencies["onnxruntime"]
            and models["ppo_onnx"]["sha256_matches"]
        ),
        "ready_for_real_mujoco_smoke": (
            dependencies["mujoco"]
            and bool(asset.get("valid", False))
            and bool(scene.get("valid", False))
            and bool(compiled.get("valid", False))
            and bool(models["ppo_onnx"]["sha256_matches"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-mjcf", type=Path, default=None)
    parser.add_argument("--scene-mjcf", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return exit status 2 unless the exact scene compiles and all real-MuJoCo "
            "contract gates pass. The JSON report is always printed."
        ),
    )
    args = parser.parse_args(argv)
    report = build_report(args.robot_mjcf, args.scene_mjcf)
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and not report["ready_for_real_mujoco_smoke"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
