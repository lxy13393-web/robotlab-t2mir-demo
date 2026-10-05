"""ROS 2 Humble + MuJoCo minimum closed-loop entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .assets import validate_g1_mjcf, validate_scene_robot_binding
from .control import DynamicsParameters
from .mujoco_ros2 import MujocoRos2Plant, make_mujoco_step_handler
from .mujoco_runner import MujocoEpisodeConfig
from .ros2_node import (
    PolicyLoopConfig,
    Ros2PolicyCore,
    _load_ros2,
    create_ros2_node,
)
from .run_mujoco import _make_backend, _validate_reward_claim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimum ROS2 policy node against an in-process 37-DoF G1 MuJoCo plant."
    )
    parser.add_argument("--mjcf", type=Path, required=True, help="MuJoCo scene or flattened XML")
    parser.add_argument(
        "--robot-mjcf",
        type=Path,
        required=True,
        help="Generated include-free torque-motor robot XML bound to --mjcf",
    )
    parser.add_argument("--backend", choices=("ppo", "t2mir"), default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
        help="Registered artifact or active pointer; mutually exclusive with direct model flags.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-window", choices=("last", "first"), default="last")
    parser.add_argument(
        "--prompt-update-mode", choices=("episode", "block"), default="episode"
    )
    parser.add_argument("--prompt-update-interval", type=int, default=64)
    parser.add_argument("--action-lag", type=float, default=0.0)
    parser.add_argument("--motor-strength", type=float, default=1.0)
    parser.add_argument("--payload-kg", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--fall-height", type=float, default=0.45)
    parser.add_argument("--stiffness-scale", type=float, default=1.0)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument(
        "--actuator-mode", choices=("implicit_pd", "explicit_pd"), default="implicit_pd"
    )
    parser.add_argument(
        "--integration-dt",
        type=float,
        default=0.001,
        help=(
            "MuJoCo internal step; implicit_pd is validated at 0.001 s. "
            "explicit_pd requires 0.0001 s for the stiff hand drives."
        ),
    )
    parser.add_argument("--max-policy-steps", type=int, default=500)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help=(
            "Stop automatically after this many wall-clock seconds and write the runtime "
            "report. Zero keeps the interactive Ctrl-C behavior."
        ),
    )
    parser.add_argument(
        "--allow-proxy-reward",
        action="store_true",
        help="Required for T2MIR interface smoke with the non-parity reward proxy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/deployment/ros2_mujoco_runtime.json"),
        help="JSON runtime/provenance report written on shutdown.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_report(
    *,
    backend: Any,
    plant: MujocoRos2Plant,
    runtime_snapshot: dict[str, Any],
    robot_asset_report: Any,
    scene_binding_report: Any,
    scene_path: Path,
    shutdown_reason: str,
) -> dict[str, Any]:
    """Build the immutable report payload without requiring a ROS graph."""

    contract = plant.contract
    resolved_scene = scene_path.expanduser().resolve()
    return {
        "format_version": 1,
        "claim_scope": "minimal-ros2-mujoco-interface-smoke-not-sim2sim-parity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shutdown_reason": shutdown_reason,
        "policy": dict(backend.provenance),
        "contract": contract.as_dict() | {"sha256": contract.sha256},
        "dynamics": asdict(plant.dynamics),
        "reward_schema": plant.reward_provider.schema_id,
        "scene": {
            "path": str(resolved_scene),
            "sha256": _sha256_file(resolved_scene),
        },
        "robot_asset_contract": robot_asset_report.as_dict(),
        "scene_robot_binding": scene_binding_report.as_dict(),
        "bindings": asdict(plant.bindings.metadata),
        "controller_calibration": {
            "stiffness_scale": float(getattr(plant, "stiffness_scale", 1.0)),
            "damping_scale": float(getattr(plant, "damping_scale", 1.0)),
            "actuator_mode": str(getattr(plant, "actuator_mode", "explicit_pd")),
            "mujoco_integrator": int(
                getattr(getattr(getattr(plant, "model", None), "opt", None), "integrator", -1)
            ),
        },
        "runtime": runtime_snapshot,
    }


def write_runtime_report(path: Path, payload: dict[str, Any]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_runtime_seconds < 0.0:
        raise ValueError("--max-runtime-seconds must be non-negative")
    if args.prompt_update_interval <= 0:
        raise ValueError("--prompt-update-interval must be positive")
    _validate_reward_claim(args)
    report = validate_g1_mjcf(args.robot_mjcf)
    scene_binding = validate_scene_robot_binding(args.mjcf, args.robot_mjcf)
    print(
        f"[MJCF-CONTRACT] PASS joints={report.policy_joint_count} "
        f"motors={report.policy_motor_count} sha256={report.source_sha256}"
    )
    backend = _make_backend(args)
    core = Ros2PolicyCore(
        backend,
        config=PolicyLoopConfig(require_fresh_reward=True),
    )
    plant = MujocoRos2Plant.from_xml(
        str(args.mjcf.expanduser().resolve()),
        dynamics=DynamicsParameters(
            action_lag=args.action_lag,
            motor_strength=args.motor_strength,
            payload_kg=args.payload_kg,
            friction=args.friction,
        ),
        episode_config=MujocoEpisodeConfig(
            max_policy_steps=args.max_policy_steps,
            fall_height=args.fall_height,
        ),
        integration_dt=args.integration_dt,
        stiffness_scale=args.stiffness_scale,
        damping_scale=args.damping_scale,
        actuator_mode=args.actuator_mode,
    )
    ros = _load_ros2()
    ros.rclpy.init(args=None)
    node = create_ros2_node(
        core,
        plant.state,
        reset_handler=plant.reset,
        step_handler=make_mujoco_step_handler(core, plant),
    )
    shutdown_reason = "normal"
    try:
        if args.max_runtime_seconds == 0.0:
            ros.rclpy.spin(node)
        else:
            deadline = time.monotonic() + args.max_runtime_seconds
            while ros.rclpy.ok() and time.monotonic() < deadline:
                ros.rclpy.spin_once(
                    node,
                    timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())),
                )
            shutdown_reason = "runtime_limit"
    except KeyboardInterrupt:
        shutdown_reason = "keyboard_interrupt"
    except Exception as exc:
        shutdown_reason = f"error:{type(exc).__name__}"
        raise
    finally:
        runtime_snapshot = node.runtime_snapshot()
        payload = build_runtime_report(
            backend=backend,
            plant=plant,
            runtime_snapshot=runtime_snapshot,
            robot_asset_report=report,
            scene_binding_report=scene_binding,
            scene_path=args.mjcf,
            shutdown_reason=shutdown_reason,
        )
        output = write_runtime_report(args.output, payload)
        print(f"[ROS2-MUJOCO] runtime report: {output}")
        try:
            node.destroy_node()
        finally:
            ros.rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
