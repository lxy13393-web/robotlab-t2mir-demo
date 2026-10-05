"""Command-line entry point for the RobotLab G1 MuJoCo smoke deployment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .assets import validate_g1_mjcf, validate_scene_robot_binding
from .control import DynamicsParameters
from .mujoco_runner import MujocoDeploymentRunner, MujocoEpisodeConfig
from .model_registry import build_backend_from_artifact, load_artifact, sha256_file
from .policy_backends import PPOOnnxBackend, T2MIRTorchBackend
from .video import MujocoVideoRecorder


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPO_ONNX = (
    REPOSITORY_ROOT
    / "logs/rsl_rl/unitree_g1_flat/2026-08-30_23-35-43/exported/policy.onnx"
)
DEFAULT_PILOT3_T2MIR = (
    REPOSITORY_ROOT
    / "third_party/T2MIR/T2MIR-DPT/runs/RobotLab-G1-MultiDynamics"
    / "official_mixed_v1_pilot3_smoke/best.pt"
)
DEFAULT_PPO_ONNX_SHA256 = "00dcf25b81b82adce1d1ab7acf4e617165b34b01c394ad9a9337e9f3cc76e3da"
DEFAULT_PILOT3_T2MIR_SHA256 = "f77898c320946222c2ba2ed9130a71de4a5e6738a840e7d69923cf2611b98d40"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict 123-state/37-action RobotLab policy in MuJoCo. "
            "This is a deployment smoke path, not a sim-to-sim parity claim."
        )
    )
    parser.add_argument("--mjcf", type=Path, required=True, help="37-DoF Unitree G1 scene XML")
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
    parser.add_argument("--device", default="cpu", help="T2MIR torch device; PPO ONNX is CPU")
    parser.add_argument("--prompt-window", choices=("last", "first"), default="last")
    parser.add_argument(
        "--prompt-update-mode",
        choices=("episode", "block"),
        default="episode",
        help="T2MIR prompt protocol; block/64 matches the Isaac block-64 evaluation.",
    )
    parser.add_argument("--prompt-update-interval", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-policy-steps", type=int, default=500)
    parser.add_argument(
        "--command", type=float, nargs=3, metavar=("VX", "VY", "YAW_RATE"), default=(0.4, 0.0, 0.0)
    )
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
    parser.add_argument(
        "--joint-limit-timeconst",
        type=float,
        default=0.002,
        help="MuJoCo joint-stop time constant; 0.002 is the safe minimum at dt=0.001.",
    )
    parser.add_argument(
        "--allow-proxy-reward",
        action="store_true",
        help=(
            "Acknowledge that the built-in MuJoCo reward is an observable proxy, "
            "not the training-parity Isaac reward. Required for T2MIR interface smoke."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/deployment/mujoco_smoke.json"))
    parser.add_argument("--video", action="store_true", help="Record an off-screen MP4.")
    parser.add_argument(
        "--video-output",
        type=Path,
        default=Path("outputs/deployment/videos/mujoco_demo.mp4"),
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    return parser


def _make_backend(args: argparse.Namespace):
    _validate_policy_source(args)
    if args.model_manifest is not None:
        artifact, reference = _resolve_model_manifest(args)
        backend = build_backend_from_artifact(
            artifact,
            device=args.device,
            prompt_window=args.prompt_window,
            prompt_update_mode=args.prompt_update_mode,
            prompt_update_interval=args.prompt_update_interval,
        )
        backend.provenance["model_artifact"]["reference"] = reference
        return backend
    backend = args.backend or "ppo"
    if backend == "ppo":
        checkpoint = args.checkpoint or DEFAULT_PPO_ONNX
        expected = args.expected_checkpoint_sha256
        if args.checkpoint is None:
            expected = expected or DEFAULT_PPO_ONNX_SHA256
        elif expected is None:
            raise ValueError("an explicit --checkpoint requires --expected-checkpoint-sha256")
        return PPOOnnxBackend(checkpoint, expected_sha256=expected)
    checkpoint = args.checkpoint or DEFAULT_PILOT3_T2MIR
    expected = args.expected_checkpoint_sha256
    if args.checkpoint is None:
        expected = expected or DEFAULT_PILOT3_T2MIR_SHA256
    elif expected is None:
        raise ValueError("an explicit --checkpoint requires --expected-checkpoint-sha256")
    return T2MIRTorchBackend(
        checkpoint,
        device=args.device,
        window=args.prompt_window,
        update_mode=args.prompt_update_mode,
        update_interval=args.prompt_update_interval,
        expected_sha256=expected,
    )


def _validate_reward_claim(args: argparse.Namespace) -> None:
    """Prevent a context policy from silently consuming a non-parity reward."""

    if _selected_backend(args) == "t2mir" and not args.allow_proxy_reward:
        raise ValueError(
            "T2MIR consumes reward in its online context, but this minimum MuJoCo "
            "path currently provides only an observable proxy. Pass "
            "--allow-proxy-reward for an interface-only smoke, or supply a "
            "training-parity RewardProvider in code for a formal evaluation."
        )


def _validate_policy_source(args: argparse.Namespace) -> None:
    if args.model_manifest is not None and any(
        value is not None
        for value in (args.backend, args.checkpoint, args.expected_checkpoint_sha256)
    ):
        raise ValueError(
            "--model-manifest is mutually exclusive with --backend, --checkpoint "
            "and --expected-checkpoint-sha256"
        )


def _selected_backend(args: argparse.Namespace) -> str:
    _validate_policy_source(args)
    if args.model_manifest is not None:
        return str(_resolve_model_manifest(args)[0]["backend"])
    return args.backend or "ppo"


def _resolve_model_manifest(args: argparse.Namespace) -> tuple[dict, dict]:
    """Freeze one artifact/reference identity for this process invocation."""

    path = args.model_manifest.expanduser().resolve()
    digest = sha256_file(path)
    cached = getattr(args, "_resolved_model_artifact", None)
    if cached is None:
        artifact = load_artifact(path)
        reference = {"path": str(path), "sha256": digest}
        args._resolved_model_artifact = (artifact, reference)
        return artifact, reference
    artifact, reference = cached
    if digest != reference["sha256"]:
        raise RuntimeError("model manifest/active pointer changed during process startup")
    return artifact, reference


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.prompt_update_interval <= 0:
        raise ValueError("--prompt-update-interval must be positive")
    if args.video and (
        args.video_fps <= 0.0
        or args.video_fps > 50.0
        or args.video_width <= 0
        or args.video_height <= 0
        or args.video_width % 2
        or args.video_height % 2
    ):
        raise ValueError(
            "video fps must be in (0, 50] and video dimensions must be positive even integers"
        )
    _validate_reward_claim(args)
    asset_report = validate_g1_mjcf(args.robot_mjcf)
    scene_binding = validate_scene_robot_binding(args.mjcf, args.robot_mjcf)
    print(
        f"[MJCF-CONTRACT] PASS joints={asset_report.policy_joint_count} "
        f"motors={asset_report.policy_motor_count} sha256={asset_report.source_sha256}"
    )
    policy = _make_backend(args)
    if args.video:
        # Must be selected before MuJoCo's OpenGL backend is imported.
        os.environ.setdefault("MUJOCO_GL", "egl")
    runner = MujocoDeploymentRunner.from_xml(
        args.mjcf,
        policy,
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
        joint_limit_time_constant=args.joint_limit_timeconst,
    )
    runner.robot_asset_report = asset_report.as_dict()
    runner.scene_binding_report = scene_binding.as_dict()
    recorder = None
    try:
        if args.video:
            recorder = MujocoVideoRecorder(
                runner.mj,
                runner.model,
                runner.data,
                base_body_id=runner.bindings.metadata.base_body_id,
                output=args.video_output,
                fps=args.video_fps,
                source_fps=runner.contract.policy_hz,
                width=args.video_width,
                height=args.video_height,
            )
        results = [
            runner.run_episode(
                args.command,
                episode_index=index,
                frame_callback=None if recorder is None else recorder.capture_policy_step,
            )
            for index in range(args.episodes)
        ]
    finally:
        if recorder is not None:
            video_path = recorder.close()
            print(f"[MUJOCO-VIDEO] frames={recorder.frame_count} output={video_path}")
    output = runner.write_report(args.output, results)
    print(json.dumps({"output": str(output), "episodes": [asdict(item) for item in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
