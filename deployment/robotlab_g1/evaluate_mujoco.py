"""Paired, goal-level MuJoCo evaluation for RobotLab G1 policies.

This evaluator mirrors the official Isaac online protocol at the experiment
boundary: held-out dynamics only, episode-zero empty context, paired goals,
and success/fall/timeout as primary metrics. Both the source-faithful frozen
previous-episode prompt and the causal block-update protocol are selectable.
It deliberately refuses to call the current observable MuJoCo reward
training-parity; T2MIR proxy-reward runs require an explicit acknowledgement
and are labelled as preliminary in the report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.protocols.dynamics_profile import DynamicsProfile, load_manifest
from pipeline.protocols.goal_controller import GoalController, GoalControllerCfg
from pipeline.protocols.paired_goal_protocol import (
    load_manifest as load_paired_goal_manifest,
)

from .assets import validate_g1_mjcf, validate_scene_robot_binding
from .control import DynamicsParameters
from .model_registry import build_backend_from_artifact, load_artifact, sha256_file
from .mujoco_runner import MujocoDeploymentRunner, MujocoEpisodeConfig
from .rewards import RewardInput


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DYNAMICS_MANIFEST = REPOSITORY_ROOT / "configs/g1_dynamics_48.json"
CONTROLLER_MODE_TO_ID = {
    GoalController.TURN_TO_GOAL: 0,
    GoalController.WALK_TO_GOAL: 1,
    GoalController.ALIGN_FINAL_YAW: 2,
    GoalController.HOLD: 3,
}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty evaluation")
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    return output


def _yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _prompt_length(value: Any) -> int:
    if not value:
        return 0
    first = value[0]
    states = getattr(first, "states", None)
    if states is None:
        return 0
    shape = tuple(states.shape)
    return int(shape[-2]) if len(shape) >= 2 else 0


@dataclass(frozen=True)
class GoalScenario:
    profile_id: int
    episode_index: int
    relative_x: float
    relative_y: float
    relative_yaw: float
    replica_id: int = 0


@dataclass(frozen=True)
class GoalEvaluationConfig:
    timeout_seconds: float = 40.0
    hold_seconds: float = 2.0
    fall_height: float = 0.45
    turn_forward_speed: float = 0.12
    align_forward_speed: float = 0.0
    hold_command_mode: str = "safe"
    reset_context_each_goal: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0.0 or self.hold_seconds <= 0.0:
            raise ValueError("timeout and hold time must be positive")
        if self.hold_seconds >= self.timeout_seconds:
            raise ValueError("hold time must be shorter than timeout")
        if not 0.0 <= self.turn_forward_speed <= 0.5:
            raise ValueError("turn forward speed must be in [0, 0.5]")
        if not 0.0 <= self.align_forward_speed <= 0.5:
            raise ValueError("align forward speed must be in [0, 0.5]")
        if self.hold_command_mode not in {"zero", "safe", "corrective"}:
            raise ValueError("hold command mode must be zero, safe or corrective")


def generate_scenarios(
    profiles: Sequence[DynamicsProfile],
    *,
    episodes: int,
    seed: int,
    min_distance: float,
    max_distance: float,
) -> list[GoalScenario]:
    if episodes < 2:
        raise ValueError("at least two episodes are required to measure adaptation")
    if not 0.0 < min_distance < max_distance:
        raise ValueError("goal distances must satisfy 0 < min < max")
    rows: list[GoalScenario] = []
    for profile in profiles:
        # Profile-local deterministic streams make a profile's scenarios
        # invariant to profile ordering and to full-suite versus subset runs.
        generator = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(profile.task_id)])
        )
        distances = generator.uniform(min_distance, max_distance, size=episodes)
        bearings = generator.uniform(-math.pi, math.pi, size=episodes)
        yaws = generator.uniform(-math.pi, math.pi, size=episodes)
        for episode in range(episodes):
            rows.append(
                GoalScenario(
                    profile_id=profile.task_id,
                    episode_index=episode,
                    replica_id=0,
                    relative_x=float(distances[episode] * math.cos(bearings[episode])),
                    relative_y=float(distances[episode] * math.sin(bearings[episode])),
                    relative_yaw=float(yaws[episode]),
                )
            )
    return rows


def scenario_sha256(scenarios: Sequence[GoalScenario]) -> str:
    canonical = json.dumps(
        [asdict(row) for row in scenarios], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_reward_claim(artifact: Mapping[str, Any], allow_proxy_reward: bool) -> str:
    """Return the report claim scope or reject an invalid formal claim."""

    if artifact["backend"] == "t2mir":
        if not allow_proxy_reward:
            raise ValueError(
                "formal T2MIR evaluation requires a training-parity MuJoCo reward. "
                "The current reward is an observable proxy; pass --allow-proxy-reward "
                "only for a preliminary protocol/closed-loop run."
            )
        return "preliminary-paired-heldout-goal-evaluation-proxy-reward"
    return "paired-heldout-goal-evaluation-ppo-reward-independent"


class MujocoGoalEvaluator:
    """Execute goal episodes while preserving the official online prompt boundary."""

    def __init__(self, runner: MujocoDeploymentRunner, config: GoalEvaluationConfig) -> None:
        self.runner = runner
        self.config = config

    def run(self, scenario: GoalScenario) -> dict[str, Any]:
        runner = self.runner
        runner.bindings.reset_to_policy_pose(runner.data)
        runner.bindings.apply_dynamics(runner.data, runner.dynamics)
        runner.action_filter.reset()
        runner.previous_executed_action.fill(0.0)

        base_id = runner.bindings.metadata.base_body_id
        initial_position = np.asarray(runner.data.xpos[base_id], dtype=np.float64).copy()
        initial_yaw = _yaw_from_wxyz(runner.data.xquat[base_id])
        cos_yaw, sin_yaw = math.cos(initial_yaw), math.sin(initial_yaw)
        target_x = initial_position[0] + cos_yaw * scenario.relative_x - sin_yaw * scenario.relative_y
        target_y = initial_position[1] + sin_yaw * scenario.relative_x + cos_yaw * scenario.relative_y
        controller_cfg = GoalControllerCfg(
            turn_forward_speed=self.config.turn_forward_speed,
            align_forward_speed=self.config.align_forward_speed,
        )
        if self.config.hold_command_mode == "zero":
            controller_cfg.hold_position_gain = 0.0
            controller_cfg.hold_yaw_gain = 0.0
        elif self.config.hold_command_mode == "safe":
            controller_cfg.hold_allow_backward = False
        controller = GoalController(
            target_x,
            target_y,
            initial_yaw + scenario.relative_yaw,
            controller_cfg,
        )

        if self.config.reset_context_each_goal:
            runner.policy.reset_context()
        prompts = runner.policy.start_episode(scenario.episode_index)
        prompt_selected_length = _prompt_length(prompts)
        completed_steps = 0
        episode_return = 0.0
        hold_steps = 0
        result = "timeout"
        reason = "goal_timeout"
        max_steps = max(1, round(self.config.timeout_seconds / runner.contract.policy_dt))
        required_hold_steps = max(1, math.ceil(self.config.hold_seconds / runner.contract.policy_dt))
        distance = float("inf")
        yaw_error = float("inf")
        final_state = controller.state
        mode_counts = {str(index): 0 for index in range(4)}
        try:
            for step in range(max_steps):
                position = np.asarray(runner.data.xpos[base_id], dtype=np.float64)
                yaw = _yaw_from_wxyz(runner.data.xquat[base_id])
                command, info = controller.compute(float(position[0]), float(position[1]), yaw)
                command_array = np.asarray(command, dtype=np.float32)
                final_state = controller.state
                mode_id = CONTROLLER_MODE_TO_ID[controller.state]
                mode_counts[str(mode_id)] += 1

                pre_state, _ = runner._robot_state(command_array)
                observation = runner.observation_builder.build(pre_state)
                raw_action = np.asarray(runner.policy.predict(observation), dtype=np.float32)
                executed_action = runner.action_filter.apply(raw_action)
                previous_executed = runner.previous_executed_action.copy()
                last_torque = runner.step_physics(executed_action)

                post_state, projected_gravity = runner._robot_state(command_array)
                terminated, termination_reason = runner._terminated(projected_gravity)
                reward = runner.reward_provider.compute(
                    RewardInput(
                        base_linear_velocity=np.asarray(post_state.base_linear_velocity),
                        base_angular_velocity=np.asarray(post_state.base_angular_velocity),
                        projected_gravity=projected_gravity,
                        command=command_array,
                        action=executed_action,
                        previous_action=previous_executed,
                        joint_torque=last_torque,
                        terminated=terminated,
                    )
                )
                runner.policy.record_transition(
                    observation, raw_action, reward, controller_modes=mode_id
                )
                runner.previous_executed_action = executed_action
                episode_return += reward
                completed_steps = step + 1

                position = np.asarray(runner.data.xpos[base_id], dtype=np.float64)
                yaw = _yaw_from_wxyz(runner.data.xquat[base_id])
                distance, _, yaw_error = controller.errors(
                    float(position[0]), float(position[1]), yaw
                )
                if terminated:
                    result, reason = "fall", termination_reason
                    break
                within = (
                    controller.state == GoalController.HOLD
                    and distance <= controller.cfg.position_tolerance
                    and abs(yaw_error) <= controller.cfg.yaw_tolerance
                )
                hold_steps = hold_steps + 1 if within else 0
                if hold_steps >= required_hold_steps:
                    result, reason = "success", "held_goal"
                    break
        finally:
            trajectories = runner.policy.finish_episode()

        recorded_length = 0
        if trajectories:
            recorded_length = int(getattr(trajectories[0], "length", completed_steps))
        elif completed_steps:
            recorded_length = completed_steps
        return {
            **asdict(scenario),
            "result": result,
            "success": int(result == "success"),
            "fall": int(result == "fall"),
            "timeout": int(result == "timeout"),
            "termination_reason": reason,
            "policy_steps": completed_steps,
            "time_seconds": completed_steps * runner.contract.policy_dt,
            "episode_return": float(episode_return),
            "position_error": float(distance),
            "yaw_error": abs(float(yaw_error)),
            "final_controller_state": final_state,
            "controller_mode_counts": json.dumps(mode_counts, sort_keys=True),
            "prompt_selected_length": prompt_selected_length,
            "recorded_episode_length": recorded_length,
        }


def summarize(rows: Sequence[Mapping[str, Any]], profile_ids: Sequence[int], episodes: int) -> dict[str, Any]:
    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = max(len(selected), 1)
        return {
            "episodes": len(selected),
            "success_rate": sum(int(row["success"]) for row in selected) / count,
            "fall_rate": sum(int(row["fall"]) for row in selected) / count,
            "timeout_rate": sum(int(row["timeout"]) for row in selected) / count,
            "mean_return": sum(float(row["episode_return"]) for row in selected) / count,
            "mean_position_error": sum(float(row["position_error"]) for row in selected) / count,
            "mean_yaw_error": sum(float(row["yaw_error"]) for row in selected) / count,
        }

    per_profile = {
        str(profile_id): aggregate([row for row in rows if int(row["profile_id"]) == profile_id])
        for profile_id in profile_ids
    }
    per_episode = [
        {"episode_index": index}
        | aggregate([row for row in rows if int(row["episode_index"]) == index])
        for index in range(episodes)
    ]
    overall = aggregate(rows)
    overall["adaptation_success_delta_last_minus_first"] = (
        per_episode[-1]["success_rate"] - per_episode[0]["success_rate"]
    )
    overall["adaptation_return_delta_last_minus_first"] = (
        per_episode[-1]["mean_return"] - per_episode[0]["mean_return"]
    )
    return {"overall": overall, "per_profile": per_profile, "per_episode": per_episode}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--robot-mjcf", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--dynamics-manifest", type=Path, default=DEFAULT_DYNAMICS_MANIFEST)
    parser.add_argument("--profile-ids", type=int, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--goal-min-distance", type=float, default=1.0)
    parser.add_argument("--goal-max-distance", type=float, default=4.0)
    parser.add_argument("--goal-timeout", type=float, default=40.0)
    parser.add_argument("--goal-hold-time", type=float, default=2.0)
    parser.add_argument(
        "--goal-scenarios",
        type=Path,
        default=None,
        help="Shared paired-goal JSON; overrides independent random goal generation.",
    )
    parser.add_argument("--fall-height", type=float, default=0.45)
    parser.add_argument("--turn-forward-speed", type=float, default=0.12)
    parser.add_argument("--align-forward-speed", type=float, default=0.0)
    parser.add_argument(
        "--hold-command-mode", choices=("zero", "safe", "corrective"), default="safe"
    )
    parser.add_argument(
        "--reset-context-each-goal",
        action="store_true",
        help=(
            "Clear the previous goal's prompt at each high-level goal boundary while "
            "retaining causal block updates within the current goal."
        ),
    )
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
    parser.add_argument("--joint-limit-timeconst", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-window", choices=("last", "first"), default="last")
    parser.add_argument(
        "--prompt-update-mode", choices=("episode", "block"), default="episode"
    )
    parser.add_argument("--prompt-update-interval", type=int, default=64)
    parser.add_argument("--allow-proxy-reward", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt_update_interval <= 0:
        raise ValueError("--prompt-update-interval must be positive")
    model_manifest_path = args.model_manifest.expanduser().resolve()
    model_reference_sha256 = sha256_file(model_manifest_path)
    artifact = load_artifact(model_manifest_path)
    claim_scope = validate_reward_claim(artifact, args.allow_proxy_reward)
    manifest, all_profiles = load_manifest(args.dynamics_manifest.resolve())
    eval_profiles = [profile for profile in all_profiles if profile.split == "eval"]
    requested = args.profile_ids if args.profile_ids is not None else manifest["eval_tasks"]
    requested = [int(value) for value in requested]
    if not requested:
        raise ValueError("at least one held-out profile is required")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate profile IDs are not allowed: {requested}")
    profile_by_id = {profile.task_id: profile for profile in eval_profiles}
    unknown = sorted(set(requested) - set(profile_by_id))
    if unknown:
        raise ValueError(f"profiles are not held-out eval profiles: {unknown}")
    profiles = [profile_by_id[task_id] for task_id in requested]
    paired_goal_manifest = None
    paired_goal_path = None
    if args.goal_scenarios is not None:
        paired_goal_path = args.goal_scenarios.expanduser().resolve()
        paired_goal_manifest, paired_rows = load_paired_goal_manifest(paired_goal_path)
        if paired_goal_manifest["profile_ids"] != requested:
            raise ValueError(
                f"paired goals profiles={paired_goal_manifest['profile_ids']} do not match "
                f"--profile-ids {requested}"
            )
        if paired_goal_manifest["episodes"] != args.episodes:
            raise ValueError("paired goals episodes do not match --episodes")
        if not math.isclose(paired_goal_manifest["min_distance"], args.goal_min_distance):
            raise ValueError("paired goals min_distance does not match CLI")
        if not math.isclose(paired_goal_manifest["max_distance"], args.goal_max_distance):
            raise ValueError("paired goals max_distance does not match CLI")
        scenarios = [GoalScenario(**asdict(row)) for row in paired_rows]
        scenario_hash = paired_goal_manifest["scenario_sha256"]
    else:
        scenarios = generate_scenarios(
            profiles,
            episodes=args.episodes,
            seed=args.seed,
            min_distance=args.goal_min_distance,
            max_distance=args.goal_max_distance,
        )
        scenario_hash = scenario_sha256(scenarios)
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / "episodes.csv"
    report_path = output_dir / "report.json"
    if not args.force and (csv_path.exists() or report_path.exists()):
        raise FileExistsError(f"refusing to overwrite {output_dir}; pass --force to rerun")

    asset_report = validate_g1_mjcf(args.robot_mjcf)
    scene_binding = validate_scene_robot_binding(args.mjcf, args.robot_mjcf)
    config = GoalEvaluationConfig(
        args.goal_timeout,
        args.goal_hold_time,
        args.fall_height,
        args.turn_forward_speed,
        args.align_forward_speed,
        args.hold_command_mode,
        args.reset_context_each_goal,
    )
    rows: list[dict[str, Any]] = []
    runner_provenance: dict[str, Any] = {}
    for profile in profiles:
        # A fresh backend prevents held-out profile context from leaking into
        # the next profile. Within a profile the CLI contract decides whether
        # context crosses high-level goal boundaries.
        policy = build_backend_from_artifact(
            artifact,
            device=args.device,
            prompt_window=args.prompt_window,
            prompt_update_mode=args.prompt_update_mode,
            prompt_update_interval=args.prompt_update_interval,
        )
        runner = MujocoDeploymentRunner.from_xml(
            args.mjcf,
            policy,
            dynamics=DynamicsParameters(
                action_lag=profile.action_lag,
                motor_strength=profile.motor_strength,
                payload_kg=profile.payload_kg,
                friction=profile.friction,
            ),
            episode_config=MujocoEpisodeConfig(
                max_policy_steps=max(1, round(args.goal_timeout / 0.02)),
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
        evaluator = MujocoGoalEvaluator(runner, config)
        profile_rows = [row for row in scenarios if row.profile_id == profile.task_id]
        for scenario in profile_rows:
            result = evaluator.run(scenario)
            result.update(
                {
                    "action_lag": profile.action_lag,
                    "motor_strength": profile.motor_strength,
                    "payload_kg": profile.payload_kg,
                    "friction": profile.friction,
                }
            )
            rows.append(result)
            print(
                f"[MUJOCO-EVAL] profile={profile.task_id} episode={scenario.episode_index} "
                f"result={result['result']} pos={result['position_error']:.3f} "
                f"yaw={result['yaw_error']:.3f}",
                flush=True,
            )
        runner_provenance[str(profile.task_id)] = runner.provenance()

    if sha256_file(model_manifest_path) != model_reference_sha256:
        raise RuntimeError(
            "model manifest/active pointer changed during evaluation; refusing mixed provenance"
        )
    output_csv = _write_csv_atomic(csv_path, rows)
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("model_registry.py").resolve(),
        REPOSITORY_ROOT / "pipeline/protocols/paired_goal_protocol.py",
        REPOSITORY_ROOT / "pipeline/protocols/goal_controller.py",
    ]
    report = {
        "format_version": 1,
        "evaluation": "robotlab_g1_mujoco_stationary_online",
        "claim_scope": claim_scope,
        "model_artifact": artifact,
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": model_reference_sha256,
        "dynamics_manifest": str(args.dynamics_manifest.resolve()),
        "dynamics_manifest_sha256": manifest["sha256"],
        "held_out_profile_ids": requested,
        "seed": args.seed,
        "scenario_sha256": scenario_hash,
        "paired_goal_manifest": str(paired_goal_path) if paired_goal_path else None,
        "paired_goal_manifest_sha256": (
            sha256_file(paired_goal_path) if paired_goal_path else None
        ),
        "goal_contract": {
            "episodes_per_profile": args.episodes,
            "replicas": (
                paired_goal_manifest["replicas"] if paired_goal_manifest else 1
            ),
            "total_goals_per_profile": (
                paired_goal_manifest["goals_per_profile"]
                if paired_goal_manifest
                else args.episodes
            ),
            "min_distance": args.goal_min_distance,
            "max_distance": args.goal_max_distance,
            "timeout_seconds": args.goal_timeout,
            "hold_seconds": args.goal_hold_time,
            "turn_forward_speed": args.turn_forward_speed,
            "align_forward_speed": args.align_forward_speed,
            "hold_command_mode": args.hold_command_mode,
            "reset_context_each_goal": args.reset_context_each_goal,
        },
        "prompt_contract": {
            "episode_zero": "empty",
            "source": "own causal rollout",
            "window": args.prompt_window,
            "horizon": 64,
            "update_mode": args.prompt_update_mode,
            "update_interval": args.prompt_update_interval,
            "goal_boundary_policy": (
                "clear_previous_prompt" if args.reset_context_each_goal else "retain_previous_prompt"
            ),
        },
        "reward_warning": (
            "observable proxy; not Isaac training-parity"
            if artifact["backend"] == "t2mir"
            else "PPO does not consume reward; return remains secondary"
        ),
        "primary_metrics": ["success_rate", "fall_rate", "timeout_rate", "per_episode_adaptation"],
        "secondary_metrics": ["episode_return", "position_error", "yaw_error"],
        "episodes_csv": str(output_csv),
        "episodes_csv_sha256": sha256_file(output_csv),
        "summary": summarize(rows, requested, args.episodes),
        "runner_provenance_by_profile": runner_provenance,
        "code_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in code_paths
        },
        "command": sys.argv if argv is None else ["evaluate_mujoco", *argv],
    }
    output_report = _write_json_atomic(report_path, report)
    print(f"[MUJOCO-EVAL] COMPLETE report={output_report}", flush=True)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
