"""Source-faithful stationary held-out T2MIR evaluation in Isaac Sim.

This is deliberately separate from :mod:`play`.  The formal collector freezes
``play.py`` as a data-provenance artifact, while this entry point implements the
online DPT protocol used for the final closed-loop experiment:

* a held-out dynamics profile stays fixed for all evaluation episodes;
* episode 0 has an empty prompt;
* the source-faithful default keeps the prompt fixed during an episode;
* after reset, each vector environment receives only its own previous episode;
* at most the most recent 64 real transitions are retained; and
* success/fall/return are primary outputs (offline action MSE is not computed).

An opt-in block mode maps RobotLab's long goal episode to causal 64-step
context blocks.  It is reported as a separate protocol and never replaces the
default result implicitly.

The script never accepts a dataset directory or task ID prompt.  Consequently
there is no code path through which held-out prompt/query pickle files can be
loaded during the official online evaluation.
"""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import sys
from collections import Counter
from pathlib import Path

from omni.isaac.lab.app import AppLauncher


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs/g1_dynamics_48.json"
DEFAULT_OUTPUT = REPO / "outputs/t2mir_online_stationary"
VARIANT_MODES = {
    "A": ("topk", "topk"),
    "B": ("topp", "topk"),
    "C": ("topk", "topp"),
    "D": ("topp", "topp"),
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0",
    help="RobotLab environment ID.",
)
parser.add_argument("--offline-checkpoint", type=Path, default=None)
parser.add_argument("--checkpoint-sha256", default=None)
parser.add_argument("--variant", choices=tuple(VARIANT_MODES), default=None)
parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
parser.add_argument("--profile-id", type=int, default=None)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
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
parser.add_argument("--turn-forward-speed", type=float, default=0.0)
parser.add_argument("--align-forward-speed", type=float, default=0.0)
parser.add_argument(
    "--hold-command-mode", choices=("zero", "safe", "corrective"), default="corrective"
)
parser.add_argument(
    "--reset-context-each-goal",
    action="store_true",
    help="Clear cross-goal prompt state while retaining causal block updates within a goal.",
)
parser.add_argument(
    "--prompt-window",
    choices=("last", "first"),
    default="last",
    help="Select the first or last 64 real transitions from an eligible trajectory.",
)
parser.add_argument(
    "--prompt-update-mode",
    choices=("episode", "block"),
    default="episode",
    help=(
        "episode keeps the official fixed previous-episode prompt; block promotes "
        "a causal current-rollout context at fixed step boundaries"
    ),
)
parser.add_argument(
    "--prompt-update-interval",
    type=int,
    default=64,
    help="Completed control steps between causal prompt promotions in block mode.",
)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--force", action="store_true")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.offline_checkpoint is None or args_cli.variant is None or args_cli.profile_id is None:
    parser.error("--offline-checkpoint, --variant and --profile-id are required")
if args_cli.episodes < 2:
    parser.error("--episodes must be at least 2 to measure online adaptation")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if not 0.0 < args_cli.goal_min_distance < args_cli.goal_max_distance:
    parser.error("goal distances must satisfy 0 < min < max")
if args_cli.goal_timeout <= 0.0 or args_cli.goal_hold_time <= 0.0:
    parser.error("goal timeout and hold time must be positive")
if args_cli.prompt_update_interval <= 0:
    parser.error("--prompt-update-interval must be positive")
if not 0.0 <= args_cli.turn_forward_speed <= 0.5:
    parser.error("--turn-forward-speed must be in [0, 0.5]")
if not 0.0 <= args_cli.align_forward_speed <= 0.5:
    parser.error("--align-forward-speed must be in [0, 0.5]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _raise_runtime_signal(signum, frame):
    """Keep Isaac Sim signal handlers from turning an interrupted run into exit code 0.

    ``SimulationApp`` installs a handler that calls ``sys.exit(0)`` on Ctrl-C.
    That makes a truncated evaluation look successful to a parent queue even
    though no CSV/JSON report was written.  Restoring normal Python interrupt
    semantics lets the queue distinguish an interruption from completion.
    """

    signal_name = signal.Signals(signum).name
    print(
        f"[T2MIR-ONLINE] INTERRUPTED: received {signal_name}; "
        "no complete result was produced",
        flush=True,
    )
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise RuntimeError(f"online evaluation interrupted by {signal_name}")


signal.signal(signal.SIGINT, _raise_runtime_signal)
signal.signal(signal.SIGTERM, _raise_runtime_signal)
signal.signal(signal.SIGABRT, _raise_runtime_signal)


"""Isaac imports must happen after SimulationApp starts."""

import gymnasium as gym
import torch

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.math import euler_xyz_from_quat
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

import robot_lab.tasks  # noqa: F401,E402

from pipeline.protocols.dynamics_profile import configure_env_dynamics, load_manifest
from pipeline.protocols.goal_controller import GoalController, GoalControllerCfg
from pipeline.protocols.paired_goal_protocol import load_manifest as load_paired_goal_manifest
from pipeline.protocols.ppo_online_protocol import (
    EPISODE_BOUNDARY_MECHANISM,
    freeze_evaluation_reset,
    pose_sha256,
    synchronize_vector_episode_boundary,
)
from pipeline.protocols.t2mir_online_context import T2MIROnlinePolicy, sha256_file


CONTROLLER_MODE_TO_ID = {
    GoalController.TURN_TO_GOAL: 0,
    GoalController.WALK_TO_GOAL: 1,
    GoalController.ALIGN_FINAL_YAW: 2,
    GoalController.HOLD: 3,
}


class RobotLabRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Compatibility wrapper for the bundled RSL-RL interface."""

    def get_privileged_observations(self) -> torch.Tensor | None:
        _, extras = self.get_observations()
        return extras["observations"].get("critic")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def profile_from_manifest(path: Path, profile_id: int):
    manifest, profiles = load_manifest(path)
    matches = [profile for profile in profiles if profile.task_id == profile_id]
    if len(matches) != 1:
        raise ValueError(f"profile {profile_id} is absent or duplicated in {path}")
    profile = matches[0]
    if profile.split != "eval":
        raise ValueError(
            f"profile {profile_id} split={profile.split!r}; official online evaluation "
            "requires split='eval'"
        )
    return manifest, profile


def prompt_metadata(groups) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for group in groups:
        selected_length = int(group.states.shape[1])
        for offset, env_id in enumerate(group.env_ids):
            result[int(env_id)] = {
                "selected_length": selected_length,
                "source_episode_length": int(group.source_lengths[offset]),
                "source_episode_index": group.source_episode_indices[offset],
                "controller_mode_histogram": {
                    str(key): int(value)
                    for key, value in group.controller_mode_histograms[offset].items()
                },
            }
    return result


def summarize(rows: list[dict], episodes: int) -> dict:
    per_episode = []
    for episode_index in range(episodes):
        selected = [row for row in rows if row["episode_index"] == episode_index]
        successes = sum(row["result"] == "success" for row in selected)
        falls = sum(row["result"] == "fall" for row in selected)
        per_episode.append(
            {
                "episode_index": episode_index,
                "episodes": len(selected),
                "successes": successes,
                "falls": falls,
                "timeouts": sum(row["result"] == "timeout" for row in selected),
                "success_rate": successes / max(len(selected), 1),
                "fall_rate": falls / max(len(selected), 1),
                "mean_return": sum(float(row["episode_return"]) for row in selected)
                / max(len(selected), 1),
                "mean_position_error": sum(float(row["position_error"]) for row in selected)
                / max(len(selected), 1),
                "mean_yaw_error": sum(float(row["yaw_error"]) for row in selected)
                / max(len(selected), 1),
            }
        )
    total_successes = sum(row["result"] == "success" for row in rows)
    total_falls = sum(row["result"] == "fall" for row in rows)
    return {
        "episodes_per_replica": episodes,
        "replicas": len(rows) // episodes,
        "total_episodes": len(rows),
        "success_rate": total_successes / max(len(rows), 1),
        "fall_rate": total_falls / max(len(rows), 1),
        "per_episode": per_episode,
        "adaptation_success_delta_last_minus_first": (
            per_episode[-1]["success_rate"] - per_episode[0]["success_rate"]
        ),
        "adaptation_return_delta_last_minus_first": (
            per_episode[-1]["mean_return"] - per_episode[0]["mean_return"]
        ),
    }


def main() -> None:
    manifest_path = args_cli.manifest.resolve()
    checkpoint_path = args_cli.offline_checkpoint.resolve()
    output_dir = args_cli.output_dir.resolve()
    manifest, profile = profile_from_manifest(manifest_path, args_cli.profile_id)

    stem = f"variant{args_cli.variant}_task{profile.task_id:02d}_seed{args_cli.seed}"
    csv_path = output_dir / f"{stem}.csv"
    report_path = output_dir / f"{stem}.json"
    if not args_cli.force and (csv_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"refusing to reuse existing result {stem}; pass --force only for an intentional rerun"
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configured = configure_env_dynamics(
        env_cfg,
        motor_strength=profile.motor_strength,
        payload_kg=profile.payload_kg,
        friction=profile.friction,
        deterministic=True,
    )
    configured["action_lag"] = profile.action_lag
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = args_cli.goal_timeout + 10.0
    reset_contract = freeze_evaluation_reset(env_cfg)

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = RobotLabRslRlVecEnvWrapper(raw_env)
    device = torch.device(env.unwrapped.device)
    policy = T2MIROnlinePolicy(
        checkpoint_path,
        device,
        REPO / "methods/t2mir",
        num_envs=args_cli.num_envs,
        prompt_horizon=64,
        window=args_cli.prompt_window,
        update_mode=args_cli.prompt_update_mode,
        update_interval=args_cli.prompt_update_interval,
        expected_sha256=args_cli.checkpoint_sha256,
        expected_supervision_modes=("all_tokens", "final_token"),
    )
    if (policy.provenance["state_dim"], policy.provenance["action_dim"]) != (
        int(env.num_obs),
        int(env.num_actions),
    ):
        raise ValueError(
            "checkpoint/environment dimension mismatch: "
            f"checkpoint={(policy.provenance['state_dim'], policy.provenance['action_dim'])} "
            f"environment={(env.num_obs, env.num_actions)}"
        )
    signature = policy.provenance["routing_signature"]
    actual_modes = (signature["token"]["mode"], signature["task"]["mode"])
    if actual_modes != VARIANT_MODES[args_cli.variant]:
        raise ValueError(
            f"checkpoint modes={actual_modes} do not match variant {args_cli.variant}="
            f"{VARIANT_MODES[args_cli.variant]}"
        )

    paired_goal_manifest = None
    paired_goal_path = None
    if args_cli.goal_scenarios is not None:
        paired_goal_path = args_cli.goal_scenarios.expanduser().resolve()
        paired_goal_manifest, paired_rows = load_paired_goal_manifest(paired_goal_path)
        if paired_goal_manifest["profile_ids"] != [profile.task_id]:
            raise ValueError(
                f"paired goals profiles={paired_goal_manifest['profile_ids']} do not match "
                f"--profile-id {profile.task_id}"
            )
        if paired_goal_manifest["episodes"] != args_cli.episodes:
            raise ValueError("paired goals episodes do not match --episodes")
        if paired_goal_manifest["replicas"] != args_cli.num_envs:
            raise ValueError("paired goals replicas do not match --num_envs")
        if not math.isclose(paired_goal_manifest["min_distance"], args_cli.goal_min_distance):
            raise ValueError("paired goals min_distance does not match CLI")
        if not math.isclose(paired_goal_manifest["max_distance"], args_cli.goal_max_distance):
            raise ValueError("paired goals max_distance does not match CLI")
        scenario_tensor = torch.empty(
            (args_cli.episodes, args_cli.num_envs, 3), dtype=torch.float32
        )
        for row in paired_rows:
            scenario_tensor[row.episode_index, row.replica_id] = torch.tensor(
                (row.relative_x, row.relative_y, row.relative_yaw), dtype=torch.float32
            )
        scenario_sha = paired_goal_manifest["scenario_sha256"]
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args_cli.seed)
        shape = (args_cli.episodes, args_cli.num_envs)
        distances = args_cli.goal_min_distance + (
            args_cli.goal_max_distance - args_cli.goal_min_distance
        ) * torch.rand(shape, generator=generator)
        bearings = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
        relative_yaws = -math.pi + 2.0 * math.pi * torch.rand(shape, generator=generator)
        relative_x = distances * torch.cos(bearings)
        relative_y = distances * torch.sin(bearings)
        scenario_tensor = torch.stack((relative_x, relative_y, relative_yaws), dim=-1)
        scenario_sha = sha256_bytes(scenario_tensor.numpy().tobytes())

    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    rows: list[dict] = []
    print(
        f"[T2MIR-ONLINE] variant={args_cli.variant} heldout_task={profile.task_id} "
        f"episodes={args_cli.episodes} replicas={args_cli.num_envs} scenario={scenario_sha}",
        flush=True,
    )

    try:
        for episode_index in range(args_cli.episodes):
            print(
                f"[T2MIR-ONLINE] episode={episode_index} phase=episode_boundary_begin",
                flush=True,
            )
            try:
                boundary_audit = synchronize_vector_episode_boundary(
                    env=env,
                    command_term=command_term,
                    episode_index=episode_index,
                )
            except SystemExit as exc:
                # Isaac Sim uses SystemExit(0) for a few internal shutdown
                # paths.  Treat it as an evaluation failure and expose the
                # exact phase instead of silently returning success without
                # artifacts.
                app_running = bool(simulation_app.is_running())
                raise RuntimeError(
                    f"unexpected SystemExit({exc.code!r}) during episode "
                    f"{episode_index} boundary; simulation_app_running={app_running}"
                ) from exc
            print(
                f"[T2MIR-ONLINE] episode={episode_index} phase=episode_boundary_ready "
                f"mechanism={boundary_audit['mechanism']} "
                f"replicas={boundary_audit['replicas_reset']}",
                flush=True,
            )
            previous_executed_actions = None  # no action-lag state may cross reset
            if args_cli.reset_context_each_goal:
                policy.reset_context()
            prompts = policy.start_episode(episode_index)
            prompt_by_env = prompt_metadata(prompts)
            latest_prompt_by_env = dict(prompt_by_env)
            prompt_update_counts = [0] * args_cli.num_envs
            print(
                f"[T2MIR-ONLINE] episode={episode_index} phase=rollout_begin "
                f"prompt_lengths={dict(Counter(item['selected_length'] for item in prompt_by_env.values()))}",
                flush=True,
            )

            robot = env.unwrapped.scene["robot"]
            initial_positions = robot.data.root_pos_w.clone()
            _, _, initial_yaws = euler_xyz_from_quat(robot.data.root_quat_w)
            goals = scenario_tensor[episode_index]
            controller_cfg = GoalControllerCfg(
                turn_forward_speed=args_cli.turn_forward_speed,
                align_forward_speed=args_cli.align_forward_speed,
            )
            if args_cli.hold_command_mode == "zero":
                controller_cfg.hold_position_gain = 0.0
                controller_cfg.hold_yaw_gain = 0.0
            elif args_cli.hold_command_mode == "safe":
                controller_cfg.hold_allow_backward = False
            controllers = []
            for env_id in range(args_cli.num_envs):
                yaw = float(initial_yaws[env_id].item())
                cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
                rel_x, rel_y, rel_yaw = (float(value) for value in goals[env_id].tolist())
                target_x = float(initial_positions[env_id, 0].item()) + cos_yaw * rel_x - sin_yaw * rel_y
                target_y = float(initial_positions[env_id, 1].item()) + sin_yaw * rel_x + cos_yaw * rel_y
                controllers.append(
                    GoalController(target_x, target_y, yaw + rel_yaw, controller_cfg)
                )

            active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
            hold_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            returns = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
            results: list[dict | None] = [None] * args_cli.num_envs
            max_steps = max(1, round(args_cli.goal_timeout / env.unwrapped.step_dt))

            for step in range(max_steps):
                positions = robot.data.root_pos_w
                _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)
                commands = torch.zeros((args_cli.num_envs, 3), device=device)
                mode_ids = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
                for env_id, controller in enumerate(controllers):
                    if not bool(active[env_id]):
                        continue
                    command, _ = controller.compute(
                        float(positions[env_id, 0].item()),
                        float(positions[env_id, 1].item()),
                        float(yaws[env_id].item()),
                    )
                    commands[env_id] = torch.tensor(command, device=device)
                    mode_ids[env_id] = CONTROLLER_MODE_TO_ID[controller.state]
                command_term.vel_command_b[:] = commands
                obs, _ = env.get_observations()
                step_states = obs.clone()
                active_before = active.clone()

                with torch.inference_mode():
                    policy_actions = policy(obs)
                    if previous_executed_actions is None:
                        previous_executed_actions = policy_actions.clone()
                    executed_actions = (
                        (1.0 - profile.action_lag) * policy_actions
                        + profile.action_lag * previous_executed_actions
                    )
                    previous_executed_actions = executed_actions.clone()
                    obs, rewards, dones, extras = env.step(executed_actions)

                returns[active_before] += rewards[active_before]
                policy.record_transition(
                    step_states, policy_actions, rewards, active_before, controller_modes=mode_ids
                )
                refreshed_prompts = policy.maybe_refresh_prompt(step + 1)
                if refreshed_prompts is not None:
                    latest_prompt_by_env = prompt_metadata(refreshed_prompts)
                    for env_id in torch.nonzero(active_before, as_tuple=False).flatten().tolist():
                        prompt_update_counts[env_id] += 1
                    print(
                        f"[T2MIR-ONLINE] episode={episode_index} "
                        f"prompt_refresh_step={step + 1} "
                        f"lengths={dict(Counter(item['selected_length'] for item in latest_prompt_by_env.values()))}",
                        flush=True,
                    )
                time_outs = extras.get("time_outs")
                if time_outs is None:
                    time_outs = torch.zeros_like(dones, dtype=torch.bool)
                falls = torch.logical_and(dones.bool(), torch.logical_not(time_outs.bool()))
                positions = robot.data.root_pos_w
                _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)

                for env_id, controller in enumerate(controllers):
                    if not bool(active[env_id]):
                        continue
                    distance, _, yaw_error = controller.errors(
                        float(positions[env_id, 0].item()),
                        float(positions[env_id, 1].item()),
                        float(yaws[env_id].item()),
                    )
                    if bool(falls[env_id]):
                        result = "fall"
                    else:
                        within = (
                            controller.state == GoalController.HOLD
                            and distance <= controller.cfg.position_tolerance
                            and abs(yaw_error) <= controller.cfg.yaw_tolerance
                        )
                        hold_steps[env_id] = hold_steps[env_id] + 1 if within else 0
                        result = (
                            "success"
                            if float(hold_steps[env_id].item()) * env.unwrapped.step_dt
                            >= args_cli.goal_hold_time
                            else None
                        )
                    if result is not None:
                        results[env_id] = {
                            "result": result,
                            "time": (step + 1) * env.unwrapped.step_dt,
                            "position_error": float(distance),
                            "yaw_error": abs(float(yaw_error)),
                            "episode_return": float(returns[env_id].item()),
                            "final_state": controller.state,
                        }
                        active[env_id] = False
                if not bool(active.any()):
                    break
                if (step + 1) % 500 == 0:
                    print(
                        f"[T2MIR-ONLINE] episode={episode_index} "
                        f"progress={step + 1}/{max_steps} active={int(active.sum().item())}",
                        flush=True,
                    )

            for env_id, controller in enumerate(controllers):
                if results[env_id] is not None:
                    continue
                positions = robot.data.root_pos_w
                _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)
                distance, _, yaw_error = controller.errors(
                    float(positions[env_id, 0].item()),
                    float(positions[env_id, 1].item()),
                    float(yaws[env_id].item()),
                )
                results[env_id] = {
                    "result": "timeout",
                    "time": args_cli.goal_timeout,
                    "position_error": float(distance),
                    "yaw_error": abs(float(yaw_error)),
                    "episode_return": float(returns[env_id].item()),
                    "final_state": controller.state,
                }
                active[env_id] = False

            trajectories = policy.finish_episode()
            for env_id, result in enumerate(results):
                assert result is not None
                initial_x = float(initial_positions[env_id, 0].item())
                initial_y = float(initial_positions[env_id, 1].item())
                initial_yaw = float(initial_yaws[env_id].item())
                row = {
                    "episode_index": episode_index,
                    "replica_id": env_id,
                    "profile_id": profile.task_id,
                    "goal_seed": args_cli.seed,
                    "relative_x": float(goals[env_id, 0].item()),
                    "relative_y": float(goals[env_id, 1].item()),
                    "relative_yaw": float(goals[env_id, 2].item()),
                    "prompt_selected_length": prompt_by_env[env_id]["selected_length"],
                    "prompt_source_episode_length": prompt_by_env[env_id]["source_episode_length"],
                    "prompt_source_episode_index": prompt_by_env[env_id]["source_episode_index"],
                    "prompt_controller_mode_histogram": json.dumps(
                        prompt_by_env[env_id]["controller_mode_histogram"], sort_keys=True
                    ),
                    "prompt_updates_within_episode": prompt_update_counts[env_id],
                    "final_prompt_selected_length": latest_prompt_by_env[env_id][
                        "selected_length"
                    ],
                    "final_prompt_source_episode_length": latest_prompt_by_env[env_id][
                        "source_episode_length"
                    ],
                    "final_prompt_source_episode_index": latest_prompt_by_env[env_id][
                        "source_episode_index"
                    ],
                    "final_prompt_controller_mode_histogram": json.dumps(
                        latest_prompt_by_env[env_id]["controller_mode_histogram"],
                        sort_keys=True,
                    ),
                    "recorded_episode_length": trajectories[env_id].length,
                    **result,
                    "policy_kind": "t2mir",
                    "checkpoint_sha256": policy.provenance["checkpoint_sha256"],
                    "scenario_sha256": scenario_sha,
                    "initial_x": initial_x,
                    "initial_y": initial_y,
                    "initial_yaw": initial_yaw,
                    "initial_pose_sha256": pose_sha256(initial_x, initial_y, initial_yaw),
                }
                rows.append(row)

            selected = rows[-args_cli.num_envs :]
            successes = sum(row["result"] == "success" for row in selected)
            falls_count = sum(row["result"] == "fall" for row in selected)
            print(
                f"[T2MIR-ONLINE] episode={episode_index} "
                f"prompt_lengths={dict(Counter(row['prompt_selected_length'] for row in selected))} "
                f"success={successes}/{args_cli.num_envs} falls={falls_count}",
                flush=True,
            )
    finally:
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary_csv = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("pipeline.protocols.ppo_online_protocol.py").resolve(),
        Path(__file__).with_name("pipeline.protocols.t2mir_online_context.py").resolve(),
        Path(__file__).with_name("pipeline.protocols.goal_controller.py").resolve(),
        Path(__file__).with_name("pipeline.protocols.dynamics_profile.py").resolve(),
        Path(__file__).with_name("pipeline.protocols.paired_goal_protocol.py").resolve(),
    ]
    report = {
        "format_version": 1,
        "evaluation": (
            "source_faithful_stationary_online_dpt"
            if args_cli.prompt_update_mode == "episode"
            else "robotlab_streaming_block_online_dpt"
        ),
        "variant": args_cli.variant,
        "checkpoint": policy.provenance,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["sha256"],
        "profile": profile.__dict__,
        "configured_dynamics": configured,
        "seed": args_cli.seed,
        "scenario_sha256": scenario_sha,
        "paired_goal_manifest": str(paired_goal_path) if paired_goal_path else None,
        "paired_goal_manifest_sha256": (
            sha256_file(paired_goal_path) if paired_goal_path else None
        ),
        "goal_contract": {
            "episodes": args_cli.episodes,
            "replicas": args_cli.num_envs,
            "min_distance": args_cli.goal_min_distance,
            "max_distance": args_cli.goal_max_distance,
            "timeout": args_cli.goal_timeout,
            "hold_time": args_cli.goal_hold_time,
            "turn_forward_speed": args_cli.turn_forward_speed,
            "align_forward_speed": args_cli.align_forward_speed,
            "hold_command_mode": args_cli.hold_command_mode,
            "reset_context_each_goal": args_cli.reset_context_each_goal,
        },
        "prompt_protocol": policy.context.protocol_metadata(),
        "reset_protocol": {
            "full_vector_reset_before_each_episode": True,
            "episode_boundary_mechanism": EPISODE_BOUNDARY_MECHANISM,
            "explicit_global_reset_calls_after_wrapper_construction": 0,
            "boundary_transition_excluded_from_metrics_and_prompt": True,
            "action_lag_state_crosses_reset": False,
            "initial_pose_recorded_per_row": True,
            "deterministic_reset": True,
            **reset_contract,
        },
        "result_csv": str(csv_path),
        "result_csv_sha256": sha256_file(csv_path),
        "summary": summarize(rows, args_cli.episodes),
        "code_sha256": {str(path.relative_to(REPO)): sha256_file(path) for path in code_paths},
        "command": sys.argv,
    }
    write_json_atomic(report_path, report)
    print(f"[T2MIR-ONLINE] COMPLETE summary={json.dumps(report['summary'])}", flush=True)
    print(f"[T2MIR-ONLINE] report={report_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Also closes Kit when argument/checkpoint/environment validation fails
        # before the environment-specific ``finally`` block is entered.
        simulation_app.close()
