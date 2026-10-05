"""Fair stationary held-out no-context baseline evaluation in Isaac Sim.

This baseline intentionally mirrors ``evaluate_t2mir_online.py``: a held-out
dynamics profile remains fixed, a fresh relative goal is used for every
episode/replica, every episode begins with a full vector-environment auto-reset,
and action-lag state never crosses a reset.  PPO uses its deterministic actor
mean; the query-only MLP uses its deterministic supervised output.  Neither
backend has a prompt or an online update path.
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
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from omni.isaac.lab.app import AppLauncher

from pipeline.protocols.ppo_online_protocol import (
    DEFAULT_EPISODES,
    DEFAULT_NUM_ENVS,
    EPISODE_BOUNDARY_MECHANISM,
    POLICY_KIND,
    freeze_evaluation_reset,
    generate_t2mir_identical_scenarios,
    pose_sha256,
    no_context_prompt_protocol,
    prepare_static_contract,
    sha256_file,
    summarize,
    synchronize_vector_episode_boundary,
    validate_protocol_values,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs/g1_dynamics_48.json"
DEFAULT_OUTPUT = REPO / "outputs/ppo_online_stationary"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0",
    help="RobotLab environment ID.",
)
parser.add_argument(
    "--policy-backend",
    choices=("ppo", "query_only"),
    default="ppo",
    help="No-context policy backend evaluated under the identical goal protocol.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Exact PPO or query-only checkpoint selected by --policy-backend.",
)
parser.add_argument(
    "--checkpoint-sha256",
    default=None,
    help="Optional expected checkpoint SHA-256; a mismatch is fatal before Kit starts.",
)
parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
parser.add_argument("--profile-id", type=int, required=True)
parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=DEFAULT_NUM_ENVS)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--goal-min-distance", type=float, default=1.0)
parser.add_argument("--goal-max-distance", type=float, default=4.0)
parser.add_argument("--goal-timeout", type=float, default=40.0)
parser.add_argument("--goal-hold-time", type=float, default=2.0)
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

# Reject invalid or non-held-out work before starting the expensive simulator.
try:
    validate_protocol_values(
        episodes=args_cli.episodes,
        num_envs=args_cli.num_envs,
        goal_min_distance=args_cli.goal_min_distance,
        goal_max_distance=args_cli.goal_max_distance,
        goal_timeout=args_cli.goal_timeout,
        goal_hold_time=args_cli.goal_hold_time,
    )
    static_contract = prepare_static_contract(
        checkpoint=args_cli.checkpoint,
        expected_checkpoint_sha256=args_cli.checkpoint_sha256,
        manifest=args_cli.manifest,
        profile_id=args_cli.profile_id,
    )
except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
    parser.error(str(error))

output_dir = args_cli.output_dir.expanduser().resolve()
result_stem = (
    f"{args_cli.policy_backend}_task{static_contract.profile.task_id:02d}_seed{args_cli.seed}"
)
csv_path = output_dir / f"{result_stem}.csv"
report_path = output_dir / f"{result_stem}.json"
if not args_cli.force and (csv_path.exists() or report_path.exists()):
    parser.error(
        f"refusing to reuse existing result {result_stem}; "
        "pass --force only for an intentional rerun"
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Isaac imports must happen after SimulationApp starts."""

import gymnasium as gym
import torch

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.math import euler_xyz_from_quat
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

import robot_lab.tasks  # noqa: F401,E402
from robot_lab.third_party.rsl_rl.runners import OnPolicyRunner

from pipeline.protocols.dynamics_profile import configure_env_dynamics
from pipeline.evaluation.context_policy import QueryOnlyPolicy
from pipeline.protocols.goal_controller import GoalController


class RobotLabRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Compatibility wrapper for the bundled RSL-RL interface."""

    def get_privileged_observations(self) -> torch.Tensor | None:
        _, extras = self.get_observations()
        return extras["observations"].get("critic")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    profile = static_contract.profile
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

    if args_cli.policy_backend == "ppo":
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
        ppo_runner.load(str(static_contract.checkpoint_path), load_optimizer=False)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        checkpoint_provenance = {
            "checkpoint": str(static_contract.checkpoint_path),
            "checkpoint_sha256": static_contract.checkpoint_sha256,
            "checkpoint_step": int(ppo_runner.current_learning_iteration),
            "state_dim": int(env.num_obs),
            "action_dim": int(env.num_actions),
            "policy_kind": POLICY_KIND,
            "action_semantics": "deterministic_actor_mean",
        }
    else:
        policy = QueryOnlyPolicy(static_contract.checkpoint_path, device)
        if int(policy.state_mean.numel()) != int(env.num_obs):
            raise ValueError(
                "query-only checkpoint/environment state dimension mismatch: "
                f"checkpoint={policy.state_mean.numel()} environment={env.num_obs}"
            )
        action_dim = int(policy.model.net[-1].out_features)
        if action_dim != int(env.num_actions):
            raise ValueError(
                "query-only checkpoint/environment action dimension mismatch: "
                f"checkpoint={action_dim} environment={env.num_actions}"
            )
        checkpoint_provenance = {
            "checkpoint": str(static_contract.checkpoint_path),
            "checkpoint_sha256": static_contract.checkpoint_sha256,
            "checkpoint_step": None,
            "state_dim": int(policy.state_mean.numel()),
            "action_dim": action_dim,
            "policy_kind": "query_only",
            "action_semantics": "deterministic_supervised_actor_mean",
        }

    scenario_tensor, scenario_sha = generate_t2mir_identical_scenarios(
        seed=args_cli.seed,
        episodes=args_cli.episodes,
        num_envs=args_cli.num_envs,
        goal_min_distance=args_cli.goal_min_distance,
        goal_max_distance=args_cli.goal_max_distance,
    )
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    rows: list[dict] = []
    print(
        f"[NO-CONTEXT-ONLINE] backend={args_cli.policy_backend} "
        f"heldout_task={profile.task_id} episodes={args_cli.episodes} "
        f"replicas={args_cli.num_envs} scenario={scenario_sha} "
        f"checkpoint={static_contract.checkpoint_sha256}",
        flush=True,
    )

    try:
        for episode_index in range(args_cli.episodes):
            print(
                f"[NO-CONTEXT-ONLINE] episode={episode_index} phase=episode_boundary_begin",
                flush=True,
            )
            boundary_audit = synchronize_vector_episode_boundary(
                env=env,
                command_term=command_term,
                episode_index=episode_index,
            )
            print(
                f"[NO-CONTEXT-ONLINE] episode={episode_index} phase=episode_boundary_ready "
                f"mechanism={boundary_audit['mechanism']} "
                f"replicas={boundary_audit['replicas_reset']}",
                flush=True,
            )
            previous_executed_actions = None  # no action-lag state may cross reset

            robot = env.unwrapped.scene["robot"]
            initial_positions = robot.data.root_pos_w.clone()
            _, _, initial_yaws = euler_xyz_from_quat(robot.data.root_quat_w)
            goals = scenario_tensor[episode_index]
            controllers = []
            for env_id in range(args_cli.num_envs):
                yaw = float(initial_yaws[env_id].item())
                cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
                rel_x, rel_y, rel_yaw = (float(value) for value in goals[env_id].tolist())
                target_x = (
                    float(initial_positions[env_id, 0].item())
                    + cos_yaw * rel_x
                    - sin_yaw * rel_y
                )
                target_y = (
                    float(initial_positions[env_id, 1].item())
                    + sin_yaw * rel_x
                    + cos_yaw * rel_y
                )
                controllers.append(GoalController(target_x, target_y, yaw + rel_yaw))

            active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
            hold_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            returns = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
            recorded_lengths = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            results: list[dict | None] = [None] * args_cli.num_envs
            max_steps = max(1, round(args_cli.goal_timeout / env.unwrapped.step_dt))

            for step in range(max_steps):
                positions = robot.data.root_pos_w
                _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)
                commands = torch.zeros((args_cli.num_envs, 3), device=device)
                for env_id, controller in enumerate(controllers):
                    if not bool(active[env_id]):
                        continue
                    command, _ = controller.compute(
                        float(positions[env_id, 0].item()),
                        float(positions[env_id, 1].item()),
                        float(yaws[env_id].item()),
                    )
                    commands[env_id] = torch.tensor(command, device=device)
                command_term.vel_command_b[:] = commands
                obs, _ = env.get_observations()
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
                    _, rewards, dones, extras = env.step(executed_actions)

                returns[active_before] += rewards[active_before]
                recorded_lengths[active_before] += 1
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

            for env_id, result in enumerate(results):
                assert result is not None
                initial_x = float(initial_positions[env_id, 0].item())
                initial_y = float(initial_positions[env_id, 1].item())
                initial_yaw = float(initial_yaws[env_id].item())
                row = {
                    # Keep the T2MIR CSV fields and order first.
                    "episode_index": episode_index,
                    "replica_id": env_id,
                    "profile_id": profile.task_id,
                    "goal_seed": args_cli.seed,
                    "relative_x": float(goals[env_id, 0].item()),
                    "relative_y": float(goals[env_id, 1].item()),
                    "relative_yaw": float(goals[env_id, 2].item()),
                    "prompt_selected_length": 0,
                    "prompt_source_episode_length": 0,
                    "prompt_source_episode_index": None,
                    "prompt_controller_mode_histogram": "{}",
                    "recorded_episode_length": int(recorded_lengths[env_id].item()),
                    **result,
                    # Baseline identity and reset-audit fields.
                    "policy_kind": args_cli.policy_backend,
                    "checkpoint_sha256": static_contract.checkpoint_sha256,
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
                f"[NO-CONTEXT-ONLINE] episode={episode_index} "
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
        Path(__file__).with_name("pipeline.protocols.goal_controller.py").resolve(),
        Path(__file__).with_name("pipeline.protocols.dynamics_profile.py").resolve(),
    ]
    report = {
        "format_version": 1,
        "evaluation": f"fair_stationary_online_{args_cli.policy_backend}",
        "policy_kind": args_cli.policy_backend,
        "variant": None,
        "checkpoint": checkpoint_provenance,
        "manifest": str(static_contract.manifest_path),
        "manifest_sha256": static_contract.manifest["sha256"],
        "profile": profile.__dict__,
        "configured_dynamics": configured,
        "seed": args_cli.seed,
        "scenario_sha256": scenario_sha,
        "goal_contract": {
            "episodes": args_cli.episodes,
            "replicas": args_cli.num_envs,
            "min_distance": args_cli.goal_min_distance,
            "max_distance": args_cli.goal_max_distance,
            "timeout": args_cli.goal_timeout,
            "hold_time": args_cli.goal_hold_time,
        },
        "prompt_protocol": no_context_prompt_protocol(
            policy_kind=args_cli.policy_backend,
            action_semantics=checkpoint_provenance["action_semantics"],
        ),
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
    print(f"[NO-CONTEXT-ONLINE] COMPLETE summary={json.dumps(report['summary'])}", flush=True)
    print(f"[NO-CONTEXT-ONLINE] report={report_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
