# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
from pipeline.expert import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--primitive_test",
    action="store_true",
    default=False,
    help="Run a deterministic velocity-command sequence and report tracking errors.",
)
parser.add_argument(
    "--primitive_duration",
    type=float,
    default=5.0,
    help="Duration in seconds for each command in --primitive_test mode.",
)
parser.add_argument("--goal_test", action="store_true", default=False, help="Navigate to a relative planar goal.")
parser.add_argument("--goal_x", type=float, default=3.0, help="Goal x in the robot's initial frame, in meters.")
parser.add_argument("--goal_y", type=float, default=2.0, help="Goal y in the robot's initial frame, in meters.")
parser.add_argument(
    "--goal_yaw", type=float, default=1.5708, help="Final yaw relative to the initial yaw, in radians."
)
parser.add_argument("--goal_timeout", type=float, default=60.0, help="Goal-test timeout in simulation seconds.")
parser.add_argument(
    "--goal_hold_time", type=float, default=2.0, help="Required continuous hold time before declaring success."
)
parser.add_argument("--goal_batch", type=int, default=0, help="Evaluate this many random goals in parallel.")
parser.add_argument("--goal_seed", type=int, default=42, help="Random seed used by --goal_batch.")
parser.add_argument("--goal_min_distance", type=float, default=1.0, help="Minimum random-goal distance in meters.")
parser.add_argument("--goal_max_distance", type=float, default=4.0, help="Maximum random-goal distance in meters.")
parser.add_argument(
    "--goal_results", type=str, default=None, help="Optional CSV path for random-goal evaluation results."
)
parser.add_argument("--record_trajectories", action="store_true", help="Record per-step tensors in goal-batch mode.")
parser.add_argument(
    "--trajectory_output", type=str, default=None, help="Output .pt path used by --record_trajectories."
)
parser.add_argument("--trajectory_task_id", type=int, default=0, help="Dynamics task label stored in trajectories.")
parser.add_argument(
    "--trajectory_policy_role",
    choices=("prompt", "expert"),
    default="prompt",
    help="Whether the loaded checkpoint supplies prompt behavior or expert query labels.",
)
parser.add_argument(
    "--motor_strength",
    type=float,
    default=1.0,
    help="Scale all robot actuator effort limits (1.0 is the unmodified baseline).",
)
parser.add_argument("--payload_kg", type=float, default=0.0,
                    help="Fixed mass added to torso_link in kilograms.")
parser.add_argument("--friction", type=float, default=None,
                    help="Fixed static/dynamic robot friction coefficient.")
parser.add_argument("--deterministic_dynamics", action="store_true",
                    help="Disable background actuator-gain and joint-parameter randomization.")
parser.add_argument(
    "--action_delay_steps",
    type=int,
    default=0,
    help="Delay policy actions by this many control steps (one step is normally 20 ms).",
)
parser.add_argument(
    "--action_lag",
    type=float,
    default=0.0,
    help="First-order action lag in [0, 1): executed=(1-lag)*new + lag*previous_executed.",
)
parser.add_argument("--action_lag_drift", action="store_true", help="Enable a continuous rise-hold-recovery lag schedule.")
parser.add_argument("--lag_drift_start", type=float, default=0.20, help="Lag before and after the drift.")
parser.add_argument("--lag_drift_peak", type=float, default=0.44, help="Peak lag reached by the drift.")
parser.add_argument("--lag_drift_warmup", type=float, default=5.0, help="Seconds to hold the starting lag.")
parser.add_argument("--lag_drift_ramp", type=float, default=15.0, help="Seconds to ramp from start to peak lag.")
parser.add_argument("--lag_drift_hold", type=float, default=10.0, help="Seconds to hold the peak lag.")
parser.add_argument("--lag_drift_recovery", type=float, default=10.0, help="Seconds to ramp back to the starting lag.")
parser.add_argument(
    "--policy_backend", choices=("ppo", "query_only", "oracle_dynamics", "t2mir"), default="ppo",
    help="Low-level policy used for closed-loop evaluation.",
)
parser.add_argument(
    "--stochastic_policy",
    action="store_true",
    help="Sample actions from the loaded PPO distribution instead of using its deterministic mean.",
)
parser.add_argument(
    "--policy_action_seed",
    type=int,
    default=None,
    help="Torch RNG seed for --stochastic_policy (defaults to --goal_seed).",
)
parser.add_argument(
    "--skip_policy_export",
    action="store_true",
    help="Do not overwrite ONNX/JIT exports while collecting or evaluating checkpoints.",
)
parser.add_argument("--offline_checkpoint", type=str, default=None,
                    help="Checkpoint for query-only, oracle-dynamics, or T2MIR inference.")
parser.add_argument("--t2mir_dataset_dir", type=str,
                    default="data/t2mir/RobotLab-G1-MultiDynamics/formal_aligned")
parser.add_argument("--prompt_task_id", type=int, default=None,
                    help="Dynamics task whose offline trajectory supplies the fixed T2MIR prompt.")
parser.add_argument("--prompt_seed", type=int, default=42, help="Deterministic T2MIR prompt sampler seed.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import math
import os
import torch
import time
from datetime import datetime
from pathlib import Path

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import robot_lab.tasks  # noqa: F401
from robot_lab.third_party.rsl_rl.runners import OnPolicyRunner

from pipeline.protocols.goal_controller import GoalController
from pipeline.protocols.dynamics_profile import configure_env_dynamics
from pipeline.evaluation.context_policy import OracleDynamicsPolicy, QueryOnlyPolicy, T2MIRPolicy


class RobotLabRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Compatibility wrapper for the RSL-RL 1.0 interface."""

    def get_privileged_observations(self) -> torch.Tensor | None:
        _, extras = self.get_observations()
        return extras["observations"].get("critic")


PRIMITIVE_COMMANDS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("slow_forward", (0.3, 0.0, 0.0)),
    ("fast_forward", (0.9, 0.0, 0.0)),
    ("left_lateral", (0.0, 0.3, 0.0)),
    ("right_lateral", (0.0, -0.3, 0.0)),
    ("turn_left", (0.0, 0.0, 0.7)),
    ("turn_right", (0.0, 0.0, -0.7)),
    ("forward_arc", (0.6, 0.0, 0.4)),
    ("final_stand", (0.0, 0.0, 0.0)),
)

CONTROLLER_MODE_TO_ID = {
    GoalController.TURN_TO_GOAL: 0,
    GoalController.WALK_TO_GOAL: 1,
    GoalController.ALIGN_FINAL_YAW: 2,
    GoalController.HOLD: 3,
}


def action_lag_schedule(elapsed_s: float) -> tuple[float, str]:
    """Return the continuous lag value and phase for the configured time."""
    warmup_end = args_cli.lag_drift_warmup
    ramp_end = warmup_end + args_cli.lag_drift_ramp
    hold_end = ramp_end + args_cli.lag_drift_hold
    recovery_end = hold_end + args_cli.lag_drift_recovery

    if elapsed_s < warmup_end:
        return args_cli.lag_drift_start, "warmup"
    if elapsed_s < ramp_end:
        progress = (elapsed_s - warmup_end) / args_cli.lag_drift_ramp
        lag = args_cli.lag_drift_start + progress * (args_cli.lag_drift_peak - args_cli.lag_drift_start)
        return lag, "ramp_up"
    if elapsed_s < hold_end:
        return args_cli.lag_drift_peak, "peak_hold"
    if elapsed_s < recovery_end:
        progress = (elapsed_s - hold_end) / args_cli.lag_drift_recovery
        lag = args_cli.lag_drift_peak + progress * (args_cli.lag_drift_start - args_cli.lag_drift_peak)
        return lag, "recovery"
    return args_cli.lag_drift_start, "recovered"


def main():
    """Play with RSL-RL agent."""
    selected_modes = int(args_cli.primitive_test) + int(args_cli.goal_test) + int(args_cli.goal_batch > 0)
    if selected_modes > 1:
        raise ValueError("--primitive_test, --goal_test, and --goal_batch are mutually exclusive")
    if args_cli.goal_test and args_cli.num_envs not in (None, 1):
        raise ValueError("--goal_test currently supports exactly one environment; use --num_envs 1")
    if args_cli.goal_batch < 0:
        raise ValueError("--goal_batch must be non-negative")
    if args_cli.goal_batch > 0 and args_cli.goal_min_distance >= args_cli.goal_max_distance:
        raise ValueError("--goal_min_distance must be smaller than --goal_max_distance")
    if args_cli.record_trajectories and args_cli.goal_batch <= 0:
        raise ValueError("--record_trajectories currently requires --goal_batch")
    if args_cli.trajectory_output is not None and not args_cli.record_trajectories:
        raise ValueError("--trajectory_output requires --record_trajectories")
    if args_cli.motor_strength <= 0.0:
        raise ValueError("--motor_strength must be greater than zero")
    if args_cli.payload_kg < 0.0:
        raise ValueError("--payload_kg must be non-negative")
    if args_cli.friction is not None and args_cli.friction <= 0.0:
        raise ValueError("--friction must be positive")
    if args_cli.action_delay_steps < 0:
        raise ValueError("--action_delay_steps must be non-negative")
    if not 0.0 <= args_cli.action_lag < 1.0:
        raise ValueError("--action_lag must be in [0, 1)")
    if not 0.0 <= args_cli.lag_drift_start < 1.0 or not 0.0 <= args_cli.lag_drift_peak < 1.0:
        raise ValueError("--lag_drift_start and --lag_drift_peak must be in [0, 1)")
    if args_cli.lag_drift_peak < args_cli.lag_drift_start:
        raise ValueError("--lag_drift_peak must be greater than or equal to --lag_drift_start")
    if args_cli.lag_drift_warmup < 0.0 or min(
        args_cli.lag_drift_ramp, args_cli.lag_drift_hold, args_cli.lag_drift_recovery
    ) <= 0.0:
        raise ValueError("Drift warmup must be non-negative and ramp/hold/recovery durations must be positive")
    lag_modes = int(args_cli.action_delay_steps > 0) + int(args_cli.action_lag > 0.0) + int(args_cli.action_lag_drift)
    if lag_modes > 1:
        raise ValueError("Use only one of --action_delay_steps, --action_lag, or --action_lag_drift")
    if args_cli.policy_backend != "ppo" and args_cli.offline_checkpoint is None:
        raise ValueError(f"--policy_backend {args_cli.policy_backend} requires --offline_checkpoint")
    if args_cli.stochastic_policy and args_cli.policy_backend != "ppo":
        raise ValueError("--stochastic_policy is only supported with --policy_backend ppo")
    if args_cli.policy_action_seed is not None and args_cli.policy_action_seed < 0:
        raise ValueError("--policy_action_seed must be non-negative")
    if args_cli.policy_backend == "t2mir" and args_cli.prompt_task_id is None:
        raise ValueError("--policy_backend t2mir requires --prompt_task_id")

    policy_action_mode = "stochastic" if args_cli.stochastic_policy else "deterministic"
    policy_action_seed = (
        args_cli.policy_action_seed if args_cli.policy_action_seed is not None else args_cli.goal_seed
    ) if args_cli.stochastic_policy else None

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    dynamics = configure_env_dynamics(
        env_cfg,
        motor_strength=args_cli.motor_strength,
        payload_kg=args_cli.payload_kg,
        friction=args_cli.friction,
        deterministic=args_cli.deterministic_dynamics,
    )
    dynamics["action_lag"] = args_cli.action_lag
    print(f"[STRESS] Fixed dynamics profile: {dynamics}")

    # make a smaller scene for play
    if args_cli.goal_batch > 0:
        env_cfg.scene.num_envs = args_cli.goal_batch
        # Use the same seed for target generation and simulator reset noise so
        # baseline/stress comparisons replay the same evaluation conditions.
        env_cfg.seed = args_cli.goal_seed
    else:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    if args_cli.primitive_test or args_cli.goal_test or args_cli.goal_batch > 0:
        # The test injects body-frame velocity commands directly. Disable the
        # heading controller and automatic resampling so they cannot overwrite
        # the deterministic test sequence.
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    if args_cli.goal_test or args_cli.goal_batch > 0:
        # Prevent the environment's regular 20-second timeout from resetting a
        # robot in the middle of a longer navigation test.
        env_cfg.episode_length_s = args_cli.goal_timeout + 10.0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.policy_backend == "ppo":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        resume_path = os.path.abspath(args_cli.offline_checkpoint)
        log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RobotLabRslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading {args_cli.policy_backend} checkpoint from: {resume_path}")
    device = torch.device(env.unwrapped.device)
    if args_cli.policy_backend == "ppo":
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        if args_cli.stochastic_policy:
            # Match RSL-RL's rollout semantics: normalize observations exactly
            # as in training, then sample from the learned Normal distribution.
            torch.manual_seed(policy_action_seed)
            ppo_runner.eval_mode()
            ppo_runner.alg.actor_critic.to(device)
            ppo_runner.obs_normalizer.to(device)

            def policy(observations):
                return ppo_runner.alg.actor_critic.act(ppo_runner.obs_normalizer(observations))

        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        seed_text = f" seed={policy_action_seed}" if policy_action_seed is not None else ""
        print(f"[POLICY] PPO action mode={policy_action_mode}{seed_text}")
        if not args_cli.skip_policy_export:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            export_policy_as_onnx(actor_critic=ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer,
                                  path=export_model_dir, filename="policy.onnx")
            export_policy_as_jit(actor_critic=ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer,
                                 path=export_model_dir, filename="policy.pt")
    elif args_cli.policy_backend == "query_only":
        policy = QueryOnlyPolicy(Path(resume_path), device)
    elif args_cli.policy_backend == "oracle_dynamics":
        friction = args_cli.friction if args_cli.friction is not None else 1.0
        policy = OracleDynamicsPolicy(Path(resume_path), device,
                                      [args_cli.action_lag, args_cli.motor_strength, args_cli.payload_kg, friction])
    else:
        repo = Path(__file__).resolve().parents[2]
        dataset_dir = Path(args_cli.t2mir_dataset_dir)
        if not dataset_dir.is_absolute():
            dataset_dir = repo / dataset_dir
        policy = T2MIRPolicy(
            Path(resume_path), dataset_dir, args_cli.prompt_task_id, args_cli.prompt_seed, device,
            repo / "methods/t2mir",
        )
        print(f"[T2MIR] matched prompt task={args_cli.prompt_task_id} "
              f"episodes={policy.prompt_episode_ids} seed={args_cli.prompt_seed}")

    # reset environment
    obs, _ = env.get_observations()
    delayed_actions = None
    previous_executed_actions = None
    current_action_lag = args_cli.action_lag
    lag_phase = None
    if args_cli.action_delay_steps > 0:
        delay_ms = 1000.0 * args_cli.action_delay_steps * env.unwrapped.step_dt
        print(f"[STRESS] Action delay={args_cli.action_delay_steps} steps ({delay_ms:.1f} ms)")
    elif args_cli.action_lag > 0.0:
        print(f"[STRESS] First-order action lag={args_cli.action_lag:.3f}")
    elif args_cli.action_lag_drift:
        total_drift_time = (
            args_cli.lag_drift_warmup
            + args_cli.lag_drift_ramp
            + args_cli.lag_drift_hold
            + args_cli.lag_drift_recovery
        )
        print(
            f"[STRESS] Continuous action-lag drift={args_cli.lag_drift_start:.3f}"
            f"->{args_cli.lag_drift_peak:.3f}->{args_cli.lag_drift_start:.3f} "
            f"over {total_drift_time:.1f} s"
        )
    timestep = 0
    sim_step = 0
    primitive_index = 0
    primitive_step = 0
    primitive_steps = max(1, round(args_cli.primitive_duration / env.unwrapped.step_dt))
    primitive_error_xy = 0.0
    primitive_error_yaw = 0.0
    primitive_falls = 0
    command_term = None
    if args_cli.primitive_test or args_cli.goal_test or args_cli.goal_batch > 0:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
    if args_cli.primitive_test:
        print("[INFO] Starting deterministic primitive test:")
        print(f"       {len(PRIMITIVE_COMMANDS)} primitives x {args_cli.primitive_duration:.1f} s")

    goal_controller = None
    goal_marker = None
    goal_step = 0
    goal_hold_steps = 0
    last_goal_state = None
    if args_cli.goal_test:
        robot = env.unwrapped.scene["robot"]
        initial_position = robot.data.root_pos_w[0].clone()
        _, _, initial_yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
        initial_yaw = initial_yaw_tensor[0].item()
        cos_yaw = torch.cos(initial_yaw_tensor[0]).item()
        sin_yaw = torch.sin(initial_yaw_tensor[0]).item()
        target_x = initial_position[0].item() + cos_yaw * args_cli.goal_x - sin_yaw * args_cli.goal_y
        target_y = initial_position[1].item() + sin_yaw * args_cli.goal_x + cos_yaw * args_cli.goal_y
        target_yaw = initial_yaw + args_cli.goal_yaw
        goal_controller = GoalController(target_x, target_y, target_yaw)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.5, 0.5, 0.5)
        goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/goal_pose"))
        marker_position = torch.tensor([[target_x, target_y, 0.05]], device=env.unwrapped.device)
        marker_yaw = torch.tensor([target_yaw], device=env.unwrapped.device)
        marker_zeros = torch.zeros_like(marker_yaw)
        marker_orientation = quat_from_euler_xyz(marker_zeros, marker_zeros, marker_yaw)
        goal_marker.visualize(marker_position, marker_orientation)
        print(
            f"[GOAL] target_world=({target_x:.2f}, {target_y:.2f}, {target_yaw:.2f} rad) "
            f"target_relative=({args_cli.goal_x:.2f}, {args_cli.goal_y:.2f}, {args_cli.goal_yaw:.2f} rad)"
        )

    batch_controllers = None
    batch_target_relative = None
    batch_active = None
    batch_hold_steps = None
    batch_returns = None
    batch_results = None
    trajectory_chunks = None
    batch_mode_ids = None
    if args_cli.record_trajectories:
        trajectory_chunks = {
            key: []
            for key in (
                "states",
                "next_states",
                "commands",
                "policy_actions",
                "executed_actions",
                "rewards",
                "dones",
                "time_outs",
                "env_ids",
                "episode_steps",
                "goal_relative",
                "controller_modes",
                "tracking_errors",
                "action_lags",
                "motor_strengths",
                "payload_kg",
                "frictions",
            )
        }
    if args_cli.goal_batch > 0:
        robot = env.unwrapped.scene["robot"]
        initial_positions = robot.data.root_pos_w.clone()
        _, _, initial_yaws = euler_xyz_from_quat(robot.data.root_quat_w)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args_cli.goal_seed)
        distances = args_cli.goal_min_distance + (
            args_cli.goal_max_distance - args_cli.goal_min_distance
        ) * torch.rand(args_cli.goal_batch, generator=generator)
        bearings = -math.pi + 2.0 * math.pi * torch.rand(args_cli.goal_batch, generator=generator)
        final_relative_yaws = -math.pi + 2.0 * math.pi * torch.rand(args_cli.goal_batch, generator=generator)
        relative_x = distances * torch.cos(bearings)
        relative_y = distances * torch.sin(bearings)
        batch_target_relative = torch.stack((relative_x, relative_y, final_relative_yaws), dim=1)

        batch_controllers = []
        marker_positions = []
        marker_yaws = []
        for index in range(args_cli.goal_batch):
            initial_yaw = initial_yaws[index].item()
            cos_yaw = math.cos(initial_yaw)
            sin_yaw = math.sin(initial_yaw)
            target_x = initial_positions[index, 0].item() + cos_yaw * relative_x[index].item() - sin_yaw * relative_y[index].item()
            target_y = initial_positions[index, 1].item() + sin_yaw * relative_x[index].item() + cos_yaw * relative_y[index].item()
            target_yaw = initial_yaw + final_relative_yaws[index].item()
            batch_controllers.append(GoalController(target_x, target_y, target_yaw))
            marker_positions.append((target_x, target_y, 0.05))
            marker_yaws.append(target_yaw)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.35, 0.35, 0.35)
        goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/batch_goal_poses"))
        marker_positions_tensor = torch.tensor(marker_positions, device=env.unwrapped.device)
        marker_yaws_tensor = torch.tensor(marker_yaws, device=env.unwrapped.device)
        marker_zeros = torch.zeros_like(marker_yaws_tensor)
        marker_orientations = quat_from_euler_xyz(marker_zeros, marker_zeros, marker_yaws_tensor)
        goal_marker.visualize(marker_positions_tensor, marker_orientations)

        batch_active = torch.ones(args_cli.goal_batch, dtype=torch.bool, device=env.unwrapped.device)
        batch_hold_steps = torch.zeros(args_cli.goal_batch, dtype=torch.long, device=env.unwrapped.device)
        batch_returns = torch.zeros(args_cli.goal_batch, dtype=torch.float32, device=env.unwrapped.device)
        batch_results = [None] * args_cli.goal_batch
        print(
            f"[GOAL-BATCH] Starting {args_cli.goal_batch} random goals: "
            f"distance=[{args_cli.goal_min_distance:.2f}, {args_cli.goal_max_distance:.2f}] m "
            f"seed={args_cli.goal_seed} timeout={args_cli.goal_timeout:.1f} s"
        )

    # simulate environment
    while simulation_app.is_running():
        if args_cli.primitive_test:
            primitive_name, primitive_command = PRIMITIVE_COMMANDS[primitive_index]
            command_term.vel_command_b[:] = torch.tensor(
                primitive_command, device=env.unwrapped.device, dtype=command_term.vel_command_b.dtype
            )
            # Refresh observations after replacing the generated command so the
            # policy sees the command selected for this control step.
            obs, _ = env.get_observations()
        elif args_cli.goal_test:
            robot = env.unwrapped.scene["robot"]
            position = robot.data.root_pos_w[0]
            _, _, yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            command, goal_info = goal_controller.compute(position[0].item(), position[1].item(), yaw_tensor[0].item())
            command_term.vel_command_b[:] = torch.tensor(
                command, device=env.unwrapped.device, dtype=command_term.vel_command_b.dtype
            )
            obs, _ = env.get_observations()
            if goal_info["state"] != last_goal_state:
                print(
                    f"[GOAL] state={goal_info['state']} distance={goal_info['distance']:.3f} m "
                    f"heading_error={goal_info['heading_error']:.3f} rad "
                    f"final_yaw_error={goal_info['final_yaw_error']:.3f} rad command={command}"
                )
                last_goal_state = goal_info["state"]
        elif args_cli.goal_batch > 0:
            robot = env.unwrapped.scene["robot"]
            positions = robot.data.root_pos_w
            _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)
            commands = torch.zeros((args_cli.goal_batch, 3), device=env.unwrapped.device)
            if args_cli.record_trajectories:
                batch_mode_ids = torch.full(
                    (args_cli.goal_batch,), -1, dtype=torch.long, device=env.unwrapped.device
                )
            for index, controller in enumerate(batch_controllers):
                if not batch_active[index]:
                    continue
                command, _ = controller.compute(
                    positions[index, 0].item(), positions[index, 1].item(), yaws[index].item()
                )
                commands[index] = torch.tensor(command, device=env.unwrapped.device)
                if args_cli.record_trajectories:
                    batch_mode_ids[index] = CONTROLLER_MODE_TO_ID[controller.state]
            command_term.vel_command_b[:] = commands
            obs, _ = env.get_observations()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            step_states = obs.clone() if args_cli.record_trajectories else None
            active_before_step = batch_active.clone() if args_cli.goal_batch > 0 else None
            policy_actions = policy(obs)
            actions = policy_actions
            if args_cli.action_delay_steps > 0:
                if delayed_actions is None:
                    delayed_actions = [torch.zeros_like(actions) for _ in range(args_cli.action_delay_steps)]
                delayed_actions.append(actions.clone())
                actions = delayed_actions.pop(0)
            elif args_cli.action_lag > 0.0 or args_cli.action_lag_drift:
                if args_cli.action_lag_drift:
                    elapsed_s = sim_step * env.unwrapped.step_dt
                    current_action_lag, current_phase = action_lag_schedule(elapsed_s)
                    if current_phase != lag_phase:
                        lag_phase = current_phase
                        print(f"[STRESS] t={elapsed_s:.2f} s phase={lag_phase} lag={current_action_lag:.4f}")
                if previous_executed_actions is None:
                    previous_executed_actions = actions.clone()
                actions = (1.0 - current_action_lag) * actions + current_action_lag * previous_executed_actions
                previous_executed_actions = actions.clone()
            # env stepping
            executed_actions = actions.clone() if args_cli.record_trajectories else None
            obs, rewards, dones, extras = env.step(actions)
            sim_step += 1

            if args_cli.goal_batch > 0:
                batch_returns[active_before_step] += rewards[active_before_step]

            if args_cli.record_trajectories:
                selected = active_before_step
                selected_env_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
                time_outs_step = extras.get("time_outs")
                if time_outs_step is None:
                    time_outs_step = torch.zeros_like(dones, dtype=torch.bool)
                robot = env.unwrapped.scene["robot"]
                velocity_error = torch.cat(
                    (
                        robot.data.root_lin_vel_b[:, :2] - command_term.vel_command_b[:, :2],
                        (robot.data.root_ang_vel_b[:, 2] - command_term.vel_command_b[:, 2]).unsqueeze(-1),
                    ),
                    dim=1,
                )
                lag_values = torch.full(
                    (args_cli.goal_batch,), current_action_lag, device=env.unwrapped.device, dtype=torch.float32
                )
                motor_values = torch.full_like(lag_values, args_cli.motor_strength)
                payload_values = torch.full_like(lag_values, args_cli.payload_kg)
                friction_value = args_cli.friction if args_cli.friction is not None else float("nan")
                friction_values = torch.full_like(lag_values, friction_value)
                step_values = torch.full(
                    (args_cli.goal_batch,), goal_step, device=env.unwrapped.device, dtype=torch.long
                )
                values = {
                    "states": step_states[selected],
                    "next_states": obs[selected],
                    "commands": command_term.vel_command_b[selected],
                    "policy_actions": policy_actions[selected],
                    "executed_actions": executed_actions[selected],
                    "rewards": rewards[selected],
                    "dones": dones[selected].bool(),
                    "time_outs": time_outs_step[selected].bool(),
                    "env_ids": selected_env_ids,
                    "episode_steps": step_values[selected],
                    "goal_relative": batch_target_relative.to(env.unwrapped.device)[selected],
                    "controller_modes": batch_mode_ids[selected],
                    "tracking_errors": velocity_error[selected],
                    "action_lags": lag_values[selected],
                    "motor_strengths": motor_values[selected],
                    "payload_kg": payload_values[selected],
                    "frictions": friction_values[selected],
                }
                for key, value in values.items():
                    trajectory_chunks[key].append(value.detach().cpu())

        if args_cli.primitive_test:
            robot = env.unwrapped.scene["robot"]
            actual_xy = robot.data.root_lin_vel_b[:, :2]
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            target = command_term.vel_command_b
            primitive_error_xy += torch.linalg.vector_norm(actual_xy - target[:, :2], dim=1).mean().item()
            primitive_error_yaw += torch.abs(actual_yaw - target[:, 2]).mean().item()
            time_outs = extras.get("time_outs")
            if time_outs is None:
                primitive_falls += int(dones.sum().item())
            else:
                primitive_falls += int(torch.logical_and(dones, torch.logical_not(time_outs)).sum().item())
            primitive_step += 1

            if primitive_step >= primitive_steps:
                print(
                    f"[PRIMITIVE] {primitive_name:>14s} command={primitive_command} "
                    f"mean_xy_error={primitive_error_xy / primitive_step:.3f} m/s "
                    f"mean_yaw_error={primitive_error_yaw / primitive_step:.3f} rad/s "
                    f"falls={primitive_falls}"
                )
                primitive_index += 1
                if primitive_index == len(PRIMITIVE_COMMANDS):
                    print("[INFO] Primitive test completed.")
                    break
                primitive_step = 0
                primitive_error_xy = 0.0
                primitive_error_yaw = 0.0
                primitive_falls = 0
        elif args_cli.goal_test:
            goal_step += 1
            time_outs = extras.get("time_outs")
            if time_outs is None:
                fell = bool(dones[0].item())
            else:
                fell = bool(torch.logical_and(dones, torch.logical_not(time_outs))[0].item())

            if fell:
                print(f"[GOAL] FAILED: robot fell after {goal_step * env.unwrapped.step_dt:.2f} s")
                break

            goal_within_tolerance = (
                goal_info["state"] == GoalController.HOLD
                and goal_info["distance"] <= goal_controller.cfg.position_tolerance
                and abs(goal_info["final_yaw_error"]) <= goal_controller.cfg.yaw_tolerance
            )
            if goal_within_tolerance:
                goal_hold_steps += 1
            else:
                goal_hold_steps = 0

            if goal_hold_steps * env.unwrapped.step_dt >= args_cli.goal_hold_time:
                print(
                    f"[GOAL] SUCCESS: time={goal_step * env.unwrapped.step_dt:.2f} s "
                    f"position_error={goal_info['distance']:.3f} m "
                    f"yaw_error={abs(goal_info['final_yaw_error']):.3f} rad"
                )
                break

            if goal_step * env.unwrapped.step_dt >= args_cli.goal_timeout:
                print(
                    f"[GOAL] TIMEOUT: time={args_cli.goal_timeout:.2f} s "
                    f"position_error={goal_info['distance']:.3f} m "
                    f"yaw_error={abs(goal_info['final_yaw_error']):.3f} rad state={goal_info['state']}"
                )
                break
        elif args_cli.goal_batch > 0:
            goal_step += 1
            robot = env.unwrapped.scene["robot"]
            positions = robot.data.root_pos_w
            _, _, yaws = euler_xyz_from_quat(robot.data.root_quat_w)
            time_outs = extras.get("time_outs")
            if time_outs is None:
                falls = dones
            else:
                falls = torch.logical_and(dones, torch.logical_not(time_outs))

            for index, controller in enumerate(batch_controllers):
                if not batch_active[index]:
                    continue
                distance, heading_error, final_yaw_error = controller.errors(
                    positions[index, 0].item(), positions[index, 1].item(), yaws[index].item()
                )
                if falls[index]:
                    batch_results[index] = {
                        "result": "fall",
                        "time": goal_step * env.unwrapped.step_dt,
                        "lag_at_result": current_action_lag,
                        "lag_phase_at_result": lag_phase or "fixed",
                        "position_error": distance,
                        "yaw_error": abs(final_yaw_error),
                        "episode_return": batch_returns[index].item(),
                        "final_state": controller.state,
                    }
                    batch_active[index] = False
                    continue

                goal_within_tolerance = (
                    controller.state == GoalController.HOLD
                    and distance <= controller.cfg.position_tolerance
                    and abs(final_yaw_error) <= controller.cfg.yaw_tolerance
                )
                if goal_within_tolerance:
                    batch_hold_steps[index] += 1
                else:
                    batch_hold_steps[index] = 0

                if batch_hold_steps[index].item() * env.unwrapped.step_dt >= args_cli.goal_hold_time:
                    batch_results[index] = {
                        "result": "success",
                        "time": goal_step * env.unwrapped.step_dt,
                        "lag_at_result": current_action_lag,
                        "lag_phase_at_result": lag_phase or "fixed",
                        "position_error": distance,
                        "yaw_error": abs(final_yaw_error),
                        "episode_return": batch_returns[index].item(),
                        "final_state": controller.state,
                    }
                    batch_active[index] = False

            if not batch_active.any() or goal_step * env.unwrapped.step_dt >= args_cli.goal_timeout:
                for index, controller in enumerate(batch_controllers):
                    if batch_results[index] is not None:
                        continue
                    distance, _, final_yaw_error = controller.errors(
                        positions[index, 0].item(), positions[index, 1].item(), yaws[index].item()
                    )
                    batch_results[index] = {
                        "result": "timeout",
                        "time": args_cli.goal_timeout,
                        "lag_at_result": current_action_lag,
                        "lag_phase_at_result": lag_phase or "fixed",
                        "position_error": distance,
                        "yaw_error": abs(final_yaw_error),
                        "episode_return": batch_returns[index].item(),
                        "final_state": controller.state,
                    }
                    batch_active[index] = False

                successes = [result for result in batch_results if result["result"] == "success"]
                falls_count = sum(result["result"] == "fall" for result in batch_results)
                timeouts_count = sum(result["result"] == "timeout" for result in batch_results)
                success_rate = len(successes) / args_cli.goal_batch
                mean_time = sum(result["time"] for result in successes) / len(successes) if successes else float("nan")
                mean_position_error = (
                    sum(result["position_error"] for result in successes) / len(successes) if successes else float("nan")
                )
                mean_yaw_error = (
                    sum(result["yaw_error"] for result in successes) / len(successes) if successes else float("nan")
                )

                results_path = args_cli.goal_results
                if results_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = os.path.join("outputs", "goal_eval", f"goal_eval_{timestamp}.csv")
                results_path = os.path.abspath(results_path)
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w", newline="", encoding="utf-8") as file:
                    fieldnames = [
                        "goal_id",
                        "motor_strength",
                        "payload_kg",
                        "friction",
                        "action_delay_steps",
                        "action_delay_ms",
                        "action_lag",
                        "action_lag_mode",
                        "lag_drift_start",
                        "lag_drift_peak",
                        "relative_x",
                        "relative_y",
                        "relative_yaw",
                        "distance",
                        "result",
                        "time",
                        "lag_at_result",
                        "lag_phase_at_result",
                        "position_error",
                        "yaw_error",
                        "episode_return",
                        "final_state",
                    ]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    for index, result in enumerate(batch_results):
                        relative_goal = batch_target_relative[index]
                        writer.writerow(
                            {
                                "goal_id": index,
                                "motor_strength": args_cli.motor_strength,
                                "payload_kg": args_cli.payload_kg,
                                "friction": args_cli.friction if args_cli.friction is not None else "",
                                "action_delay_steps": args_cli.action_delay_steps,
                                "action_delay_ms": 1000.0 * args_cli.action_delay_steps * env.unwrapped.step_dt,
                                "action_lag": args_cli.action_lag,
                                "action_lag_mode": "drift" if args_cli.action_lag_drift else "fixed",
                                "lag_drift_start": args_cli.lag_drift_start if args_cli.action_lag_drift else "",
                                "lag_drift_peak": args_cli.lag_drift_peak if args_cli.action_lag_drift else "",
                                "relative_x": relative_goal[0].item(),
                                "relative_y": relative_goal[1].item(),
                                "relative_yaw": relative_goal[2].item(),
                                "distance": math.hypot(relative_goal[0].item(), relative_goal[1].item()),
                                **result,
                            }
                        )

                print(
                    f"[GOAL-BATCH] COMPLETE: success={len(successes)}/{args_cli.goal_batch} "
                    f"({100.0 * success_rate:.1f}%) falls={falls_count} timeouts={timeouts_count} "
                    f"motor_strength={args_cli.motor_strength:.3f} "
                    f"action_delay={args_cli.action_delay_steps} steps "
                    f"action_lag={args_cli.action_lag:.3f} "
                    f"action_lag_mode={'drift' if args_cli.action_lag_drift else 'fixed'} "
                    f"mean_success_time={mean_time:.2f} s mean_position_error={mean_position_error:.3f} m "
                    f"mean_yaw_error={mean_yaw_error:.3f} rad"
                )
                print(f"[GOAL-BATCH] Results saved to: {results_path}")
                break
        # Yield briefly to the desktop compositor during interactive playback.
        # Without this, the unthrottled simulation loop can make GNOME mark the
        # Isaac Sim window as unresponsive even though simulation is progressing.
        if not args_cli.headless:
            time.sleep(0.01)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if args_cli.record_trajectories:
        trajectory_data = {key: torch.cat(chunks, dim=0) for key, chunks in trajectory_chunks.items()}
        trajectory_ends = torch.zeros_like(trajectory_data["dones"], dtype=torch.bool)
        for env_id in range(args_cli.goal_batch):
            indices = torch.nonzero(trajectory_data["env_ids"] == env_id, as_tuple=False).squeeze(-1)
            if indices.numel() > 0:
                trajectory_ends[indices[-1]] = True
        trajectory_data["trajectory_ends"] = trajectory_ends
        # Match T2MIR's replay-buffer convention: time-limit truncations keep
        # the bootstrap mask, while true terminations clear it.
        trajectory_data["masks"] = torch.logical_not(
            torch.logical_and(trajectory_data["dones"], torch.logical_not(trajectory_data["time_outs"]))
        )
        trajectory_data["task_ids"] = torch.full_like(
            trajectory_data["env_ids"], args_cli.trajectory_task_id, dtype=torch.long
        )
        trajectory_data["metadata"] = {
            "format_version": 2,
            "task": args_cli.task,
            "checkpoint": resume_path,
            "policy_role": args_cli.trajectory_policy_role,
            "policy_action_mode": policy_action_mode,
            "policy_action_seed": policy_action_seed,
            "task_id": args_cli.trajectory_task_id,
            "goal_seed": args_cli.goal_seed,
            "num_envs": args_cli.goal_batch,
            "step_dt": env.unwrapped.step_dt,
            "observation_dim": trajectory_data["states"].shape[1],
            "action_dim": trajectory_data["policy_actions"].shape[1],
            "controller_mode_names": {value: key for key, value in CONTROLLER_MODE_TO_ID.items()},
            "action_lag": args_cli.action_lag,
            "motor_strength": args_cli.motor_strength,
            "payload_kg": args_cli.payload_kg,
            "friction": args_cli.friction,
            "deterministic_dynamics": args_cli.deterministic_dynamics,
            "action_lag_drift": args_cli.action_lag_drift,
            "lag_drift_start": args_cli.lag_drift_start,
            "lag_drift_peak": args_cli.lag_drift_peak,
        }
        trajectory_path = args_cli.trajectory_output
        if trajectory_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = os.path.join("outputs", "trajectories", f"g1_trajectory_{timestamp}.pt")
        trajectory_path = os.path.abspath(trajectory_path)
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
        torch.save(trajectory_data, trajectory_path)
        print(
            f"[TRAJECTORY] Saved {trajectory_data['states'].shape[0]} transitions from "
            f"{args_cli.goal_batch} trajectories to: {trajectory_path}"
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
