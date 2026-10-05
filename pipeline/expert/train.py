# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
from pipeline.expert import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help="Optional checkpoint interval in PPO learning iterations.",
)
parser.add_argument(
    "--action_lag",
    type=float,
    default=0.0,
    help="Fixed first-order action lag in [0, 1) used to train a dynamics-specialist policy.",
)
parser.add_argument("--motor_strength", type=float, default=1.0,
                    help="Scale all actuator effort limits for a fixed dynamics task.")
parser.add_argument("--payload_kg", type=float, default=0.0,
                    help="Fixed mass added to torso_link for a dynamics task.")
parser.add_argument("--friction", type=float, default=None,
                    help="Fixed static/dynamic robot friction coefficient.")
parser.add_argument("--deterministic_dynamics", action="store_true",
                    help="Disable background actuator-gain and joint-parameter randomization.")
parser.add_argument("--learning_rate", type=float, default=None, help="Optional PPO learning-rate override.")
parser.add_argument("--entropy_coef", type=float, default=None, help="Optional PPO entropy-coefficient override.")
parser.add_argument(
    "--load_optimizer",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Restore optimizer state when resuming; disable for conservative specialist fine-tuning.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import robot_lab.tasks  # noqa: F401
from robot_lab.third_party.rsl_rl.runners import OnPolicyRunner
from pipeline.protocols.dynamics_profile import configure_env_dynamics


class RobotLabRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Compatibility wrapper for the RSL-RL 1.0 interface.

    Isaac Lab 1.2's wrapper predates the abstract
    ``get_privileged_observations`` method added to the installed RSL-RL
    package. Robot Lab's runner consumes privileged observations from the
    extras dictionary, so exposing the same tensor here preserves both APIs.
    """

    def get_privileged_observations(self) -> torch.Tensor | None:
        _, extras = self.get_observations()
        return extras["observations"].get("critic")


class ActionLagEnvWrapper(gym.Wrapper):
    """Apply a fixed first-order lag to actions before the simulator step."""

    def __init__(self, env: gym.Env, lag: float):
        super().__init__(env)
        self.lag = lag
        self._previous_executed_actions = None
        self._reset_mask = None

    def reset(self, **kwargs):
        self._previous_executed_actions = None
        self._reset_mask = None
        return self.env.reset(**kwargs)

    def step(self, actions: torch.Tensor):
        if self._previous_executed_actions is None:
            self._previous_executed_actions = actions.clone()
        elif self._reset_mask is not None and self._reset_mask.any():
            # Isaac Lab auto-resets completed vector environments inside step.
            # Start their new episodes without leaking the previous episode's
            # final action into the lag state.
            self._previous_executed_actions[self._reset_mask] = actions[self._reset_mask]

        executed_actions = (1.0 - self.lag) * actions + self.lag * self._previous_executed_actions
        self._previous_executed_actions = executed_actions.clone()
        result = self.env.step(executed_actions)

        if len(result) == 5:
            _, _, terminated, truncated, _ = result
            self._reset_mask = torch.logical_or(terminated, truncated).bool()
        else:
            self._reset_mask = result[2].bool()
        return result

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.save_interval is not None:
        if args_cli.save_interval <= 0:
            raise ValueError("--save_interval must be greater than zero")
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.learning_rate is not None:
        if args_cli.learning_rate <= 0.0:
            raise ValueError("--learning_rate must be greater than zero")
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.entropy_coef is not None:
        if args_cli.entropy_coef < 0.0:
            raise ValueError("--entropy_coef must be non-negative")
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if not 0.0 <= args_cli.action_lag < 1.0:
        raise ValueError("--action_lag must be in [0, 1)")
    if args_cli.motor_strength <= 0.0:
        raise ValueError("--motor_strength must be positive")
    if args_cli.payload_kg < 0.0:
        raise ValueError("--payload_kg must be non-negative")
    if args_cli.friction is not None and args_cli.friction <= 0.0:
        raise ValueError("--friction must be positive")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    dynamics = configure_env_dynamics(
        env_cfg,
        motor_strength=args_cli.motor_strength,
        payload_kg=args_cli.payload_kg,
        friction=args_cli.friction,
        deterministic=args_cli.deterministic_dynamics,
    )
    dynamics["action_lag"] = args_cli.action_lag
    print(f"[TRAIN] Fixed dynamics profile: {dynamics}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.action_lag > 0.0:
        env = ActionLagEnvWrapper(env, args_cli.action_lag)
        print(f"[TRAIN] Fixed first-order action lag={args_cli.action_lag:.4f}")

    # wrap around environment for rsl-rl
    env = RobotLabRslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=args_cli.load_optimizer)
        print(f"[INFO]: Optimizer state restored: {args_cli.load_optimizer}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "dynamics.yaml"),
        {
            "action_lag": args_cli.action_lag,
            "model": "first_order_action_lag",
            **dynamics,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "load_optimizer": args_cli.load_optimizer,
        },
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
