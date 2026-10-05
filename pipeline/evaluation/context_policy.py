"""Inference adapters for RobotLab G1 offline policies.

The adapters expose the same ``policy(observations) -> actions`` interface used
by RSL-RL.  T2MIR uses a fixed, task-matched prompt sampled from the formal
dataset; observations supplied by Isaac Sim are used as query states.
"""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ActorMLP(nn.Module):
    """Architecture used by the multidynamics diagnostic baselines."""

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class QueryOnlyPolicy:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_mean = torch.as_tensor(checkpoint["state_mean"], device=device)
        state_std = torch.as_tensor(checkpoint["state_std"], device=device)
        state_dim = state_mean.numel()
        action_dim = checkpoint["model"]["net.6.weight"].shape[0]
        self.model = ActorMLP(state_dim, action_dim).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.state_mean = state_mean
        self.state_std = state_std

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model((observations - self.state_mean) / self.state_std)


class OracleDynamicsPolicy(QueryOnlyPolicy):
    def __init__(self, checkpoint_path: Path, device: torch.device, parameters: list[float]):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_mean = torch.as_tensor(checkpoint["state_mean"], device=device)
        state_std = torch.as_tensor(checkpoint["state_std"], device=device)
        state_dim = state_mean.numel()
        action_dim = checkpoint["model"]["net.6.weight"].shape[0]
        self.model = ActorMLP(state_dim + 4, action_dim).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.state_mean = state_mean
        self.state_std = state_std
        parameter_mean = torch.as_tensor(checkpoint["parameter_mean"], device=device)
        parameter_std = torch.as_tensor(checkpoint["parameter_std"], device=device)
        value = torch.tensor(parameters, dtype=torch.float32, device=device)
        self.parameters = (value - parameter_mean) / parameter_std

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        states = (observations - self.state_mean) / self.state_std
        parameters = self.parameters.unsqueeze(0).expand(len(observations), -1)
        return self.model(torch.cat((states, parameters), dim=1))


class T2MIRPolicy:
    def __init__(
        self,
        checkpoint_path: Path,
        dataset_dir: Path,
        prompt_task_id: int,
        prompt_seed: int,
        device: torch.device,
        t2mir_root: Path,
    ):
        if str(t2mir_root) not in sys.path:
            sys.path.insert(0, str(t2mir_root))
        from algorithms.policy import DPTTransformerMOE

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        self.model = DPTTransformerMOE(
            checkpoint["state_dim"], checkpoint["action_dim"], config,
            action_tanh=False, discrete_environment=False,
        ).to(device)
        self.model.load_state_dict(checkpoint["policy"])
        self.model.eval()
        self.state_mean = torch.as_tensor(checkpoint["state_mean"], device=device)
        self.state_std = torch.as_tensor(checkpoint["state_std"], device=device)

        dataset_path = dataset_dir / f"dataset_task_{prompt_task_id}.pkl"
        with dataset_path.open("rb") as file:
            dataset = pickle.load(file)
        episode_length = int(config["max_episode_steps"])
        episode_count = len(dataset["states"]) // episode_length
        prompt_episodes = int(config["prompt_episode_horizon"])
        if episode_count < prompt_episodes:
            raise ValueError(f"{dataset_path} has only {episode_count} episodes")
        rng = np.random.default_rng(prompt_seed)
        episode_ids = np.sort(rng.choice(episode_count, prompt_episodes, replace=False))
        indices = np.concatenate([
            np.arange(index * episode_length, (index + 1) * episode_length) for index in episode_ids
        ])
        states = torch.from_numpy(np.asarray(dataset["states"][indices], np.float32)).to(device)
        actions = torch.from_numpy(np.asarray(dataset["actions"][indices], np.float32)).to(device)
        rewards = torch.from_numpy(np.asarray(dataset["rewards"][indices], np.float32)).to(device).unsqueeze(-1)
        self.prompt_states = ((states - self.state_mean) / self.state_std).unsqueeze(0)
        self.prompt_actions = actions.unsqueeze(0)
        self.prompt_rewards = rewards.unsqueeze(0)
        self.prompt_episode_ids = episode_ids.tolist()

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        batch = len(observations)
        query = ((observations - self.state_mean) / self.state_std).unsqueeze(1)
        return self.model(
            self.prompt_states.expand(batch, -1, -1),
            self.prompt_actions.expand(batch, -1, -1),
            self.prompt_rewards.expand(batch, -1, -1),
            query,
            eval=True,
        )
