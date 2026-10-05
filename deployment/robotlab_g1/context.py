"""CPU-only episode context lifecycle for T2MIR deployment.

The buffer intentionally knows nothing about PyTorch, ROS 2 or MuJoCo.  It
freezes a prompt when an episode starts and only publishes completed episodes
to later prompts, preventing current-episode data or reset transitions from
leaking into the model context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .contract import DEFAULT_CONTRACT, DeploymentContract


@dataclass(frozen=True)
class ContextProtocol:
    name: str
    prompt_episode_horizon: int
    prompt_horizon: int
    max_episode_steps: int = 64
    window: Literal["first", "last"] = "last"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("context protocol name cannot be empty")
        if min(self.prompt_episode_horizon, self.prompt_horizon, self.max_episode_steps) <= 0:
            raise ValueError("context horizons must be positive")
        if self.prompt_horizon > self.prompt_episode_horizon * self.max_episode_steps:
            raise ValueError("prompt_horizon exceeds the configured episode capacity")
        if self.window not in {"first", "last"}:
            raise ValueError("window must be 'first' or 'last'")

    @classmethod
    def official(cls) -> "ContextProtocol":
        return cls(
            name="official_previous_episode_64",
            prompt_episode_horizon=1,
            prompt_horizon=64,
            max_episode_steps=64,
            window="last",
        )

    @classmethod
    def legacy(cls) -> "ContextProtocol":
        return cls(
            name="legacy_four_episodes_256",
            prompt_episode_horizon=4,
            prompt_horizon=256,
            max_episode_steps=64,
            window="last",
        )


OFFICIAL_CONTEXT_PROTOCOL = ContextProtocol.official()
LEGACY_CONTEXT_PROTOCOL = ContextProtocol.legacy()


@dataclass(frozen=True)
class ContextWindow:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    attention_mask: np.ndarray
    episode_indices: tuple[int, ...]
    episode_lengths: tuple[int, ...]

    @property
    def length(self) -> int:
        return int(self.states.shape[0])

    def as_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays with the leading batch dimension expected by T2MIR."""

        return (
            self.states[None, ...],
            self.actions[None, ...],
            self.rewards[None, ...],
            self.attention_mask[None, ...],
        )


@dataclass(frozen=True)
class _Episode:
    index: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    @property
    def length(self) -> int:
        return int(self.states.shape[0])


def _empty_window(contract: DeploymentContract) -> ContextWindow:
    return ContextWindow(
        states=np.empty((0, contract.state_dim), dtype=np.float32),
        actions=np.empty((0, contract.action_dim), dtype=np.float32),
        rewards=np.empty((0, 1), dtype=np.float32),
        attention_mask=np.empty((0,), dtype=np.int64),
        episode_indices=(),
        episode_lengths=(),
    )


class EpisodeContextBuffer:
    """Maintain the immutable prompt for one online environment."""

    def __init__(
        self,
        protocol: ContextProtocol = OFFICIAL_CONTEXT_PROTOCOL,
        contract: DeploymentContract = DEFAULT_CONTRACT,
    ) -> None:
        self.protocol = protocol
        self.contract = contract
        self._history: deque[_Episode] = deque(maxlen=protocol.prompt_episode_horizon)
        self._current_index: int | None = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.float32] = []
        self._frozen_prompt = _empty_window(contract)

    @property
    def episode_active(self) -> bool:
        return self._current_index is not None

    @property
    def completed_episode_indices(self) -> tuple[int, ...]:
        return tuple(episode.index for episode in self._history)

    def start_episode(self, episode_index: int) -> ContextWindow:
        if self.episode_active:
            raise RuntimeError("finish_episode must be called before starting another episode")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self._history and episode_index <= self._history[-1].index:
            raise ValueError("episode_index must increase monotonically")
        self._current_index = int(episode_index)
        self._states = []
        self._actions = []
        self._rewards = []
        self._frozen_prompt = self._build_prompt()
        return self.prompt()

    def append(
        self,
        state: np.ndarray | Sequence[float],
        policy_action: np.ndarray | Sequence[float],
        reward: float,
    ) -> None:
        if not self.episode_active:
            raise RuntimeError("start_episode must be called before append")
        state_array = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(policy_action, dtype=np.float32)
        reward_value = np.float32(reward)
        if state_array.shape != (self.contract.state_dim,):
            raise ValueError(
                f"state shape must be {(self.contract.state_dim,)}, got {state_array.shape}"
            )
        if action_array.shape != (self.contract.action_dim,):
            raise ValueError(
                f"policy_action shape must be {(self.contract.action_dim,)}, got {action_array.shape}"
            )
        if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
            raise ValueError("context state/action contains NaN or Inf")
        if not np.isfinite(reward_value):
            raise ValueError("context reward contains NaN or Inf")
        self._states.append(state_array.copy())
        self._actions.append(action_array.copy())
        self._rewards.append(reward_value)

    def finish_episode(self) -> ContextWindow:
        if self._current_index is None:
            raise RuntimeError("no active episode to finish")
        if self._states:
            states = np.stack(self._states).astype(np.float32, copy=False)
            actions = np.stack(self._actions).astype(np.float32, copy=False)
            rewards = np.asarray(self._rewards, dtype=np.float32).reshape(-1, 1)
        else:
            states = np.empty((0, self.contract.state_dim), dtype=np.float32)
            actions = np.empty((0, self.contract.action_dim), dtype=np.float32)
            rewards = np.empty((0, 1), dtype=np.float32)
        self._history.append(_Episode(self._current_index, states, actions, rewards))
        self._current_index = None
        self._states = []
        self._actions = []
        self._rewards = []
        return self.prompt()

    def clear(self) -> None:
        """Clear all context and any partially recorded episode."""

        self._history.clear()
        self._current_index = None
        self._states = []
        self._actions = []
        self._rewards = []
        self._frozen_prompt = _empty_window(self.contract)

    def prompt(self) -> ContextWindow:
        """Return a defensive copy of the prompt frozen at episode start."""

        source = self._frozen_prompt
        return ContextWindow(
            states=source.states.copy(),
            actions=source.actions.copy(),
            rewards=source.rewards.copy(),
            attention_mask=source.attention_mask.copy(),
            episode_indices=source.episode_indices,
            episode_lengths=source.episode_lengths,
        )

    def _build_prompt(self) -> ContextWindow:
        if not self._history:
            return _empty_window(self.contract)

        selected: list[_Episode] = []
        for episode in self._history:
            if self.protocol.window == "last":
                span = slice(-self.protocol.max_episode_steps, None)
            else:
                span = slice(0, self.protocol.max_episode_steps)
            selected.append(
                _Episode(
                    episode.index,
                    episode.states[span],
                    episode.actions[span],
                    episode.rewards[span],
                )
            )

        states = np.concatenate([episode.states for episode in selected], axis=0)
        actions = np.concatenate([episode.actions for episode in selected], axis=0)
        rewards = np.concatenate([episode.rewards for episode in selected], axis=0)
        if states.shape[0] > self.protocol.prompt_horizon:
            if self.protocol.window == "last":
                span = slice(-self.protocol.prompt_horizon, None)
            else:
                span = slice(0, self.protocol.prompt_horizon)
            states = states[span]
            actions = actions[span]
            rewards = rewards[span]

        return ContextWindow(
            states=states.astype(np.float32, copy=True),
            actions=actions.astype(np.float32, copy=True),
            rewards=rewards.astype(np.float32, copy=True),
            attention_mask=np.ones((states.shape[0],), dtype=np.int64),
            episode_indices=tuple(episode.index for episode in selected),
            episode_lengths=tuple(episode.length for episode in selected),
        )


__all__ = [
    "ContextProtocol",
    "ContextWindow",
    "EpisodeContextBuffer",
    "LEGACY_CONTEXT_PROTOCOL",
    "OFFICIAL_CONTEXT_PROTOCOL",
]
