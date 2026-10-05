"""Source-faithful online prompt handling for RobotLab T2MIR evaluation.

This module intentionally has no Isaac Sim dependency.  It owns the protocol
boundary between simulator episodes and a DPT policy:

* episode zero uses an empty prompt;
* the source-faithful default keeps the prompt fixed for the complete episode;
* every vector environment records its own ``(state, policy_action, reward)``
  trajectory;
* a reset is a hard trajectory boundary; and
* only the immediately preceding episode is eligible for the next prompt.

RobotLab goal episodes are much longer than the 64-step episodes used by the
formal DPT dataset.  For an explicitly requested diagnostic/streaming mode the
same buffer can therefore promote a complete 64-step block from the current
episode.  The default remains the previous-episode protocol and existing
formal results are not silently reinterpreted.

``policy_action`` means the action emitted by the policy *before* RobotLab's
external action-delay/first-order-lag transform.  This is the same action
semantic used by the formal prompt dataset.
"""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


FORMAT_VERSION = 1
ACTION_SEMANTICS = "policy_action_before_external_dynamics_transform"
SUPPORTED_SUPERVISION_MODES = frozenset({"all_tokens", "final_token"})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def routing_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the routing signature stored by formal RobotLab checkpoints.

    Keeping this small implementation next to inference avoids importing the
    training entry point (and its dataset/optimizer dependencies) into Isaac.
    """

    moe = config["moe_config"]

    def branch(prefix: str, experts_key: str, selects_key: str) -> dict[str, Any]:
        mode = str(moe.get(f"{prefix}_routing_mode", moe.get("routing_mode", "topk"))).lower()
        if mode not in {"topk", "topp"}:
            raise ValueError(f"unsupported {prefix} routing mode: {mode!r}")
        num_experts = int(moe[experts_key])
        fixed_top_k = int(moe[selects_key])
        threshold = float(moe.get(f"{prefix}_top_p_threshold", moe.get("top_p_threshold", 0.4)))
        raw_max = moe.get(f"{prefix}_top_p_max_selects", moe.get("top_p_max_selects", num_experts))
        max_selects = num_experts if raw_max is None else int(raw_max)
        if not 1 <= fixed_top_k <= num_experts:
            raise ValueError(f"invalid {prefix} fixed Top-K: {fixed_top_k}/{num_experts}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"invalid {prefix} Top-p threshold: {threshold}")
        if not 1 <= max_selects <= num_experts:
            raise ValueError(f"invalid {prefix} Top-p maximum: {max_selects}/{num_experts}")
        return {
            "mode": mode,
            "num_experts": num_experts,
            "fixed_top_k": fixed_top_k,
            "top_p_threshold": threshold,
            "top_p_max_selects": max_selects,
        }

    return {
        "schema_version": 1,
        "token": branch("token", "num_experts", "num_selects"),
        "task": branch("task", "num_experts_contrastive", "num_selects_contrastive"),
        "task_hard_router": bool(moe.get("task_hard_router", False)),
    }


def load_expected_signature(value: Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Load a routing signature JSON (or accept an already decoded mapping)."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    # Accept either a bare signature or a model-manifest record.
    if "routing_signature" in payload:
        payload = payload["routing_signature"]
    if not isinstance(payload, dict):
        raise ValueError("expected routing signature must decode to a JSON object")
    return payload


def validate_checkpoint_contract(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
    expected_routing: Mapping[str, Any] | None = None,
    expected_prompt_horizon: int | None = 64,
    expected_supervision_modes: Sequence[str] = ("all_tokens",),
) -> dict[str, Any]:
    """Validate checkpoint identity, dimensions and routing semantics."""

    required = {"policy", "config", "state_mean", "state_std", "state_dim", "action_dim"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    digest = sha256_file(checkpoint_path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest} != {expected_sha256}")

    config = checkpoint["config"]
    derived_routing = routing_signature(config)
    saved_routing = checkpoint.get("routing_signature", derived_routing)
    if saved_routing != derived_routing:
        raise ValueError("checkpoint routing_signature disagrees with checkpoint config")
    if expected_routing is not None and dict(expected_routing) != saved_routing:
        raise ValueError(
            "checkpoint routing signature does not match the requested variant:\n"
            f"checkpoint={json.dumps(saved_routing, sort_keys=True)}\n"
            f"expected={json.dumps(dict(expected_routing), sort_keys=True)}"
        )

    prompt_horizon = int(config["prompt_horizon"])
    if expected_prompt_horizon is not None and prompt_horizon != expected_prompt_horizon:
        raise ValueError(
            f"checkpoint prompt_horizon={prompt_horizon}; expected {expected_prompt_horizon}"
        )
    if int(config.get("max_episode_steps", -1)) != prompt_horizon:
        raise ValueError("checkpoint max_episode_steps must equal prompt_horizon")
    if int(config.get("prompt_episode_horizon", -1)) != 1:
        raise ValueError("official RobotLab DPT evaluation requires prompt_episode_horizon=1")
    supervision_mode = str(config.get("supervision_mode", "all_tokens"))
    if supervision_mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError(f"unsupported checkpoint supervision_mode={supervision_mode!r}")
    if isinstance(expected_supervision_modes, str):
        expected_supervision_modes = (expected_supervision_modes,)
    expected_modes = tuple(str(value) for value in expected_supervision_modes)
    if not expected_modes:
        raise ValueError("expected_supervision_modes must not be empty")
    unsupported_expected = sorted(set(expected_modes) - SUPPORTED_SUPERVISION_MODES)
    if unsupported_expected:
        raise ValueError(
            f"unsupported expected supervision modes: {unsupported_expected}"
        )
    if supervision_mode not in expected_modes:
        raise ValueError(
            f"checkpoint supervision_mode={supervision_mode!r}; "
            f"expected one of {sorted(set(expected_modes))}"
        )
    if bool(config.get("action_tanh", False)):
        raise ValueError("RobotLab G1 formal checkpoints must use unbounded (non-tanh) actions")
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    if torch.as_tensor(checkpoint["state_mean"]).numel() != state_dim:
        raise ValueError("checkpoint state_mean does not match state_dim")
    if torch.as_tensor(checkpoint["state_std"]).numel() != state_dim:
        raise ValueError("checkpoint state_std does not match state_dim")
    state_mean = torch.as_tensor(checkpoint["state_mean"])
    state_std = torch.as_tensor(checkpoint["state_std"])
    if not torch.isfinite(state_mean).all() or not torch.isfinite(state_std).all():
        raise ValueError("checkpoint state normalization contains NaN or Inf")
    if not torch.all(state_std > 0):
        raise ValueError("checkpoint state_std must be strictly positive")

    return {
        "format_version": FORMAT_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_step": checkpoint.get("step"),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "prompt_horizon": prompt_horizon,
        "supervision_mode": supervision_mode,
        "routing_signature": saved_routing,
        # Load-balancing changes training semantics without changing the
        # inference graph.  Preserve it in deployment manifests so two
        # otherwise identical routing variants cannot be confused later.
        "balance_signature": checkpoint.get("balance_signature"),
    }


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One environment's transitions from exactly one simulator episode."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    controller_modes: torch.Tensor
    episode_index: int

    @property
    def length(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class PromptBatch:
    env_ids: tuple[int, ...]
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    attention_mask: torch.Tensor
    source_lengths: tuple[int, ...]
    source_episode_indices: tuple[int | None, ...]
    controller_mode_histograms: tuple[dict[int, int], ...]


class OnlineEpisodePromptBuffer:
    """Maintain independent previous-episode prompts for vector environments."""

    VALID_WINDOWS = {"last", "first"}
    VALID_UPDATE_MODES = {"episode", "block"}

    def __init__(
        self,
        num_envs: int,
        state_dim: int,
        action_dim: int,
        *,
        prompt_horizon: int = 64,
        window: str = "last",
        update_mode: str = "episode",
        update_interval: int | None = None,
    ) -> None:
        if min(num_envs, state_dim, action_dim, prompt_horizon) <= 0:
            raise ValueError("num_envs, dimensions and prompt_horizon must be positive")
        if window not in self.VALID_WINDOWS:
            raise ValueError(f"window must be one of {sorted(self.VALID_WINDOWS)}")
        if update_mode not in self.VALID_UPDATE_MODES:
            raise ValueError(
                f"update_mode must be one of {sorted(self.VALID_UPDATE_MODES)}"
            )
        if update_interval is None:
            update_interval = prompt_horizon
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prompt_horizon = prompt_horizon
        self.window = window
        self.update_mode = update_mode
        self.update_interval = update_interval
        self._previous: list[EpisodeTrajectory | None] = [None] * num_envs
        self._current: list[dict[str, list[torch.Tensor]]] | None = None
        self._current_episode_index: int | None = None

    @property
    def previous_trajectories(self) -> tuple[EpisodeTrajectory | None, ...]:
        return tuple(self._previous)

    def reset(self) -> None:
        """Discard all prompt history at an explicit external task boundary."""

        if self._current is not None:
            raise RuntimeError("cannot reset prompt history during an active episode")
        self._previous = [None] * self.num_envs
        self._current_episode_index = None

    def start_episode(self, episode_index: int) -> None:
        if self._current is not None:
            raise RuntimeError("cannot start a new episode before finish_episode")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        self._current_episode_index = episode_index
        self._current = [
            {"states": [], "actions": [], "rewards": [], "controller_modes": []}
            for _ in range(self.num_envs)
        ]

    def append(
        self,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
        rewards: torch.Tensor,
        active: torch.Tensor | Sequence[bool],
        controller_modes: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Append one pre-reset transition for each currently active env."""

        if self._current is None:
            raise RuntimeError("start_episode must be called before append")
        states = torch.as_tensor(states).detach().cpu()
        policy_actions = torch.as_tensor(policy_actions).detach().cpu()
        rewards = torch.as_tensor(rewards).detach().cpu().reshape(-1)
        active = torch.as_tensor(active, dtype=torch.bool).detach().cpu().reshape(-1)
        if controller_modes is None:
            controller_modes = torch.full((self.num_envs,), -1, dtype=torch.long)
        controller_modes = torch.as_tensor(
            controller_modes, dtype=torch.long
        ).detach().cpu().reshape(-1)
        if tuple(states.shape) != (self.num_envs, self.state_dim):
            raise ValueError(f"states shape {tuple(states.shape)} is not {(self.num_envs, self.state_dim)}")
        if tuple(policy_actions.shape) != (self.num_envs, self.action_dim):
            raise ValueError(
                f"policy_actions shape {tuple(policy_actions.shape)} is not "
                f"{(self.num_envs, self.action_dim)}"
            )
        if (
            tuple(rewards.shape) != (self.num_envs,)
            or tuple(active.shape) != (self.num_envs,)
            or tuple(controller_modes.shape) != (self.num_envs,)
        ):
            raise ValueError("rewards, active and controller_modes need one value per environment")
        if not torch.isfinite(states[active]).all() or not torch.isfinite(policy_actions[active]).all():
            raise ValueError("active trajectory contains non-finite state/action values")
        if not torch.isfinite(rewards[active]).all():
            raise ValueError("active trajectory contains non-finite rewards")
        for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
            self._current[env_id]["states"].append(states[env_id].clone())
            self._current[env_id]["actions"].append(policy_actions[env_id].clone())
            self._current[env_id]["rewards"].append(rewards[env_id].clone())
            self._current[env_id]["controller_modes"].append(controller_modes[env_id].clone())

    def _trajectory_from_values(
        self,
        values: dict[str, list[torch.Tensor]],
        episode_index: int,
    ) -> EpisodeTrajectory:
        """Materialize one environment trajectory without mutating the buffer."""

        if values["states"]:
            states = torch.stack(values["states"]).to(dtype=torch.float32)
            actions = torch.stack(values["actions"]).to(dtype=torch.float32)
            rewards = torch.stack(values["rewards"]).to(dtype=torch.float32).reshape(-1, 1)
            controller_modes = torch.stack(values["controller_modes"]).to(dtype=torch.long)
        else:
            states = torch.empty((0, self.state_dim), dtype=torch.float32)
            actions = torch.empty((0, self.action_dim), dtype=torch.float32)
            rewards = torch.empty((0, 1), dtype=torch.float32)
            controller_modes = torch.empty((0,), dtype=torch.long)
        return EpisodeTrajectory(
            states, actions, rewards, controller_modes, episode_index
        )

    def finish_episode(self) -> tuple[EpisodeTrajectory, ...]:
        """Seal all env trajectories; this is the only prompt update point."""

        if self._current is None or self._current_episode_index is None:
            raise RuntimeError("no active episode to finish")
        trajectories = [
            self._trajectory_from_values(values, self._current_episode_index)
            for values in self._current
        ]
        if self.update_mode == "episode":
            self._previous = list(trajectories)
        else:
            # A sub-horizon success/fall must not replace a previously valid
            # full prompt with a length never seen during formal training.
            self._previous = [
                trajectory
                if trajectory.length >= self.prompt_horizon or previous is None
                else previous
                for trajectory, previous in zip(trajectories, self._previous)
            ]
        self._current = None
        self._current_episode_index = None
        return tuple(trajectories)

    def _window(self, trajectory: EpisodeTrajectory) -> tuple[torch.Tensor, ...]:
        length = trajectory.length
        if length >= self.prompt_horizon:
            span = slice(-self.prompt_horizon, None) if self.window == "last" else slice(0, self.prompt_horizon)
            mask = torch.ones(self.prompt_horizon, dtype=torch.long)
            return (
                trajectory.states[span],
                trajectory.actions[span],
                trajectory.rewards[span],
                mask,
                trajectory.controller_modes[span],
            )
        # Do not invent transitions after early success/fall.  Exact-length
        # groups are evaluated separately, so no padding crosses this boundary.
        return (
            trajectory.states,
            trajectory.actions,
            trajectory.rewards,
            torch.ones(length, dtype=torch.long),
            trajectory.controller_modes,
        )

    def prompt_groups(
        self,
        device: torch.device | str = "cpu",
        *,
        current_if_complete: bool = False,
    ) -> tuple[PromptBatch, ...]:
        """Return exact-length prompt batches grouped across environments.

        Vector environments can finish at different times.  Grouping by exact
        source length preserves real 0..64-step prompts without padding while
        still batching environments that have compatible prompt shapes.
        """

        if self._current is not None:
            # Reading is valid during an episode: _previous is immutable until
            # finish_episode, which enforces source-faithful update timing.
            pass
        by_length: dict[int, list[tuple[int, EpisodeTrajectory]]] = {}
        for env_id, item in enumerate(self._previous):
            if current_if_complete:
                if self._current is None or self._current_episode_index is None:
                    raise RuntimeError(
                        "current_if_complete requires an active episode"
                    )
                current = self._trajectory_from_values(
                    self._current[env_id], self._current_episode_index
                )
                if current.length >= self.prompt_horizon:
                    item = current
            if item is None:
                item = EpisodeTrajectory(
                    torch.empty((0, self.state_dim), dtype=torch.float32),
                    torch.empty((0, self.action_dim), dtype=torch.float32),
                    torch.empty((0, 1), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.long),
                    episode_index=-1,
                )
            selected_length = min(item.length, self.prompt_horizon)
            by_length.setdefault(selected_length, []).append((env_id, item))

        groups = []
        for _, rows in sorted(by_length.items()):
            windows = [self._window(item) for _, item in rows]
            states, actions, rewards, masks, modes = zip(*windows)
            mode_histograms = []
            for values in modes:
                unique, counts = torch.unique(values, return_counts=True)
                mode_histograms.append(
                    {int(key): int(value) for key, value in zip(unique.tolist(), counts.tolist())}
                )
            groups.append(
                PromptBatch(
                    env_ids=tuple(env_id for env_id, _ in rows),
                    states=torch.stack(states).to(device=device),
                    actions=torch.stack(actions).to(device=device),
                    rewards=torch.stack(rewards).to(device=device),
                    attention_mask=torch.stack(masks).to(device=device),
                    source_lengths=tuple(item.length for _, item in rows),
                    source_episode_indices=tuple(
                        None if item.episode_index < 0 else item.episode_index for _, item in rows
                    ),
                    controller_mode_histograms=tuple(mode_histograms),
                )
            )
        return tuple(groups)

    def prompt_batch(self, device: torch.device | str = "cpu") -> PromptBatch:
        """Return one batch when all envs have the same selected prompt length."""

        groups = self.prompt_groups(device)
        if len(groups) != 1:
            lengths = [group.states.shape[1] for group in groups]
            raise ValueError(
                f"environments have different prompt lengths {lengths}; use prompt_groups()"
            )
        return groups[0]

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "protocol": (
                "previous-own-episode-fixed-prompt"
                if self.update_mode == "episode"
                else "causal-own-rollout-block-prompt"
            ),
            "prompt_horizon": self.prompt_horizon,
            "window": self.window,
            "update_mode": self.update_mode,
            "update_interval": self.update_interval,
            "short_episode_policy": "real_variable_length_no_padding",
            "empty_prompt_policy": "zero_length",
            "vector_batching": "grouped_by_exact_prompt_length",
            "action_semantics": ACTION_SEMANTICS,
            "cross_reset_transitions_allowed": False,
            "online_prompt_updates_within_episode": self.update_mode == "block",
        }


class T2MIROnlinePolicy:
    """DPT policy whose prompt comes only from its own preceding rollout."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        t2mir_root: Path,
        *,
        num_envs: int,
        prompt_horizon: int = 64,
        window: str = "last",
        update_mode: str = "episode",
        update_interval: int | None = None,
        expected_sha256: str | None = None,
        expected_routing: Mapping[str, Any] | None = None,
        expected_supervision_modes: Sequence[str] = ("all_tokens",),
    ) -> None:
        import sys

        root_text = str(t2mir_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from algorithms.policy import DPTTransformerMOE

        checkpoint_path = checkpoint_path.resolve()
        # Formal checkpoints also contain optimizer state.  Load on CPU first
        # so evaluation does not temporarily duplicate those tensors in VRAM.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.provenance = validate_checkpoint_contract(
            checkpoint_path,
            checkpoint,
            expected_sha256=expected_sha256,
            expected_routing=expected_routing,
            expected_prompt_horizon=prompt_horizon,
            expected_supervision_modes=expected_supervision_modes,
        )
        config = checkpoint["config"]
        self.model = DPTTransformerMOE(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            config,
            action_tanh=False,
            discrete_environment=False,
        ).to(device)
        self.model.load_state_dict(checkpoint["policy"])
        self.model.eval()
        self.device = device
        self.state_mean = torch.as_tensor(checkpoint["state_mean"], device=device)
        self.state_std = torch.as_tensor(checkpoint["state_std"], device=device)
        self.context = OnlineEpisodePromptBuffer(
            num_envs,
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            prompt_horizon=prompt_horizon,
            window=window,
            update_mode=update_mode,
            update_interval=update_interval,
        )
        self._episode_prompts: tuple[PromptBatch, ...] | None = None

    def start_episode(self, episode_index: int) -> tuple[PromptBatch, ...]:
        self.context.start_episode(episode_index)
        self._episode_prompts = self.context.prompt_groups(self.device)
        return self._episode_prompts

    def reset_context(self) -> None:
        """Clear prompt history after a declared high-level task switch."""

        self.context.reset()
        self._episode_prompts = None

    def maybe_refresh_prompt(self, completed_steps: int) -> tuple[PromptBatch, ...] | None:
        """Promote a causal current-episode block at a declared boundary.

        The transition at ``completed_steps - 1`` has already been recorded,
        so the refreshed prompt can only affect future actions.  No state,
        reward, or action from the query step itself leaks into its prediction.
        """

        if completed_steps <= 0:
            raise ValueError("completed_steps must be positive")
        if self.context.update_mode != "block":
            return None
        if completed_steps % self.context.update_interval != 0:
            return None
        self._episode_prompts = self.context.prompt_groups(
            self.device, current_if_complete=True
        )
        return self._episode_prompts

    @torch.inference_mode()
    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        if self._episode_prompts is None:
            raise RuntimeError("start_episode must be called before policy inference")
        if tuple(observations.shape) != (self.context.num_envs, self.context.state_dim):
            raise ValueError(
                f"observation shape {tuple(observations.shape)} does not match "
                f"{(self.context.num_envs, self.context.state_dim)}"
            )
        observations = observations.to(self.device)
        result = torch.empty(
            (self.context.num_envs, self.context.action_dim),
            dtype=observations.dtype,
            device=self.device,
        )
        for prompt in self._episode_prompts:
            ids = torch.tensor(prompt.env_ids, dtype=torch.long, device=self.device)
            prompt_states = (prompt.states - self.state_mean) / self.state_std
            query = ((observations.index_select(0, ids) - self.state_mean) / self.state_std).unsqueeze(1)
            timesteps = torch.arange(
                prompt.states.shape[1] + 1, dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(len(prompt.env_ids), -1)
            group_actions = self.model(
                prompt_states,
                prompt.actions,
                prompt.rewards,
                query,
                timesteps=timesteps,
                attention_masks=prompt.attention_mask,
                eval=True,
            )
            result.index_copy_(0, ids, group_actions)
        return result

    def record_transition(
        self,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
        rewards: torch.Tensor,
        active: torch.Tensor | Sequence[bool],
        controller_modes: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        self.context.append(
            states, policy_actions, rewards, active, controller_modes=controller_modes
        )

    def finish_episode(self) -> tuple[EpisodeTrajectory, ...]:
        trajectories = self.context.finish_episode()
        self._episode_prompts = None
        return trajectories
