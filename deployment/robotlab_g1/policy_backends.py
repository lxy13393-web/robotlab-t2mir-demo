"""Policy inference adapters for the RobotLab G1 deployment stack.

The public API is deliberately simulator and ROS independent.  Observations and
actions cross the boundary as NumPy arrays, while each backend owns the runtime
needed by its model format.

The T2MIR adapter does not implement a second prompt buffer.  It delegates the
complete episode protocol to :class:`T2MIROnlinePolicy`, the same implementation
used by the official online evaluator.  A transition recorded here therefore
means ``(pre-step observation, raw policy action, post-step reward)``; actuator
lag or other external action transforms must not be written into the prompt.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import DEFAULT_CONTRACT

G1_OBSERVATION_DIM = DEFAULT_CONTRACT.state_dim
G1_ACTION_DIM = DEFAULT_CONTRACT.action_dim
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T2MIR_ROOT = _REPOSITORY_ROOT / "methods/t2mir"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float32_batch(
    values: ArrayLike,
    *,
    width: int,
    batch_size: int,
    name: str,
) -> tuple[NDArray[np.float32], bool]:
    """Validate one vector or one fixed-size batch and return a contiguous batch."""

    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    expected = (batch_size, width)
    if array.shape != expected:
        raise ValueError(f"{name} shape {array.shape} does not match {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array), was_vector


def _as_vector(
    values: ArrayLike,
    *,
    batch_size: int,
    name: str,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0 and batch_size == 1:
        array = array.reshape(1)
    if array.shape == (batch_size, 1):
        array = array.reshape(batch_size)
    if array.shape != (batch_size,):
        raise ValueError(f"{name} shape {array.shape} does not match {(batch_size,)}")
    return np.ascontiguousarray(array)


class PolicyBackend(ABC):
    """Common deployment boundary for stateless and episode-context policies.

    ``predict`` preserves input rank: a single ``[observation_dim]`` vector
    produces ``[action_dim]`` and a batch produces ``[batch, action_dim]``.
    Episode methods are no-ops for policies such as PPO and meaningful for
    context policies such as T2MIR.
    """

    observation_dim: int
    action_dim: int
    batch_size: int
    provenance: dict[str, Any]

    @abstractmethod
    def predict(self, observations: ArrayLike) -> NDArray[np.float32]:
        """Infer raw policy actions from policy observations."""

    def start_episode(self, episode_index: int | None = None) -> Any:
        """Begin an episode and freeze any prompt used during that episode."""

        return None

    def record_transition(
        self,
        states: ArrayLike,
        policy_actions: ArrayLike,
        rewards: ArrayLike,
        active: ArrayLike | None = None,
        controller_modes: ArrayLike | None = None,
    ) -> None:
        """Record prompt data, if the policy consumes online context."""

    def finish_episode(self) -> Any:
        """Seal an episode and make it eligible as the next prompt."""

        return None

    def reset_context(self) -> None:
        """Discard context at an explicit high-level task boundary, if any."""


class PPOOnnxBackend(PolicyBackend):
    """CPU ONNX Runtime adapter for the exported RobotLab G1 PPO policy."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        observation_dim: int = G1_OBSERVATION_DIM,
        action_dim: int = G1_ACTION_DIM,
        batch_size: int = 1,
        input_name: str | None = None,
        output_name: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on deployment environment
            raise RuntimeError("PPOOnnxBackend requires the 'onnxruntime' package") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PPO ONNX model does not exist: {path}")
        if min(observation_dim, action_dim, batch_size) <= 0:
            raise ValueError("observation_dim, action_dim and batch_size must be positive")

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._select_port(self._session.get_inputs(), input_name, "input")
        self._output = self._select_port(self._session.get_outputs(), output_name, "output")
        self._validate_port(
            self._input,
            expected_shape=(batch_size, observation_dim),
            expected_type="tensor(float)",
            role="input",
        )
        self._validate_port(
            self._output,
            expected_shape=(batch_size, action_dim),
            expected_type="tensor(float)",
            role="output",
        )

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        model_sha256 = _sha256_file(path)
        if expected_sha256 is not None and model_sha256 != expected_sha256:
            raise ValueError(
                f"PPO model SHA-256 mismatch: expected {expected_sha256}, got {model_sha256}"
            )
        self.provenance = {
            "backend": "ppo_onnx",
            "model": str(path),
            "model_sha256": model_sha256,
            "provider": "CPUExecutionProvider",
            "input_name": self._input.name,
            "output_name": self._output.name,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def _select_port(ports: Sequence[Any], requested_name: str | None, role: str) -> Any:
        if requested_name is None:
            if len(ports) != 1:
                names = [port.name for port in ports]
                raise ValueError(
                    f"ONNX model has {len(ports)} {role}s {names}; specify {role}_name"
                )
            return ports[0]
        matches = [port for port in ports if port.name == requested_name]
        if len(matches) != 1:
            names = [port.name for port in ports]
            raise ValueError(f"ONNX {role} {requested_name!r} is absent; available={names}")
        return matches[0]

    @staticmethod
    def _validate_port(
        port: Any,
        *,
        expected_shape: tuple[int, int],
        expected_type: str,
        role: str,
    ) -> None:
        actual_shape = tuple(port.shape)
        if len(actual_shape) != 2:
            raise ValueError(f"ONNX {role} {port.name!r} must be rank 2, got {actual_shape}")
        for axis, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if isinstance(actual, Integral) and int(actual) != expected:
                raise ValueError(
                    f"ONNX {role} {port.name!r} axis {axis} is {actual}, expected {expected}"
                )
        if port.type != expected_type:
            raise ValueError(
                f"ONNX {role} {port.name!r} type is {port.type!r}, expected {expected_type!r}"
            )

    def predict(self, observations: ArrayLike) -> NDArray[np.float32]:
        batch, was_vector = _as_float32_batch(
            observations,
            width=self.observation_dim,
            batch_size=self.batch_size,
            name="observations",
        )
        outputs = self._session.run([self._output.name], {self._input.name: batch})
        actions = np.asarray(outputs[0], dtype=np.float32)
        expected = (self.batch_size, self.action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"PPO output shape {actions.shape} does not match {expected}")
        if not np.isfinite(actions).all():
            raise RuntimeError("PPO output contains NaN or Inf")
        actions = np.ascontiguousarray(actions)
        return actions[0] if was_vector else actions


class T2MIRTorchBackend(PolicyBackend):
    """Torch adapter for source-faithful official RobotLab T2MIR checkpoints.

    The backend defaults to CPU and to the source-faithful previous-episode
    prompt contract.  ``update_mode="block"`` selects the causal within-rollout
    protocol used by the best RobotLab Isaac evaluations.  The pilot3 smoke
    checkpoint is intentionally accepted by the same checkpoint validator used
    for formal A--D evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        t2mir_root: str | Path = DEFAULT_T2MIR_ROOT,
        num_envs: int = 1,
        prompt_horizon: int = 64,
        window: str = "last",
        update_mode: str = "episode",
        update_interval: int | None = None,
        device: str = "cpu",
        expected_observation_dim: int | None = G1_OBSERVATION_DIM,
        expected_action_dim: int | None = G1_ACTION_DIM,
        expected_sha256: str | None = None,
        expected_routing: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on deployment environment
            raise RuntimeError("T2MIRTorchBackend requires the 'torch' package") from exc

        from pipeline.protocols.t2mir_online_context import T2MIROnlinePolicy

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        root = Path(t2mir_root).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"T2MIR checkpoint does not exist: {checkpoint}")
        if not root.is_dir():
            raise FileNotFoundError(f"T2MIR source tree does not exist: {root}")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        torch_device = torch.device(device)
        self._policy = T2MIROnlinePolicy(
            checkpoint,
            torch_device,
            root,
            num_envs=num_envs,
            prompt_horizon=prompt_horizon,
            window=window,
            update_mode=update_mode,
            update_interval=update_interval,
            expected_sha256=expected_sha256,
            expected_routing=expected_routing,
            expected_supervision_modes=("all_tokens", "final_token"),
        )
        self.observation_dim = int(self._policy.provenance["state_dim"])
        self.action_dim = int(self._policy.provenance["action_dim"])
        self.batch_size = int(num_envs)
        if expected_observation_dim is not None and self.observation_dim != expected_observation_dim:
            raise ValueError(
                f"T2MIR state_dim={self.observation_dim}; expected {expected_observation_dim}"
            )
        if expected_action_dim is not None and self.action_dim != expected_action_dim:
            raise ValueError(f"T2MIR action_dim={self.action_dim}; expected {expected_action_dim}")

        self._torch = torch
        self._episode_active = False
        self._last_episode_index = -1
        self._episode_steps = 0
        self.provenance = dict(self._policy.provenance)
        self.provenance.update(
            {
                "backend": "t2mir_torch",
                "device": str(torch_device),
                "batch_size": self.batch_size,
                "prompt_protocol": self._policy.context.protocol_metadata(),
            }
        )

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def start_episode(self, episode_index: int | None = None) -> Any:
        if self._episode_active:
            raise RuntimeError("finish_episode must be called before starting another episode")
        if episode_index is None:
            episode_index = self._last_episode_index + 1
        prompts = self._policy.start_episode(int(episode_index))
        self._episode_active = True
        self._last_episode_index = int(episode_index)
        self._episode_steps = 0
        return prompts

    def predict(self, observations: ArrayLike) -> NDArray[np.float32]:
        batch, was_vector = _as_float32_batch(
            observations,
            width=self.observation_dim,
            batch_size=self.batch_size,
            name="observations",
        )
        if not self._episode_active:
            raise RuntimeError("start_episode must be called before T2MIR inference")
        tensor = self._torch.from_numpy(batch)
        actions = self._policy(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        expected = (self.batch_size, self.action_dim)
        if actions.shape != expected:
            raise RuntimeError(f"T2MIR output shape {actions.shape} does not match {expected}")
        if not np.isfinite(actions).all():
            raise RuntimeError("T2MIR output contains NaN or Inf")
        actions = np.ascontiguousarray(actions)
        return actions[0] if was_vector else actions

    def record_transition(
        self,
        states: ArrayLike,
        policy_actions: ArrayLike,
        rewards: ArrayLike,
        active: ArrayLike | None = None,
        controller_modes: ArrayLike | None = None,
    ) -> None:
        """Append pre-reset transitions using raw, pre-actuator policy actions."""

        if not self._episode_active:
            raise RuntimeError("start_episode must be called before recording transitions")
        state_batch, _ = _as_float32_batch(
            states,
            width=self.observation_dim,
            batch_size=self.batch_size,
            name="states",
        )
        action_batch, _ = _as_float32_batch(
            policy_actions,
            width=self.action_dim,
            batch_size=self.batch_size,
            name="policy_actions",
        )
        reward_vector = _as_vector(
            rewards,
            batch_size=self.batch_size,
            name="rewards",
            dtype=np.dtype(np.float32),
        )
        if not np.isfinite(reward_vector).all():
            raise ValueError("rewards contains NaN or Inf")
        if active is None:
            active_vector = np.ones(self.batch_size, dtype=np.bool_)
        else:
            active_vector = _as_vector(
                active,
                batch_size=self.batch_size,
                name="active",
                dtype=np.dtype(np.bool_),
            )
        mode_vector = None
        if controller_modes is not None:
            mode_vector = _as_vector(
                controller_modes,
                batch_size=self.batch_size,
                name="controller_modes",
                dtype=np.dtype(np.int64),
            )
            mode_vector = self._torch.from_numpy(mode_vector)

        self._policy.record_transition(
            self._torch.from_numpy(state_batch),
            self._torch.from_numpy(action_batch),
            self._torch.from_numpy(reward_vector),
            self._torch.from_numpy(active_vector),
            controller_modes=mode_vector,
        )
        # A block prompt becomes eligible only after its final transition has
        # been recorded. Refreshing here preserves the same causal boundary as
        # evaluate_t2mir_online.py: it can affect the next action, never the
        # action that generated the transition itself.
        self._episode_steps += 1
        self._policy.maybe_refresh_prompt(self._episode_steps)

    def finish_episode(self) -> Any:
        if not self._episode_active:
            raise RuntimeError("no active T2MIR episode to finish")
        trajectories = self._policy.finish_episode()
        self._episode_active = False
        self._episode_steps = 0
        return trajectories

    def reset_context(self) -> None:
        if self._episode_active:
            raise RuntimeError("finish_episode must be called before resetting context")
        self._policy.reset_context()
        self._last_episode_index = -1
        self._episode_steps = 0


__all__ = [
    "DEFAULT_T2MIR_ROOT",
    "G1_ACTION_DIM",
    "G1_OBSERVATION_DIM",
    "PPOOnnxBackend",
    "PolicyBackend",
    "T2MIRTorchBackend",
]
