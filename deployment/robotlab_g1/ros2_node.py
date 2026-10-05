"""ROS 2 transport and a ROS-independent policy-loop core for RobotLab G1.

The module deliberately does not import :mod:`rclpy` or ROS message packages
at import time.  ``Ros2PolicyCore`` is therefore usable in CPU-only tests and
on machines without ROS.  Call :func:`create_ros2_node` only in a sourced ROS
2 environment to add the transport layer.

The external reward topic is interpreted as feedback for the most recently
published policy action.  That transition is recorded immediately before the
next policy tick (or before an episode is closed), which keeps T2MIR prompt
tuples aligned as ``(pre-step observation, raw policy action, post-step
reward)``.  ``done`` and ``reset`` are hard episode boundaries; transitions
are never joined across either boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import importlib
import importlib.util
import math
import os
from threading import RLock
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import DEFAULT_CONTRACT, DeploymentContract
from .observation import ObservationBuilder, RobotState


class PolicyBackendLike(Protocol):
    """Structural interface implemented by PPO and T2MIR backends."""

    observation_dim: int
    action_dim: int
    batch_size: int
    provenance: Mapping[str, Any]

    def predict(self, observations: Any) -> Any: ...

    def start_episode(self, episode_index: int | None = None) -> Any: ...

    def record_transition(
        self,
        states: Any,
        policy_actions: Any,
        rewards: Any,
        active: Any | None = None,
        controller_modes: Any | None = None,
    ) -> None: ...

    def finish_episode(self) -> Any: ...


@dataclass(frozen=True)
class PolicyLoopConfig:
    """Timing and safety settings independent of ROS message transport."""

    command_timeout_s: float = 0.5
    initial_reward: float = 0.0
    require_fresh_reward: bool = True

    def __post_init__(self) -> None:
        if self.command_timeout_s <= 0.0 or not math.isfinite(self.command_timeout_s):
            raise ValueError("command_timeout_s must be finite and positive")
        if not math.isfinite(self.initial_reward):
            raise ValueError("initial_reward must be finite")


@dataclass(frozen=True)
class Ros2NodeConfig:
    """Standard-message ROS topic contract for the minimum deployment node."""

    node_name: str = "robotlab_g1_policy"
    cmd_vel_topic: str = "/cmd_vel"
    reset_topic: str = "/robotlab_g1/reset"
    reward_topic: str = "/robotlab_g1/reward"
    done_topic: str = "/robotlab_g1/done"
    joint_state_topic: str = "/joint_states"
    joint_target_topic: str = "/robotlab_g1/joint_targets"
    policy_action_topic: str = "/robotlab_g1/policy_action"
    diagnostics_topic: str = "/diagnostics"
    diagnostics_period_s: float = 1.0
    qos_depth: int = 10
    hardware_id: str = "robotlab_g1"
    enable_unsequenced_feedback_topics: bool = False

    def __post_init__(self) -> None:
        string_fields = (
            self.node_name,
            self.cmd_vel_topic,
            self.reset_topic,
            self.reward_topic,
            self.done_topic,
            self.joint_state_topic,
            self.joint_target_topic,
            self.policy_action_topic,
            self.diagnostics_topic,
            self.hardware_id,
        )
        if any(not value for value in string_fields):
            raise ValueError("ROS node, topic and hardware names cannot be empty")
        if self.diagnostics_period_s <= 0.0 or not math.isfinite(self.diagnostics_period_s):
            raise ValueError("diagnostics_period_s must be finite and positive")
        if self.qos_depth <= 0:
            raise ValueError("qos_depth must be positive")


@dataclass(frozen=True)
class PolicyStep:
    """One policy result in canonical RobotLab joint order."""

    observation: NDArray[np.float32]
    policy_action: NDArray[np.float32]
    joint_targets: NDArray[np.float32]
    velocity_command: NDArray[np.float32]
    command_stale: bool
    recorded_reward: float | None
    applied_boundary: str | None
    episode_index: int
    episode_step: int
    total_step: int


@dataclass
class PolicyTimingStats:
    """Bounded 50 Hz callback timing statistics for diagnostics and reports."""

    deadline_s: float
    sample_count: int = 0
    deadline_misses: int = 0
    elapsed_sum_s: float = 0.0
    last_elapsed_s: float = 0.0
    max_elapsed_s: float = 0.0
    _window: deque[float] = field(
        default_factory=lambda: deque(maxlen=4096), repr=False
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0.0:
            raise ValueError("policy timing deadline must be finite and positive")

    def observe(self, elapsed_s: float) -> None:
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("policy callback duration must be finite and non-negative")
        elapsed = float(elapsed_s)
        self.sample_count += 1
        self.elapsed_sum_s += elapsed
        self.last_elapsed_s = elapsed
        self.max_elapsed_s = max(self.max_elapsed_s, elapsed)
        self.deadline_misses += int(elapsed > self.deadline_s)
        self._window.append(elapsed)

    def as_dict(self) -> dict[str, float | int]:
        mean_s = self.elapsed_sum_s / self.sample_count if self.sample_count else 0.0
        p95_s = (
            float(np.percentile(np.asarray(self._window, dtype=np.float64), 95.0))
            if self._window
            else 0.0
        )
        return {
            "deadline_ms": self.deadline_s * 1000.0,
            "sample_count": self.sample_count,
            "window_sample_count": len(self._window),
            "last_ms": self.last_elapsed_s * 1000.0,
            "mean_ms": mean_s * 1000.0,
            "p95_ms": p95_s * 1000.0,
            "max_ms": self.max_elapsed_s * 1000.0,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": (
                self.deadline_misses / self.sample_count if self.sample_count else 0.0
            ),
        }

    def diagnostics(self) -> dict[str, str]:
        return {
            f"timing_{key}": f"{value:.6g}" if isinstance(value, float) else str(value)
            for key, value in self.as_dict().items()
        }


@dataclass(frozen=True)
class _PendingTransition:
    observation: NDArray[np.float32]
    action: NDArray[np.float32]


class Ros2PolicyCore:
    """Single-robot policy loop with explicit episode and feedback semantics.

    The core owns the command timeout, previous policy action and backend
    episode lifecycle.  A simulator supplies a :class:`RobotState`; the core
    overwrites its velocity command and previous action with the authoritative
    values received/generated here before constructing the 123-vector.
    """

    def __init__(
        self,
        backend: PolicyBackendLike,
        *,
        contract: DeploymentContract = DEFAULT_CONTRACT,
        observation_builder: ObservationBuilder | None = None,
        config: PolicyLoopConfig | None = None,
    ) -> None:
        self.backend = backend
        self.contract = contract
        self.observation_builder = observation_builder or ObservationBuilder(contract)
        self.config = config or PolicyLoopConfig()
        if int(backend.observation_dim) != contract.state_dim:
            raise ValueError(
                f"backend observation_dim={backend.observation_dim}; expected {contract.state_dim}"
            )
        if int(backend.action_dim) != contract.action_dim:
            raise ValueError(f"backend action_dim={backend.action_dim}; expected {contract.action_dim}")
        if int(getattr(backend, "batch_size", 1)) != 1:
            raise ValueError("the minimum ROS 2 node supports exactly one robot/backend batch item")

        self._lock = RLock()
        self._command = np.zeros(3, dtype=np.float32)
        self._command_stamp_s: float | None = None
        self._previous_action = np.zeros(contract.action_dim, dtype=np.float32)
        self._pending_reward: float | None = None
        self._reward_updates = 0
        self._missing_reward_feedback = 0
        self._discarded_transitions = 0
        self._pending: _PendingTransition | None = None
        self._pending_boundary: str | None = None
        self._episode_active = False
        self._episode_index = 0
        self._episode_step = 0
        self._total_step = 0
        self._closed = False

    @property
    def episode_index(self) -> int:
        with self._lock:
            return self._episode_index

    @property
    def previous_action(self) -> NDArray[np.float32]:
        with self._lock:
            return self._previous_action.copy()

    def update_command(self, linear_x: float, linear_y: float, angular_z: float, *, now_s: float) -> None:
        """Store the planar velocity command from a ``geometry_msgs/Twist``."""

        command = np.asarray((linear_x, linear_y, angular_z), dtype=np.float32)
        if not np.isfinite(command).all() or not math.isfinite(now_s):
            raise ValueError("velocity command and timestamp must be finite")
        with self._lock:
            self._command = command
            self._command_stamp_s = float(now_s)

    def update_reward(self, reward: float) -> None:
        """Set post-step feedback for the most recently emitted action."""

        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        with self._lock:
            if self._pending is None:
                raise RuntimeError("reward arrived without a pending policy transition")
            if self._pending_reward is not None:
                raise RuntimeError("duplicate reward for the pending policy transition")
            self._pending_reward = float(reward)
            self._reward_updates += 1

    def update_executed_action(self, action: Sequence[float]) -> None:
        """Override the next observation's action term with actuator feedback.

        With no actuator transform the raw policy action is also the executed
        action, so :meth:`step` sets it automatically.  A MuJoCo bridge that
        applies first-order lag must call this method after stepping physics.
        The pending T2MIR tuple intentionally retains the *raw* policy action.
        """

        executed = np.asarray(action, dtype=np.float32)
        if executed.shape != (self.contract.action_dim,):
            raise ValueError(
                f"executed action shape {executed.shape}; expected {(self.contract.action_dim,)}"
            )
        if not np.isfinite(executed).all():
            raise ValueError("executed action contains NaN or Inf")
        with self._lock:
            if self._closed:
                raise RuntimeError("policy core is closed")
            self._previous_action = np.ascontiguousarray(executed).copy()

    def request_reset(self, reason: str = "reset") -> None:
        self._request_boundary(reason)

    def notify_done(self, done: bool = True, reason: str = "done") -> None:
        if done:
            self._request_boundary(reason)

    def abort_pending_transition(self, reason: str = "transition_fault") -> bool:
        """Discard an incomplete action and schedule a clean episode boundary.

        This is only for a transport/plant fault after policy inference but
        before trustworthy post-step feedback exists.  Fabricating a reward is
        worse than dropping the transition: it would poison a T2MIR prompt and
        strict reward mode would otherwise remain permanently blocked.
        """

        if not reason:
            raise ValueError("episode-boundary reason cannot be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("policy core is closed")
            discarded = self._pending is not None
            if discarded:
                self._discarded_transitions += 1
            self._pending = None
            self._pending_reward = None
            self._previous_action.fill(0.0)
            if self._pending_boundary is None:
                self._pending_boundary = reason
            return discarded

    def _request_boundary(self, reason: str) -> None:
        if not reason:
            raise ValueError("episode-boundary reason cannot be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("policy core is closed")
            # Coalesce repeated reset/done messages before the next policy tick.
            if self._pending_boundary is None:
                self._pending_boundary = reason

    def _command_at(self, now_s: float) -> tuple[NDArray[np.float32], bool]:
        if not math.isfinite(now_s):
            raise ValueError("policy timestamp must be finite")
        stamp = self._command_stamp_s
        stale = stamp is None or now_s < stamp or now_s - stamp > self.config.command_timeout_s
        if stale:
            return np.zeros(3, dtype=np.float32), True
        return self._command.copy(), False

    def _start_episode(self) -> None:
        self.backend.start_episode(self._episode_index)
        self._episode_active = True
        self._episode_step = 0

    def _record_pending(self) -> float | None:
        if self._pending is None:
            return None
        if self._pending_reward is None:
            if self.config.require_fresh_reward:
                raise RuntimeError("no fresh reward is available for the pending transition")
            reward = float(self.config.initial_reward)
            self._missing_reward_feedback += 1
        else:
            reward = self._pending_reward
        self.backend.record_transition(
            self._pending.observation,
            self._pending.action,
            reward,
            active=True,
        )
        self._pending = None
        self._pending_reward = None
        return reward

    def _apply_boundary(self) -> str | None:
        reason = self._pending_boundary
        if reason is None:
            return None
        if self._episode_active:
            self.backend.finish_episode()
            self._episode_active = False
            self._episode_index += 1
        self._previous_action.fill(0.0)
        self._pending_reward = None
        self._start_episode()
        self._pending_boundary = None
        return reason

    def step(self, state: RobotState, *, now_s: float) -> PolicyStep:
        """Infer and stage one transition; no ROS graph is required."""

        with self._lock:
            if self._closed:
                raise RuntimeError("policy core is closed")
            if not self._episode_active:
                # A reset received before the first tick does not fabricate an
                # empty episode or change the official episode-zero prompt.
                if self._pending_boundary is not None:
                    self._pending_reward = None
                self._pending_boundary = None
                self._start_episode()

            recorded_reward = self._record_pending()
            applied_boundary = self._apply_boundary()
            command, command_stale = self._command_at(now_s)
            policy_state = replace(
                state,
                velocity_command=command,
                previous_action=self._previous_action.copy(),
            )
            observation = self.observation_builder.build(policy_state)
            action = np.asarray(self.backend.predict(observation), dtype=np.float32)
            if action.shape == (1, self.contract.action_dim):
                action = action[0]
            if action.shape != (self.contract.action_dim,):
                raise ValueError(
                    f"backend action shape {action.shape}; expected {(self.contract.action_dim,)}"
                )
            if not np.isfinite(action).all():
                raise ValueError("backend action contains NaN or Inf")
            action = np.ascontiguousarray(action)
            targets = np.ascontiguousarray(self.contract.action_to_joint_targets(action))

            self._pending = _PendingTransition(observation.copy(), action.copy())
            self._previous_action = action.copy()
            step = PolicyStep(
                observation=observation.copy(),
                policy_action=action.copy(),
                joint_targets=targets,
                velocity_command=command,
                command_stale=command_stale,
                recorded_reward=recorded_reward,
                applied_boundary=applied_boundary,
                episode_index=self._episode_index,
                episode_step=self._episode_step,
                total_step=self._total_step,
            )
            self._episode_step += 1
            self._total_step += 1
            return step

    def diagnostics(self, *, now_s: float) -> dict[str, str]:
        """Return stable key/value diagnostics consumable without ROS."""

        with self._lock:
            _, command_stale = self._command_at(now_s)
            provenance = getattr(self.backend, "provenance", {})
            backend_name = str(provenance.get("backend", type(self.backend).__name__))
            return {
                "backend": backend_name,
                "episode_index": str(self._episode_index),
                "episode_step": str(self._episode_step),
                "total_step": str(self._total_step),
                "command_stale": str(command_stale).lower(),
                "reward_updates": str(self._reward_updates),
                "reward_pending": str(self._pending_reward is not None).lower(),
                "missing_reward_feedback": str(self._missing_reward_feedback),
                "discarded_transitions": str(self._discarded_transitions),
                "require_fresh_reward": str(self.config.require_fresh_reward).lower(),
                "pending_transition": str(self._pending is not None).lower(),
                "pending_boundary": self._pending_boundary or "",
                "policy_hz": f"{self.contract.policy_hz:.6g}",
                "contract_sha256": self.contract.sha256,
            }

    def finalize(self) -> None:
        """Record the final pending transition and close the active episode."""

        with self._lock:
            if self._closed:
                return
            if self._episode_active:
                self._record_pending()
                self.backend.finish_episode()
                self._episode_active = False
            self._closed = True


StateProvider = Callable[[], RobotState | None]
ResetHandler = Callable[[], None]
PolicyStepHandler = Callable[[PolicyStep], Any]


def _diagnostic_level_byte(level: int) -> bytes:
    """Encode ``diagnostic_msgs/DiagnosticStatus.level`` for ROS 2 Humble.

    Humble's generated Python message represents the IDL ``byte`` field as a
    one-byte ``bytes`` object rather than an integer.  Keeping this conversion
    at the transport boundary also remains accepted by newer generated
    messages that retain the same IDL type.
    """

    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 255:
        raise ValueError("diagnostic level must be an integer in [0, 255]")
    return bytes((level,))


def ros2_environment_report() -> dict[str, Any]:
    """Inspect ROS Python package visibility without importing ROS modules."""

    package_names = (
        "rclpy",
        "geometry_msgs",
        "sensor_msgs",
        "std_msgs",
        "diagnostic_msgs",
    )
    packages: dict[str, bool] = {}
    for name in package_names:
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[name] = False
    distro = os.environ.get("ROS_DISTRO")
    return {
        "available": all(packages.values()),
        "ros_distro": distro,
        "is_humble": distro == "humble",
        "packages": packages,
    }


def _load_ros2() -> SimpleNamespace:
    """Load ROS lazily and produce one actionable error when it is absent."""

    try:
        rclpy = importlib.import_module("rclpy")
        node_module = importlib.import_module("rclpy.node")
        geometry = importlib.import_module("geometry_msgs.msg")
        sensor = importlib.import_module("sensor_msgs.msg")
        std = importlib.import_module("std_msgs.msg")
        diagnostic = importlib.import_module("diagnostic_msgs.msg")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source /opt/ros/humble/setup.bash "
            "before creating the ROS node"
        ) from exc
    return SimpleNamespace(
        rclpy=rclpy,
        Node=node_module.Node,
        Twist=geometry.Twist,
        JointState=sensor.JointState,
        Empty=std.Empty,
        Float32=std.Float32,
        Bool=std.Bool,
        Float32MultiArray=std.Float32MultiArray,
        DiagnosticArray=diagnostic.DiagnosticArray,
        DiagnosticStatus=diagnostic.DiagnosticStatus,
        KeyValue=diagnostic.KeyValue,
    )


def create_ros2_node(
    core: Ros2PolicyCore,
    state_provider: StateProvider,
    *,
    reset_handler: ResetHandler | None = None,
    step_handler: PolicyStepHandler | None = None,
    config: Ros2NodeConfig | None = None,
) -> Any:
    """Create an ``rclpy.node.Node`` backed by ``core``.

    ``rclpy.init()`` must be called by the application before this factory.
    The simulator/robot adapter remains injectable through ``state_provider``
    and ``reset_handler``; this module never imports or starts MuJoCo.
    """

    ros = _load_ros2()
    node_config = config or Ros2NodeConfig()

    class _RobotLabG1RosNode(ros.Node):
        def __init__(self) -> None:
            super().__init__(node_config.node_name)
            self._core = core
            self._state_provider = state_provider
            self._reset_handler = reset_handler
            self._step_handler = step_handler
            self._policy_timing = PolicyTimingStats(core.contract.policy_dt)
            self._last_diagnostic_s = -math.inf
            self._last_fault: str | None = None
            depth = node_config.qos_depth
            self._joint_state_publisher = self.create_publisher(
                ros.JointState, node_config.joint_state_topic, depth
            )
            self._joint_target_publisher = self.create_publisher(
                ros.JointState, node_config.joint_target_topic, depth
            )
            self._policy_action_publisher = self.create_publisher(
                ros.Float32MultiArray, node_config.policy_action_topic, depth
            )
            self._diagnostics_publisher = self.create_publisher(
                ros.DiagnosticArray, node_config.diagnostics_topic, depth
            )
            self._cmd_subscription = self.create_subscription(
                ros.Twist, node_config.cmd_vel_topic, self._on_cmd_vel, depth
            )
            self._reset_subscription = self.create_subscription(
                ros.Empty, node_config.reset_topic, self._on_reset, depth
            )
            # Float32/Bool messages carry no transition sequence id.  They are
            # disabled by default because delayed terminal feedback can cross
            # an episode boundary.  The in-process MuJoCo handler is
            # synchronous and does not use these topics.
            self._reward_subscription = None
            self._done_subscription = None
            if node_config.enable_unsequenced_feedback_topics:
                self._reward_subscription = self.create_subscription(
                    ros.Float32, node_config.reward_topic, self._on_reward, depth
                )
                self._done_subscription = self.create_subscription(
                    ros.Bool, node_config.done_topic, self._on_done, depth
                )
            self._policy_timer = self.create_timer(core.contract.policy_dt, self._on_policy_timer)

        def _now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1.0e-9

        def _on_cmd_vel(self, message: Any) -> None:
            try:
                self._core.update_command(
                    message.linear.x,
                    message.linear.y,
                    message.angular.z,
                    now_s=self._now_s(),
                )
            except Exception as exc:  # ROS callback boundary
                self._report_fault(f"cmd_vel rejected: {exc}")

        def _on_reward(self, message: Any) -> None:
            try:
                self._core.update_reward(message.data)
            except Exception as exc:  # ROS callback boundary
                self._report_fault(f"reward rejected: {exc}")

        def _on_done(self, message: Any) -> None:
            try:
                self._core.notify_done(bool(message.data), "done_topic")
            except Exception as exc:  # ROS callback boundary
                self._report_fault(f"done rejected: {exc}")

        def _on_reset(self, _message: Any) -> None:
            try:
                # Mark the context boundary first.  Even if the physical reset
                # handler fails or partially succeeds, no transition may leak
                # across an attempted reset.
                self._core.request_reset("reset_topic")
                if self._reset_handler is not None:
                    self._reset_handler()
            except Exception as exc:  # ROS callback boundary
                self._report_fault(f"reset failed: {exc}")

        def _on_policy_timer(self) -> None:
            started = time.perf_counter()
            timing_observed = False
            try:
                state = self._state_provider()
                if state is None:
                    self._publish_diagnostics(1, "state provider is not ready")
                    return
                step = self._core.step(state, now_s=self._now_s())
                handler_result = None
                if self._step_handler is not None:
                    # The callback may step MuJoCo and feed the resulting
                    # reward/executed action back into ``core``.  It runs
                    # before publishing so callback failures suppress stale
                    # target messages and become diagnostics.
                    try:
                        handler_result = self._step_handler(step)
                    except Exception as handler_error:
                        self._core.abort_pending_transition("step_handler_fault")
                        if self._reset_handler is not None:
                            try:
                                self._reset_handler()
                            except Exception as reset_error:
                                raise RuntimeError(
                                    "policy step handler failed and recovery reset also "
                                    f"failed: handler={handler_error}; reset={reset_error}"
                                ) from handler_error
                        raise
                    post_state = self._state_provider()
                    if post_state is not None:
                        state = post_state
                stamp = self.get_clock().now().to_msg()
                self._publish_joint_state(state, stamp)
                executed_action = getattr(handler_result, "executed_action", None)
                if bool(getattr(handler_result, "terminated", False)) or bool(
                    getattr(handler_result, "truncated", False)
                ):
                    executed_action = np.zeros(
                        self._core.contract.action_dim, dtype=np.float32
                    )
                self._publish_targets(step, stamp, executed_action=executed_action)
                self._policy_timing.observe(time.perf_counter() - started)
                timing_observed = True
                self._last_fault = None
                message = "command timeout: zero command applied" if step.command_stale else "policy loop healthy"
                self._publish_diagnostics(1 if step.command_stale else 0, message)
            except Exception as exc:  # keep the executor alive and expose the fault
                if not timing_observed:
                    self._policy_timing.observe(time.perf_counter() - started)
                self._report_fault(f"policy tick failed: {exc}")

        def runtime_snapshot(self) -> dict[str, Any]:
            """Return JSON-ready policy/core timing state without changing it."""

            return {
                "policy_loop": self._core.diagnostics(now_s=self._now_s()),
                "timing": self._policy_timing.as_dict(),
                "last_fault": self._last_fault,
            }

        def _publish_joint_state(self, state: RobotState, stamp: Any) -> None:
            names = tuple(state.joint_names or self._core.contract.joint_names)
            self._core.contract.validate_joint_names(names, require_order=False)
            positions = np.asarray(state.joint_position, dtype=np.float32)
            velocities = np.asarray(state.joint_velocity, dtype=np.float32)
            expected = (self._core.contract.action_dim,)
            if positions.shape != expected or velocities.shape != expected:
                raise ValueError("joint state position/velocity dimensions do not match the contract")
            message = ros.JointState()
            message.header.stamp = stamp
            message.name = list(names)
            message.position = positions.astype(float).tolist()
            message.velocity = velocities.astype(float).tolist()
            self._joint_state_publisher.publish(message)

        def _publish_targets(
            self,
            step: PolicyStep,
            stamp: Any,
            *,
            executed_action: Any | None = None,
        ) -> None:
            target = ros.JointState()
            target.header.stamp = stamp
            target.name = list(self._core.contract.joint_names)
            if executed_action is None:
                positions = step.joint_targets
            else:
                positions = self._core.contract.action_to_joint_targets(executed_action)
            target.position = np.asarray(positions, dtype=float).tolist()
            self._joint_target_publisher.publish(target)
            raw_action = ros.Float32MultiArray()
            raw_action.data = step.policy_action.astype(float).tolist()
            self._policy_action_publisher.publish(raw_action)

        def _report_fault(self, message: str) -> None:
            self._last_fault = message
            self.get_logger().error(message)
            self._publish_diagnostics(2, message, force=True)

        def _publish_diagnostics(self, level: int, message: str, *, force: bool = False) -> None:
            now_s = self._now_s()
            if not force and now_s - self._last_diagnostic_s < node_config.diagnostics_period_s:
                return
            self._last_diagnostic_s = now_s
            values = self._core.diagnostics(now_s=now_s)
            values.update(self._policy_timing.diagnostics())
            status = ros.DiagnosticStatus()
            status.level = _diagnostic_level_byte(level)
            status.name = f"{self.get_fully_qualified_name()}/policy_loop"
            status.hardware_id = node_config.hardware_id
            status.message = self._last_fault or message
            status.values = [ros.KeyValue(key=key, value=value) for key, value in values.items()]
            array = ros.DiagnosticArray()
            array.header.stamp = self.get_clock().now().to_msg()
            array.status = [status]
            self._diagnostics_publisher.publish(array)

        def destroy_node(self) -> Any:
            self._core.finalize()
            return super().destroy_node()

    return _RobotLabG1RosNode()


def spin_ros2_node(node: Any) -> None:
    """Spin a node created by :func:`create_ros2_node`; caller owns init/shutdown."""

    _load_ros2().rclpy.spin(node)


__all__ = [
    "PolicyBackendLike",
    "PolicyLoopConfig",
    "PolicyStep",
    "PolicyTimingStats",
    "PolicyStepHandler",
    "ResetHandler",
    "Ros2NodeConfig",
    "Ros2PolicyCore",
    "StateProvider",
    "create_ros2_node",
    "ros2_environment_report",
    "spin_ros2_node",
]
