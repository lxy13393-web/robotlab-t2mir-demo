"""High-level planar goal controller for a velocity-conditioned locomotion policy."""

from __future__ import annotations

# Allow this CLI to be executed directly from a repository checkout.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repository_root = _Path(__file__).resolve().parents[2]
    if str(_repository_root) not in _sys.path:
        _sys.path.insert(0, str(_repository_root))

import math
from dataclasses import dataclass


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class GoalControllerCfg:
    """Tunable parameters for the goal controller."""

    position_tolerance: float = 0.25
    position_exit_tolerance: float = 0.40
    yaw_tolerance: float = 0.18
    yaw_exit_tolerance: float = 0.30
    turn_enter_error: float = 0.70
    turn_exit_error: float = 0.25
    max_forward_speed: float = 0.85
    min_forward_speed: float = 0.18
    max_yaw_rate: float = 0.80
    distance_gain: float = 0.55
    heading_gain: float = 1.35
    align_gain: float = 1.50
    hold_position_gain: float = 0.60
    hold_yaw_gain: float = 1.00
    max_hold_speed: float = 0.15
    max_hold_yaw_rate: float = 0.30
    hold_allow_backward: bool = True
    # A small forward stepping command turns an in-place yaw request into an
    # arc command.  This stays inside the velocity-policy training envelope
    # and is useful for low-level policies whose sim-to-sim in-place turning
    # authority is weak.  The default is zero, so dataset collection and the
    # original Isaac controller semantics are unchanged unless explicitly
    # enabled by a deployment evaluator.
    turn_forward_speed: float = 0.0
    # Final-yaw alignment happens near the target, so it needs a smaller arc
    # than the initial turn.  Keeping this separate avoids trading turning
    # authority for excessive position drift.
    align_forward_speed: float = 0.0


class GoalController:
    """Four-state controller that converts a planar goal into velocity commands."""

    TURN_TO_GOAL = "TURN_TO_GOAL"
    WALK_TO_GOAL = "WALK_TO_GOAL"
    ALIGN_FINAL_YAW = "ALIGN_FINAL_YAW"
    HOLD = "HOLD"

    def __init__(self, target_x: float, target_y: float, target_yaw: float, cfg: GoalControllerCfg | None = None):
        self.target_x = target_x
        self.target_y = target_y
        self.target_yaw = wrap_to_pi(target_yaw)
        self.cfg = cfg or GoalControllerCfg()
        self.state = self.TURN_TO_GOAL

    def errors(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """Return distance, heading-to-position error, and final-yaw error."""
        dx = self.target_x - x
        dy = self.target_y - y
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(desired_heading - yaw)
        final_yaw_error = wrap_to_pi(self.target_yaw - yaw)
        return distance, heading_error, final_yaw_error

    def compute(self, x: float, y: float, yaw: float) -> tuple[tuple[float, float, float], dict[str, float | str]]:
        """Compute body-frame ``(vx, vy, yaw_rate)`` and controller diagnostics."""
        distance, heading_error, final_yaw_error = self.errors(x, y, yaw)
        cfg = self.cfg

        # State transitions use separate enter/exit thresholds to avoid chatter.
        if self.state == self.TURN_TO_GOAL:
            if distance <= cfg.position_tolerance:
                self.state = self.ALIGN_FINAL_YAW
            elif abs(heading_error) <= cfg.turn_exit_error:
                self.state = self.WALK_TO_GOAL
        elif self.state == self.WALK_TO_GOAL:
            if distance <= cfg.position_tolerance:
                self.state = self.ALIGN_FINAL_YAW
            elif abs(heading_error) >= cfg.turn_enter_error:
                self.state = self.TURN_TO_GOAL
        elif self.state == self.ALIGN_FINAL_YAW:
            if distance >= cfg.position_exit_tolerance:
                self.state = self.TURN_TO_GOAL
            elif abs(final_yaw_error) <= cfg.yaw_tolerance:
                self.state = self.HOLD
        elif self.state == self.HOLD:
            if distance >= cfg.position_exit_tolerance:
                self.state = self.TURN_TO_GOAL
            elif abs(final_yaw_error) >= cfg.yaw_exit_tolerance:
                self.state = self.ALIGN_FINAL_YAW

        if self.state == self.TURN_TO_GOAL:
            yaw_rate = self._clip(cfg.heading_gain * heading_error, cfg.max_yaw_rate)
            command = (
                cfg.turn_forward_speed,
                0.0,
                yaw_rate,
            )
        elif self.state == self.WALK_TO_GOAL:
            forward_speed = self._clip_positive(cfg.distance_gain * distance, cfg.min_forward_speed, cfg.max_forward_speed)
            # Reduce forward speed while the heading is imperfect, without
            # stopping for the small errors expected during curved walking.
            heading_scale = max(0.25, math.cos(heading_error))
            command = (
                forward_speed * heading_scale,
                0.0,
                self._clip(cfg.heading_gain * heading_error, cfg.max_yaw_rate),
            )
        elif self.state == self.ALIGN_FINAL_YAW:
            yaw_rate = self._clip(cfg.align_gain * final_yaw_error, cfg.max_yaw_rate)
            command = (
                cfg.align_forward_speed,
                0.0,
                yaw_rate,
            )
        else:
            # A zero velocity command makes the current locomotion policy step
            # in place and slowly drift. Transform the remaining world-frame
            # position error into the robot frame and close the loop with small
            # velocity corrections while preserving the HOLD state.
            dx = self.target_x - x
            dy = self.target_y - y
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            error_x_body = cos_yaw * dx + sin_yaw * dy
            error_y_body = -sin_yaw * dx + cos_yaw * dy
            hold_x = self._clip(cfg.hold_position_gain * error_x_body, cfg.max_hold_speed)
            if not cfg.hold_allow_backward:
                hold_x = max(0.0, hold_x)
            command = (
                hold_x,
                self._clip(cfg.hold_position_gain * error_y_body, cfg.max_hold_speed),
                self._clip(cfg.hold_yaw_gain * final_yaw_error, cfg.max_hold_yaw_rate),
            )

        info: dict[str, float | str] = {
            "state": self.state,
            "distance": distance,
            "heading_error": heading_error,
            "final_yaw_error": final_yaw_error,
        }
        return command, info

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _clip_positive(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
