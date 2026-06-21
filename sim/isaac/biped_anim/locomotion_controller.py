"""Locomotion controller: turn body travel into a gait phase + smoothed speed.

Pure logic, no Isaac imports. The waypoint system in ``isaac_env`` already MOVES
the body and decides its speed (PERSON_WALK_SPEED / PERSON_STAIR_SPEED); this
component does NOT re-derive any of that. It only converts the motion the body
actually made between frames into:

  * ``phase``  -- a 0..1 gait-cycle clock advanced by *distance travelled* divided
                  by the active gait's stride length. Advancing by distance (not by
                  wall-clock) is what keeps the feet planted on the ground instead
                  of skating: one full L/R cycle per ``stride`` metres, always.
  * ``speed``  -- an EMA-smoothed estimate of the body speed, used by the gait to
                  scale stride/arm-swing and by the state machine to detect motion.

This is the "speed handling" wiring the goal asks for: speed stays owned by the
waypoint follower, and here it is read back from real displacement and fed into
the gait cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LocomotionSample:
    phase: float          # 0..1 gait-cycle position
    speed: float          # m/s, EMA-smoothed
    distance_delta: float # m travelled since last update
    moving: bool


class LocomotionController:
    def __init__(
        self,
        *,
        moving_speed_threshold: float = 0.04,
        speed_ema_alpha: float = 0.25,
        min_stride_m: float = 0.05,
    ) -> None:
        self._moving_threshold = float(moving_speed_threshold)
        self._alpha = float(speed_ema_alpha)
        self._min_stride = float(min_stride_m)
        self._phase = 0.0
        self._speed = 0.0
        self._last_xy: Optional[Tuple[float, float]] = None
        self._last_t: Optional[float] = None

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def speed(self) -> float:
        return self._speed

    def reset(self, xy: Tuple[float, float], now: Optional[float] = None) -> None:
        self._last_xy = (float(xy[0]), float(xy[1]))
        self._last_t = now
        self._speed = 0.0
        # Keep self._phase so a brief stop-and-go doesn't snap the legs to a new
        # cycle position; the gait blends out via the state machine instead.

    def update(
        self,
        xy: Tuple[float, float],
        now: float,
        stride_length_m: float,
    ) -> LocomotionSample:
        x, y = float(xy[0]), float(xy[1])
        stride = max(self._min_stride, float(stride_length_m))

        if self._last_xy is None or self._last_t is None:
            self._last_xy = (x, y)
            self._last_t = now
            return LocomotionSample(self._phase, 0.0, 0.0, False)

        dx = x - self._last_xy[0]
        dy = y - self._last_xy[1]
        dist = (dx * dx + dy * dy) ** 0.5
        dt = now - self._last_t

        inst_speed = (dist / dt) if dt > 1e-6 else 0.0
        # EMA toward the instantaneous speed (also decays to ~0 when standing).
        self._speed += self._alpha * (inst_speed - self._speed)

        # Advance the gait clock by ground covered, so footfalls track terrain.
        self._phase = (self._phase + dist / stride) % 1.0

        moving = inst_speed > self._moving_threshold

        self._last_xy = (x, y)
        self._last_t = now
        return LocomotionSample(self._phase, self._speed, dist, moving)
