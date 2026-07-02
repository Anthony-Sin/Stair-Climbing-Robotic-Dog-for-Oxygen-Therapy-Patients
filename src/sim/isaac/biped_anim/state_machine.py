"""Animation state machine: choose & smoothly blend Idle / FlatWalk / StairClimb.

Pure logic, no Isaac imports. It turns the two discrete inputs ``moving`` and
``terrain_class`` into a set of blend weights over the three gait styles, with:

  * an idle debounce so brief sub-threshold frames (waypoint-arrival snaps, a one-
    step rest) don't flicker the legs to a stand and back, and
  * a timed crossfade between flat-walk and stair-climb so the posture eases across
    the bottom/top of the staircase instead of popping.

Output ``weights`` always sum to ~1.0; the controller evaluates each active gait
and blends the resulting poses by these weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .types import AnimStyle, TerrainClass


@dataclass
class StateOutput:
    weights: Dict[AnimStyle, float]
    locomotion_style: AnimStyle  # dominant *moving* style (FLAT_WALK / STAIR_CLIMB)
    moving: bool


class AnimationStateMachine:
    def __init__(
        self,
        *,
        idle_debounce_sec: float = 0.5,
        terrain_blend_sec: float = 0.4,
        idle_blend_sec: float = 0.25,
    ) -> None:
        self._idle_debounce = float(idle_debounce_sec)
        self._terrain_blend = max(1e-3, float(terrain_blend_sec))
        self._idle_blend = max(1e-3, float(idle_blend_sec))

        # Crossfade scalars, each in [0, 1].
        self._move_w = 0.0   # 0 == idle, 1 == fully in a moving gait
        self._stair_w = 0.0  # 0 == flat walk, 1 == stair climb (within the moving gait)

        self._last_moving_time = -1e9
        self._last_t = None

    def update(self, moving: bool, terrain_class: TerrainClass, now: float) -> StateOutput:
        if self._last_t is None:
            self._last_t = now
        dt = max(0.0, now - self._last_t)
        self._last_t = now

        # --- idle debounce: enter motion instantly, leave it only after quiet ---
        if moving:
            self._last_moving_time = now
            effective_moving = True
        else:
            effective_moving = (now - self._last_moving_time) < self._idle_debounce

        move_target = 1.0 if effective_moving else 0.0
        stair_target = 1.0 if terrain_class == TerrainClass.STAIR else 0.0

        self._move_w = _approach(self._move_w, move_target, dt, self._idle_blend)
        self._stair_w = _approach(self._stair_w, stair_target, dt, self._terrain_blend)

        idle_w = 1.0 - self._move_w
        flat_w = self._move_w * (1.0 - self._stair_w)
        stair_w = self._move_w * self._stair_w

        weights = {
            AnimStyle.IDLE: idle_w,
            AnimStyle.FLAT_WALK: flat_w,
            AnimStyle.STAIR_CLIMB: stair_w,
        }
        loco_style = (
            AnimStyle.STAIR_CLIMB if stair_target >= 0.5 else AnimStyle.FLAT_WALK
        )
        return StateOutput(weights=weights, locomotion_style=loco_style, moving=effective_moving)


def _approach(current: float, target: float, dt: float, blend_time: float) -> float:
    """Move ``current`` toward ``target`` linearly, reaching it in ``blend_time``."""
    step = dt / blend_time
    if current < target:
        return min(target, current + step)
    if current > target:
        return max(target, current - step)
    return current
