"""BipedAnimationController: the facade that wires the modular pieces together.

The caller (``sim_person_actor.SimPersonTarget``) feeds it the patient's current
world (x, y) and whether the waypoint follower considers it moving; the controller
classifies the terrain, advances the gait phase from real travel, picks/blends the
gait style and applies the resulting limb pose to the rig -- once per frame.

Each collaborator is injected and independently swappable: pass a different
``TerrainClassifier``, ``LocomotionController``, ``AnimationStateMachine`` or gait
set without touching this class, and the rig adapter can be replaced wholesale for
a different character.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Optional

from .gait import FlatWalk, Gait, Idle, StairClimb
from .locomotion_controller import LocomotionController
from .state_machine import AnimationStateMachine
from .terrain_classifier import TerrainClassifier
from .types import AnimStyle, JointPose, TerrainClass


class BipedAnimationController:
    def __init__(
        self,
        rig,
        *,
        classifier: TerrainClassifier,
        locomotion: Optional[LocomotionController] = None,
        state_machine: Optional[AnimationStateMachine] = None,
        gaits: Optional[Dict[AnimStyle, Gait]] = None,
        clock: Optional[Callable[[], float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._rig = rig
        self._classifier = classifier
        self._loco = locomotion or LocomotionController()
        self._sm = state_machine or AnimationStateMachine()
        self._gaits: Dict[AnimStyle, Gait] = gaits or {
            AnimStyle.IDLE: Idle(),
            AnimStyle.FLAT_WALK: FlatWalk(),
            AnimStyle.STAIR_CLIMB: StairClimb(),
        }
        self._clock = clock if clock is not None else (lambda: 0.0)
        self._logger = logger
        self._last_speed = 0.0
        self._last_pose: Optional[JointPose] = None
        self._last_terrain = TerrainClass.FLAT
        # Heading unit vector (travel direction), so a foot's forward offset maps to
        # the right world (x, y) for the ground sample. Defaults to +x until the
        # patient has moved enough to establish a direction.
        self._heading = (1.0, 0.0)
        self._last_xy_for_heading: Optional[tuple] = None
        # Throttled verify-via-logs diagnostic (one line every _diag_period sec).
        self._last_diag_t = -1e9
        self._diag_period = 1.5

    def reset(self, xy, current_time: float = 0.0) -> None:
        self._loco.reset((float(xy[0]), float(xy[1])), current_time)
        self._heading = (1.0, 0.0)
        self._last_xy_for_heading = (float(xy[0]), float(xy[1]))

    def _update_heading(self, x: float, y: float) -> None:
        """Track the travel direction so foot offsets map to world (x, y)."""
        last = self._last_xy_for_heading
        if last is not None:
            dx = x - last[0]
            dy = y - last[1]
            d = math.hypot(dx, dy)
            if d > 1e-4:  # only update on real movement (ignore jitter)
                self._heading = (dx / d, dy / d)
        self._last_xy_for_heading = (x, y)

    def update(
        self,
        x: float,
        y: float,
        *,
        moving_hint: Optional[bool] = None,
        body_z: Optional[float] = None,
        ground_height_fn: Optional[Callable[[float, float], float]] = None,
        current_time: Optional[float] = None,
    ) -> None:
        if self._rig is None or not getattr(self._rig, "ready", False):
            return

        now = self._clock() if current_time is None else float(current_time)
        terrain = self._classifier.classify(x, y)
        geom = self._classifier.stair_geometry()
        self._last_terrain = terrain

        # Stride for phase advance comes from the gait the terrain calls for, using
        # the last known speed (the gait that will dominate this frame).
        target_style = (
            AnimStyle.STAIR_CLIMB if terrain == TerrainClass.STAIR else AnimStyle.FLAT_WALK
        )
        stride = self._gaits[target_style].stride_length(self._last_speed, geom)

        self._update_heading(x, y)
        loco = self._loco.update((x, y), now, stride)
        self._last_speed = loco.speed
        moving = loco.moving if moving_hint is None else bool(moving_hint)

        state = self._sm.update(moving, terrain, now)

        # Ground-referenced foot placement: given a foot's forward offset, return how
        # much further DOWN than the standing reach it must go to land on the actual
        # ground/tread under it (body_z minus the ground height at that foot). Without
        # body_z + a height function this is None and the gait uses its heuristic.
        ground_sampler = None
        if body_z is not None and ground_height_fn is not None:
            hx, hy = self._heading
            bz = float(body_z)

            def _sample_ground(s, _bz=bz, _hx=hx, _hy=hy, _gx=x, _gy=y, _fn=ground_height_fn):
                return _bz - float(_fn(_gx + s * _hx, _gy + s * _hy))

            ground_sampler = _sample_ground

        # Blend the active gait poses by their state-machine weights.
        contributions = []
        for style, weight in state.weights.items():
            if weight <= 1e-4:
                continue
            gait = self._gaits[style]
            pose = gait.evaluate(loco.phase, loco.speed, geom, ground_sampler=ground_sampler)
            contributions.append((pose, weight))
        pose = JointPose.weighted_sum(contributions) if contributions else JointPose.zero()

        self._last_pose = pose
        self._rig.apply(pose)

        self._maybe_log(now, x, y, terrain, state, loco, pose)

    def _maybe_log(self, now, x, y, terrain, state, loco, pose) -> None:
        if self._logger is None or (now - self._last_diag_t) < self._diag_period:
            return
        self._last_diag_t = now
        try:
            from sim_logging_utils import log_event
            import logging as _logging

            dominant = max(state.weights.items(), key=lambda kv: kv[1])[0]
            log_event(
                self._logger,
                _logging.INFO,
                "biped_gait_state",
                "Procedural patient gait state (throttled)",
                terrain=terrain.value,
                dominant_style=dominant.value,
                weights={k.value: round(float(v), 2) for k, v in state.weights.items()},
                phase=round(float(loco.phase), 3),
                speed_m_s=round(float(loco.speed), 3),
                moving=bool(state.moving),
                x=round(float(x), 3),
                y=round(float(y), 3),
                hip_l=round(float(pose.hip_l), 3),
                knee_l=round(float(pose.knee_l), 3),
                shoulder_l=round(float(pose.shoulder_l), 3),
                spine_pitch=round(float(pose.spine_pitch), 3),
            )
        except Exception:
            pass

    # -- introspection (for diagnostics / external HUD) ---------------------- #
    @property
    def last_speed(self) -> float:
        return self._last_speed

    @property
    def last_terrain(self) -> TerrainClass:
        return self._last_terrain

    @property
    def phase(self) -> float:
        return self._loco.phase

    def set_gait_phase(self, val: float) -> None:
        self._loco.phase = val
