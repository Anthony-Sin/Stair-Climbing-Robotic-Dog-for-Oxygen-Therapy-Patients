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
import time
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
        clock: Callable[[], float] = time.monotonic,
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
        self._clock = clock
        self._logger = logger
        self._last_speed = 0.0
        self._last_pose: Optional[JointPose] = None
        self._last_terrain = TerrainClass.FLAT
        # Throttled verify-via-logs diagnostic (one line every _diag_period sec).
        self._last_diag_t = -1e9
        self._diag_period = 1.5

    def reset(self, xy) -> None:
        self._loco.reset((float(xy[0]), float(xy[1])), self._clock())

    def update(self, x: float, y: float, *, moving_hint: Optional[bool] = None) -> None:
        if self._rig is None or not getattr(self._rig, "ready", False):
            return

        now = self._clock()
        terrain = self._classifier.classify(x, y)
        geom = self._classifier.stair_geometry()
        self._last_terrain = terrain

        # Stride for phase advance comes from the gait the terrain calls for, using
        # the last known speed (the gait that will dominate this frame).
        target_style = (
            AnimStyle.STAIR_CLIMB if terrain == TerrainClass.STAIR else AnimStyle.FLAT_WALK
        )
        stride = self._gaits[target_style].stride_length(self._last_speed, geom)

        loco = self._loco.update((x, y), now, stride)
        self._last_speed = loco.speed
        moving = loco.moving if moving_hint is None else bool(moving_hint)

        state = self._sm.update(moving, terrain, now)

        # Blend the active gait poses by their state-machine weights.
        contributions = []
        for style, weight in state.weights.items():
            if weight <= 1e-4:
                continue
            gait = self._gaits[style]
            pose = gait.evaluate(loco.phase, loco.speed, geom)
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
