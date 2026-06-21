"""Procedural, limb-driven animation for the BipedMannequin patient character.

Replaces the old baked walk/idle SkelAnimation clip playback (which slid a canned
loop along the waypoints and faked stairs with a lean+bob) with a modular gait
system that drives the rig's actual hip/knee/ankle/shoulder/elbow/spine joints,
differentiates flat vs stair terrain, and syncs the gait cycle to real travel.

Modules:
  types               -- shared enums + JointPose (no Isaac deps)
  terrain_classifier  -- FLAT vs STAIR from (x, y) + active stair geometry
  locomotion_controller -- travel -> gait phase + smoothed speed
  gait                -- FlatWalk / StairClimb / Idle profiles -> JointPose
  state_machine       -- pick & crossfade gait styles (debounced)
  rig                 -- UsdSkel adapter: derives axes, binds + writes joints
  controller          -- BipedAnimationController facade (one update() per frame)

Build the runtime controller with ``build_biped_animation_controller`` from a live
stage + SkelRoot path; the pure-logic pieces can be imported and unit-tested on a
plain host without Isaac.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .controller import BipedAnimationController
from .gait import FlatWalk, Gait, GaitParams, Idle, StairClimb
from .locomotion_controller import LocomotionController
from .state_machine import AnimationStateMachine
from .terrain_classifier import StairGeometry, TerrainClassifier
from .types import AnimStyle, JointPose, TerrainClass

__all__ = [
    "AnimStyle",
    "AnimationStateMachine",
    "BipedAnimationController",
    "FlatWalk",
    "Gait",
    "GaitParams",
    "Idle",
    "JointPose",
    "LocomotionController",
    "StairClimb",
    "StairGeometry",
    "TerrainClass",
    "TerrainClassifier",
    "build_biped_animation_controller",
]


def build_biped_animation_controller(
    stage,
    skel_root_path: str,
    stairs_provider: Callable[[], object],
    *,
    logger: Optional[logging.Logger] = None,
    pump_on_apply: bool = False,
) -> Optional[BipedAnimationController]:
    """Construct the full runtime controller from a live stage + SkelRoot.

    Returns ``None`` if the rig could not be initialized (the caller can then leave
    the character in its rest pose). Importing ``rig`` is deferred so the pure
    modules above stay importable without Isaac/USD present.
    """
    from .rig import BipedRig

    rig = BipedRig(stage, skel_root_path, logger=logger, pump_on_apply=pump_on_apply)
    if not rig.ready:
        return None
    classifier = TerrainClassifier(stairs_provider)
    return BipedAnimationController(rig, classifier=classifier, logger=logger)
