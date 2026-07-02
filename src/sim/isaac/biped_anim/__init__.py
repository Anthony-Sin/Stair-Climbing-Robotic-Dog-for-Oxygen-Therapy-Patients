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
    anim_mode: str = "clip",
) -> Optional[BipedAnimationController]:
    """Construct the full runtime controller from a live stage + SkelRoot.

    ``anim_mode``:
      * ``"clip"`` (default) -- play the character's baked walk ``SkelAnimation`` as
        full per-bone mocap on flat ground (realistic), falling back to the analytic
        gait for any style without a clip (e.g. stair-climb until a stair clip is
        supplied). This is "stop discarding the realistic clip the asset already has".
      * ``"procedural"`` -- the analytic 13-DOF foot-planting gait everywhere (the
        prior behaviour; kept as a fallback).

    Returns ``None`` if the rig could not be initialized (the caller can then leave
    the character in its rest pose). Importing ``rig`` is deferred so the pure
    modules above stay importable without Isaac/USD present.
    """
    from .clip_player import ClipPlayer
    from .rig import BipedRig

    rig = BipedRig(stage, skel_root_path, logger=logger, pump_on_apply=pump_on_apply)
    if not rig.ready:
        return None
    classifier = TerrainClassifier(stairs_provider)

    # Build the walking gaits with the rig's measured leg proportions so they drive
    # FOOT-PLANTING IK (planted stance, no skate). If the geometry could not be
    # measured, leave gaits as None so the controller uses its open-loop defaults.
    geom = rig.leg_geometry
    gaits = None
    if geom is not None:
        gaits = {
            AnimStyle.IDLE: Idle(),
            AnimStyle.FLAT_WALK: FlatWalk(leg_geom=geom),
            AnimStyle.STAIR_CLIMB: StairClimb(leg_geom=geom),
        }

    # Mocap-clip playback: extract the asset's baked walk clip and register it for the
    # flat-walk style. If no clip is found the player stays empty and the controller
    # uses the analytic gait everywhere (graceful fallback).
    clip_player = None
    if str(anim_mode).lower() == "clip":
        walk_clip = rig.extract_clip_tracks()
        if walk_clip is not None:
            clip_player = ClipPlayer({AnimStyle.FLAT_WALK: walk_clip})

    # Stair-climb MOCAP clip: a real human "walk up stairs" motion (CMU subject 83,
    # assets/mocap/83_27.fbx) drives the STAIR_CLIMB style via per-frame anatomical
    # angles -- replacing the synthetic procedural stair gait that read as gliding. If
    # the FBX is missing/unparseable the analytic StairClimb gait is used (graceful).
    mocap_pose_clips = None
    if str(anim_mode).lower() == "clip":
        try:
            import os
            from .mocap_clip import build_stair_clip
            _fbx = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "assets", "mocap", "83_27.fbx")
            if os.path.exists(_fbx):
                _stair = build_stair_clip(_fbx)
                if _stair is not None:
                    mocap_pose_clips = {AnimStyle.STAIR_CLIMB: _stair}
                    if logger is not None:
                        from sim_logging_utils import log_event
                        log_event(logger, logging.INFO, "patient_stair_mocap_loaded",
                                  "Loaded CMU stair-climb mocap clip for the STAIR_CLIMB style",
                                  fbx=_fbx, loop_frames=int(_stair.win_len))
        except Exception as _e:
            if logger is not None:
                try:
                    from sim_logging_utils import log_event
                    log_event(logger, logging.WARNING, "patient_stair_mocap_failed",
                              "Could not load the stair mocap clip; using the analytic stair gait",
                              error=str(_e))
                except Exception:
                    pass

    return BipedAnimationController(
        rig, classifier=classifier, gaits=gaits, clip_player=clip_player,
        mocap_pose_clips=mocap_pose_clips, logger=logger
    )
