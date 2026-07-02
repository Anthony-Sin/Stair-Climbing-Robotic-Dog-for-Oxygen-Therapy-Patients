"""Reference the final-scene environment (Hospital USD + realistic staircase) onto
the live Isaac stage.

This is the only Isaac-Sim-dependent module. ``pxr`` and the Isaac stage/nucleus
helpers are imported lazily inside :func:`attach_final_scene` so the rest of the
package (spec / asset generation) stays importable under plain Python.

Call ``attach_final_scene(stage, world)`` once, AFTER the stage exists and BEFORE
``world.reset()`` (same timing rule as ``o2_payload.attach_o2_payload``), so PhysX
ingests the hospital + staircase as part of the initial scene. The robot, patient,
cameras and control stack are spawned by ``isaac_env`` exactly as in the default
sim -- this module only swaps in the upgraded environment.

No silent fallback: if no Hospital USD candidate resolves, this raises so the run
fails loudly (per the project's "no fakes" rule) rather than running on a bare
floor that looks like the upgrade succeeded.

Structure note: the implementation was split into cohesive sibling modules
(``common`` / ``geometry`` / ``cinematic`` / ``cameras`` / ``mount``). This module
is now a thin facade that re-exports every previously-top-level name so existing
importers (``from final_scene.isaac_mount import ...``) keep working unchanged.
"""

from __future__ import annotations

# Stdlib / typing names that were previously top-level in this module, re-exported
# so ``isaac_mount.os`` / ``isaac_mount.math`` / ``isaac_mount.Optional`` etc. still
# resolve exactly as before.
import logging  # noqa: F401
import math  # noqa: F401
import os  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from typing import Callable, Dict, List, Optional, Tuple  # noqa: F401

# Spec names were imported at module top level in the original file; keep them
# importable from this module path.
from ..spec import SPEC, FinalSceneSpec, WallCameraSpec  # noqa: F401

from .common import (  # noqa: F401
    LogFn,
    Vec3,
    _LOGGER,
    _default_log,
)
from .geometry import (  # noqa: F401
    _add,
    _apply_transform,
    _as_vec3,
    _asset_uri,
    _clamp,
    _damp,
    _length,
    _normalize,
    _round_vec3,
    _scale,
    _set_camera_look_at,
    _sub,
)
from .cinematic import (  # noqa: F401
    CinematicDirector,
    _CameraState,
    _MAX_LOOKAHEAD_M,
    _OCCLUSION_MARGIN_M,
    _PATIENT_OCCLUSION_RADIUS_M,
    _RAYCAST_ORIGIN_BIAS_M,
    _ROBOT_FRAME_HEIGHT_M,
    _ROBOT_TORSO_MIN_HEIGHT_M,
)
from .cameras import (  # noqa: F401
    _CINEMATIC_DIRECTOR,
    _CINEMATIC_DIRECTOR_SPEC_ID,
    _ensure_wall_mount_visual,
    create_wall_recording_camera,
    update_wall_recording_cameras,
)
from .mount import (  # noqa: F401
    STAIRCASE_USDA,
    FinalSceneHandle,
    _ASSETS_DIR,
    _hide_default_ground_visual,
    _import_isaac,
    _resolve_hospital_usd,
    attach_final_scene,
    hide_stair_collision_visuals,
)
