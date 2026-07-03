"""Spawns and animates the simulated patient (BipedMannequin) in the Isaac scene.

Resolves and caches the modified ``Biped_Setup`` USD, wires it to the procedural
``biped_anim`` gait controller, and exposes the ``SimPersonTarget`` pose driver
used by the follow/handoff logic. The low-level UsdSkel animation-channel surgery
(root-motion zeroing, walk-clip looping, gait-period estimation) lives in the
sibling ``skel_anim_utils`` module.

This module is now a thin re-export facade: the implementation was split by
responsibility into the sibling ``sim_person_config`` (constants), ``sim_person_xform``
(pose helpers), ``sim_person_asset`` (USD asset resolution + extension bootstrap), and
``sim_person_driver`` (the ``SimPersonTarget`` pose driver + ``spawn_sim_person``)
modules. Everything that was previously a top-level name here is re-exported below so
``from world.sim_person_actor import ...`` keeps working. The patient is a kinematic
UsdSkel character posed by the procedural gait -- there is no physics puppet.
"""
# DynamicCapsule shim kept for parity with the historical import surface (some call
# sites/back-compat referenced it via this module); not used by the split code itself.
try:
    from omni.isaac.core.objects import DynamicCapsule  # noqa: F401
except ModuleNotFoundError:
    from isaacsim.core.api.objects import DynamicCapsule  # noqa: F401

# UsdSkel animation-channel surgery and the walk-cadence constant (re-exported for
# parity; the asset module imports them directly for its own use).
from world.skel_anim_utils import (  # noqa: F401
    _PERSON_GAIT_CADENCE_MULT,
    _zero_root_translation_channel,
    _zero_root_rotation_channel,
    _loop_animation_channels,
    _estimate_gait_period,
)

# --- Static configuration / constants -------------------------------------------------
from world.sim_person_config import (  # noqa: F401
    CHARACTER_PARENT_PRIM,
    PERSON_VISUAL_PRIM,
    PERSON_COLLIDER_PRIM,
    ANIMATED_CHARACTERS,
    BIPED_SETUP_PRIM,
    _BIPED_WALK_ANIM_SUBPATH,
    _BIPED_IDLE_ANIM_SUBPATH,
    PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD,
    PERSON_IDLE_DEBOUNCE_SEC,
    _BIPED_SETUP_USD_CANDIDATES,
    _BIPED_MODIFIED_CACHE_VERSION,
)

# --- USD xform/pose helpers -----------------------------------------------------------
from world.sim_person_xform import (  # noqa: F401
    _set_xform_pose,
    _yaw_quat_for_orient_op,
    _find_first_skel_root,
)

# --- Character-asset resolution + animation-extension bootstrap -----------------------
from world.sim_person_asset import (  # noqa: F401
    _load_biped_setup,
    _resolve_custom_character,
    _resolve_character_with_clips,
    _initialize_extensions,
    _EXTENSIONS_READY,
    _EXTENSION_CHECK_DONE,
)

# --- The patient pose driver + spawn entrypoint ---------------------------------------
from world.sim_person_driver import (  # noqa: F401
    SimPersonTarget,
    _start_timeline_and_pump,
    spawn_sim_person,
    _char_usd_cache,
    _char_name_cache,
    _walk_clip_cache,
    _skel_root_path_cache,
    _idle_clip_cache,
)
