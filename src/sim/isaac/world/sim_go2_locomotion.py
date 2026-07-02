"""Staircase geometry and passive Go2 body/stair telemetry.

Despite the module name, this is NOT a locomotion controller. It owns two things:
  1. The load-bearing ``StairSpec`` geometry (presets + ``configure_stairs`` +
     ``get_active_stairs``) -- the single source of truth every stair consumer
     reads so the spawned scene and ground-truth telemetry stay in sync.
  2. The decorative "stair_demo" HUD telemetry, rebuilt each step from the
     robot's measured body pose. It is observation only and drives no physics.

This module is now a thin re-export facade: the implementation was split by
responsibility into the sibling ``sim_go2_state`` (the ``Go2LocomotionState``
struct + rate-limited logging helper), ``sim_go2_body_pose`` (base-link
rigid-body discovery + pose/orientation math), ``sim_go2_stairs`` (``StairSpec``
presets, the mutable ``ACTIVE_STAIRS`` global, the analytical terrain model, and
the stair-demo HUD-telemetry builder), and ``sim_go2_telemetry`` (the passive
body-telemetry recorder + ``record_go2_telemetry`` entry point) modules.
Everything that was previously a top-level name here is re-exported below so
``from world.sim_go2_locomotion import ...`` keeps working.
"""
# Locomotion is driven by the parkour depth/vision policy in
# parkour_locomotion_policy.py. This module now only owns the stair-demo
# perception/telemetry and the analytical terrain model used to build that
# telemetry from the robot's measured body pose.

# ArticulationAction shim kept for parity with the historical import surface
# (Phase-1 cleanup removed its only use, but the shim is an intentional keep);
# not used by the split code itself.
try:
    from omni.isaac.core.utils.types import ArticulationAction  # noqa: F401
except ModuleNotFoundError:
    from isaacsim.core.utils.types import ArticulationAction  # noqa: F401

# --- Telemetry/bookkeeping state struct + rate-limited logging helper -----------------
from world.sim_go2_state import (  # noqa: F401
    Go2LocomotionState,
    _warn_rate_limited,
)

# --- Base-link rigid-body discovery + pose/orientation math ---------------------------
from world.sim_go2_body_pose import (  # noqa: F401
    _attr_is_valid,
    _prim_has_rigid_body,
    _prim_is_kinematic,
    _candidate_rigid_body_prims,
    _find_rigid_body_api,
    _extract_roll_pitch_yaw,
    _clamp,
)

# --- Staircase geometry (source of truth) + analytical terrain model + stair-demo HUD -
# NOTE: ``ACTIVE_STAIRS`` re-exported here is a SNAPSHOT bound at facade-import time.
# ``configure_stairs`` reassigns ``sim_go2_stairs.ACTIVE_STAIRS`` (not this alias), so
# call ``get_active_stairs()`` for the live spec -- do not read the facade attribute.
from world.sim_go2_stairs import (  # noqa: F401
    StairSpec,
    STAIR_PRESETS,
    ACTIVE_STAIRS,
    configure_stairs,
    get_active_stairs,
    STAIR_LIDAR_LOOKAHEAD_M,
    _terrain_phase,
    _next_stair_edge,
    _build_stair_demo_telemetry,
    _effective_swing_height_for_mode,
    _build_leg_command_summary,
    _get_analytical_terrain_height,
)

# --- Passive body-telemetry recorder + public per-step entry point --------------------
from world.sim_go2_telemetry import (  # noqa: F401
    _record_stair_demo_telemetry,
    get_stair_demo_telemetry,
    _query_terrain_height,
    _record_passive_body_telemetry,
    record_go2_telemetry,
)
