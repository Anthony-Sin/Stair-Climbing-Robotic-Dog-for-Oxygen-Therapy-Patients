"""isaac_env.py extraction (Phase 2 split): terrain_queries. Verbatim bodies; only env_state requalification added."""
import logging
import math
from sim_logging_utils import log_event
from world.sim_go2_locomotion import get_active_stairs

from env import env_state

ROBOT_COLLAPSE_HEIGHT_M = 0.18
# Time constant (s) for easing the VISUAL body height toward the body reference.
# Small => the body tracks the steps crisply (feet stay planted); large => it glides
# but LAGS, and on stairs a lagging body sinks the planted feet INTO the tread after
# each nosing. 0.05 s keeps the lag (hence the foot dip) tiny while still taking the
# hard edge off the per-tread rise.
_PERSON_VISUAL_Z_TAU = 0.05
# Faster tau used ONLY when the discrete-tread target has jumped ~a full riser ahead of
# the eased root (the stairs->landing CREST). At the crest the terrain steps up a full
# riser onto the landing; with the normal 0.05 s tau the visual root lags far enough that
# the planted feet end up ~0.22-0.25 m BELOW the landing (feet cannot reach up past the
# root). Snapping the root up quickly there closes the gap so the planted feet sit ON the
# landing. Small per-tread deltas keep the normal smoothness -- only the big crest jump
# eases fast, so the flat-ground/per-step feel is unchanged.
_PERSON_VISUAL_Z_TAU_CREST = 0.012
_PHYSX_QUERY_IFACE = None
_PHYSX_QUERY_RESOLVED = False
# Prim-path prefixes whose colliders the XT16 raycast must IGNORE. The ray origin sits in the
# robot trunk frame (mount_x=0, ~0.10 m up), so the robot's own body shell AND the O2 payload
# (tank + rails) surround/front it. An unfiltered raycast_closest then returns a constant short
# self-hit (~0.167 m dead ahead, confirmed in run_sim_20260628_110205_860) that hides the real
# target behind it and pins distance fusion to depth-only (98% "depth_disagree"). The robot is
# spawned at GO2_USD_PATH and the payload under /World/O2Payload (rails are children of the
# trunk, i.e. under GO2_USD_PATH), so these two prefixes cover every self collider.
_LIDAR_SELF_PRIM_PREFIXES = (env_state.GO2_USD_PATH, "/World/O2Payload")
_LIDAR_RAYCAST_MODE_LOGGED = False

def get_terrain_height(x: float, y: float) -> float:
    """Return the exact terrain height at coordinate (x, y) based on spawned geometry.

    Discrete tread-top height (snaps one step rise at each tread boundary). This
    is the physically correct value for foot contact, the collider footing, the
    distractor, and idle holds. For the patient's *rendered* root and recorded
    ground-truth Z use get_terrain_height_smooth() instead, which avoids the
    teleport pops the snapped value produces. Geometry comes from the active
    StairSpec (see --stair-preset) so it tracks spawn_obstacles exactly.
    """
    s = get_active_stairs()
    if not (-s.half_width_m <= y <= s.half_width_m):
        return 0.0
    if s.start_x_m <= x < s.end_x_m:
        step_idx = int((x - s.start_x_m) / s.step_depth_m)
        return min(s.top_height_m, (step_idx + 1) * s.step_height_m)
    # Top landing: hold at full stair height
    if x >= s.end_x_m:
        return s.top_height_m
    # Flat ground
    return 0.0

def _collapse_height_threshold(x: float, y: float) -> float:
    """Effective body-height-above-terrain "collapse" floor at (x, y).

    On flat ground this is the plain ROBOT_COLLAPSE_HEIGHT_M. During the staircase
    phase the terrain reference is get_terrain_height()'s DISCRETE tread top, which
    jumps a full riser the instant x crosses a riser edge. The body sits above/behind
    the tread it is stepping onto, so for a moment `rz - terrain_z` reads ~one riser
    too low even on a clean climb (e.g. 0.139 m vs the 0.18 m floor) while tilt never
    exceeds ~27.5 deg. Add a riser-height slack on the stairs (and one riser back /
    forward of the span, to cover the approach/crest samples) so a single-riser
    discretization jump cannot brand a clean climb a "collapse". The genuine fall test
    (low AND tilted past ROBOT_COLLAPSE_TILT_DEG) is unaffected -- this only widens the
    LOW-height half of it on the stairs, where the low reading is a measurement artifact.
    """
    s = get_active_stairs()
    slack = float(s.step_height_m)
    # One tread of margin around the span so the last approach step and the top-riser
    # crest sample (where the discretization jump is largest) are both covered.
    on_or_near_stairs = (
        (-s.half_width_m <= y <= s.half_width_m)
        and (s.start_x_m - s.step_depth_m) <= x <= (s.end_x_m + s.step_depth_m)
    )
    if on_or_near_stairs:
        return ROBOT_COLLAPSE_HEIGHT_M - slack
    return ROBOT_COLLAPSE_HEIGHT_M

def get_terrain_height_smooth(x: float, y: float) -> float:
    """Continuous stair height for the patient's rendered root and ground-truth Z.

    get_terrain_height() snaps Z up one tread rise the instant x crosses each
    tread boundary, which teleported the person (and the recorded GT trajectory)
    up the stairs in discrete pops -- the "jumps 2 stairs / skips steps" symptom.
    A climbing body's pelvis actually rides a continuous slope along the stair
    nosing line, so this returns that slope (top_height of rise over the stair
    run). The result is C0-continuous, so both the climb and the GT data are
    faithful. Feet still contact discrete tread tops via get_terrain_height().
    Geometry comes from the active StairSpec (see --stair-preset).
    """
    s = get_active_stairs()
    if not (-s.half_width_m <= y <= s.half_width_m):
        return 0.0
    run = s.end_x_m - s.start_x_m
    if run > 0.0 and s.start_x_m <= x < s.end_x_m:
        return max(0.0, min(s.top_height_m, (x - s.start_x_m) * (s.top_height_m / run)))
    if x >= s.end_x_m:
        return s.top_height_m
    return 0.0

def _get_person_pose_z(x: float, y: float, *, smooth: bool = True) -> float:
    base_z = get_terrain_height_smooth(x, y) if smooth else get_terrain_height(x, y)
    if env_state._FINAL_SCENE_SPEC is None:
        return base_z
    from final_scene import person_pose_z
    return person_pose_z(base_z, env_state._FINAL_SCENE_SPEC)

def _person_visual_z(state, x: float, y: float, dt: float) -> float:
    """Rendered-root + gait body reference Z: the DISCRETE tread top, eased over time.

    The body reference must sit at the tread the feet stand on, NOT raised: the asset's
    legs stand near-straight (hip->ankle reach is ~98% of full leg length), so they have
    almost no extra reach. Raising the body even half a riser put the treads OUT of reach
    -> the feet could no longer plant and just HOVERED a few cm above every step while the
    swing barely lifted (the "no foot ever in the air / mushy float" look). At the tread
    height the planted foot reaches the step and the swing foot lifts a full ~0.11 m clear,
    i.e. a real alternating step. The two-legs-up artifact is handled by the short easing
    tau (it was the easing LAG, not the body height). Per-foot IK still references each
    foot's own discrete tread. GT Z is on the smooth ramp separately, so GT is unaffected.

    Adaptive crest easing: when the discrete-tread target has stepped up ~a full riser
    ahead of the eased root (the stairs->landing crest), ease with the faster
    _PERSON_VISUAL_Z_TAU_CREST so the root reaches the landing fast enough that the
    planted feet sit ON it (they otherwise end up ~a riser below). Small per-tread deltas
    keep the normal _PERSON_VISUAL_Z_TAU, so flat ground and per-step feel are unchanged.
    """
    target = _get_person_pose_z(x, y, smooth=False)  # discrete tread top (+ final-scene offset)
    delta = target - state.visual_pz
    # Crest detection: fast-ease ONLY when stepping up onto the TOP landing (the last
    # riser -> flat landing), where the accumulated smoothing lag leaves the planted feet
    # ~a riser below the landing. Restrict to the crest by POSITION (last tread onto the
    # landing) AND a real pending step-up, so mid-climb per-tread rises keep the normal
    # smooth tau (and flat ground is never affected). The landing starts at end_x_m; the
    # last tread spans [end_x_m - step_depth_m, end_x_m].
    s = get_active_stairs()
    riser = float(s.step_height_m)
    _at_crest = (
        (-s.half_width_m <= y <= s.half_width_m)
        and x >= (s.end_x_m - s.step_depth_m)
        and delta > 0.4 * riser
    )
    tau = _PERSON_VISUAL_Z_TAU_CREST if _at_crest else _PERSON_VISUAL_Z_TAU
    a = 1.0 if dt <= 0.0 else min(1.0, dt / tau)
    state.visual_pz += a * delta
    return state.visual_pz

def _pgtt_raycast_height(x: float, y: float, origin_z: float) -> float:
    """Terrain-top Z at (x, y) via a PhysX down-ray (PGTT --pgtt-height-backend raycast).

    Mirrors what the real robot's LiDAR elevation map provides: a ray cast straight
    down from above returns the world Z of the first hit. Falls back to the analytic
    ground-truth height if the physics query is unavailable.
    """
    dist = _physx_raycast_distance(
        (float(x), float(y), float(origin_z)), (0.0, 0.0, -1.0), float(origin_z) + 2.0
    )
    if dist is None:
        return get_terrain_height(float(x), float(y))
    return float(origin_z) - float(dist)

def _hit_prim_is_self(path) -> bool:
    """True if a raycast hit's collider/rigid-body prim path is the robot or the O2 payload."""
    if not path:
        return False
    p = str(path)
    return any(p.startswith(prefix) for prefix in _LIDAR_SELF_PRIM_PREFIXES)

def _note_lidar_raycast_mode(mode: str, hits) -> None:
    """Log ONCE which raycast path the XT16 uses + a sample of hit prim paths.

    Observability so the user can confirm in their Isaac run that the self-filter sees the
    real prim paths (and thus actually excludes the body/payload) rather than silently
    no-op'ing because the hit struct exposes no path key on this Isaac build.
    """
    global _LIDAR_RAYCAST_MODE_LOGGED
    if _LIDAR_RAYCAST_MODE_LOGGED:
        return
    _LIDAR_RAYCAST_MODE_LOGGED = True
    try:
        sample = [str(p) for _, p in (hits or [])][:6]
    except Exception:
        sample = []
    log_event(env_state.LOGGER, logging.INFO, "lidar_raycast_self_filter",
              "XT16 raycast self-filter active",
              mode=mode, self_prefixes=list(_LIDAR_SELF_PRIM_PREFIXES),
              sample_hit_prims=sample)

def _get_physx_query_iface():
    """Lazily resolve a PhysX scene-query interface usable for raycasts."""
    global _PHYSX_QUERY_IFACE, _PHYSX_QUERY_RESOLVED
    if _PHYSX_QUERY_RESOLVED:
        return _PHYSX_QUERY_IFACE
    _PHYSX_QUERY_RESOLVED = True
    try:
        from omni.physx import get_physx_scene_query_interface
        _PHYSX_QUERY_IFACE = get_physx_scene_query_interface()
    except Exception:
        try:
            import omni.physx
            _PHYSX_QUERY_IFACE = omni.physx.get_physx_interface()
        except Exception as exc:
            log_event(env_state.LOGGER, logging.WARNING, "lidar_physx_iface_missing",
                      "No PhysX scene-query interface available; XT16 LiDAR will return no hits",
                      error=str(exc))
            _PHYSX_QUERY_IFACE = None
    return _PHYSX_QUERY_IFACE

def _physx_raycast_distance(origin, direction, max_dist):
    """raycast_fn for sim_lidar_xt16: nearest NON-self hit distance, or None.

    The XT16 ray origin is in the robot trunk frame, so the robot's own body shell and the O2
    payload colliders surround/front it. raycast_closest alone returns that self-hit (a constant
    ~0.167 m) and hides the real target behind it, pinning distance fusion to depth-only. So
    prefer raycast_all and return the nearest hit whose collider is NOT the robot/payload
    (_hit_prim_is_self); fall back to a self-filtered raycast_closest when raycast_all is absent.

    Tolerates the different shapes the PhysX query returns across Isaac builds (dict with
    hit/distance/position/rigidBody, or a (hit_bool, hit_info) tuple / struct attributes).
    """
    iface = _get_physx_query_iface()
    if iface is None:
        return None

    o = (float(origin[0]), float(origin[1]), float(origin[2]))
    d = (float(direction[0]), float(direction[1]), float(direction[2]))
    md = float(max_dist)

    def _from_position(pos):
        dx = float(pos[0]) - o[0]
        dy = float(pos[1]) - o[1]
        dz = float(pos[2]) - o[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # Preferred path: collect ALL hits along the ray and skip self-colliders, so the body /
    # payload sitting in front of the origin no longer masks the real return behind it.
    raycast_all = getattr(iface, "raycast_all", None)
    if raycast_all is not None:
        hits = []

        def _report(hit):
            try:
                dist = getattr(hit, "distance", None)
                path = getattr(hit, "rigid_body", None) or getattr(hit, "collision", None)
                if dist is None and isinstance(hit, dict):
                    dist = hit.get("distance")
                    path = hit.get("rigidBody") or hit.get("collision")
                if dist is not None:
                    hits.append((float(dist), path))
            except Exception:
                pass
            return True  # keep collecting all hits

        try:
            raycast_all(o, d, md, _report)
        except Exception:
            hits = []
        # Only trust raycast_all when it actually produced usable (distance-bearing) hits.
        # If it yielded nothing (a genuine miss OR an unexpected callback signature on this
        # Isaac build), fall through to raycast_closest rather than blanking the whole LiDAR.
        if hits:
            _note_lidar_raycast_mode("raycast_all", hits)
            best = None
            for dist, path in hits:
                if _hit_prim_is_self(path):
                    continue
                if best is None or dist < best:
                    best = dist
            if best is not None:
                return best
            # Every hit was self -> the real target (if any) is masked; try closest below as a
            # self-filtered second opinion (returns None on a pure self-hit).

    # Fallback: closest hit, self-filtered. Returning None on a forward self-hit makes the ray a
    # clean miss (fusion sees depth_only, not a false 0.167 m depth_disagree).
    try:
        hit = iface.raycast_closest(o, d, md)
    except Exception:
        return None
    if not hit:
        return None

    if isinstance(hit, dict):
        if not hit.get("hit"):
            return None
        if _hit_prim_is_self(hit.get("rigidBody") or hit.get("collision")):
            return None
        if hit.get("distance") is not None:
            return float(hit["distance"])
        if hit.get("position") is not None:
            return _from_position(hit["position"])
        return None
    if isinstance(hit, (list, tuple)) and len(hit) >= 2:
        if not hit[0]:
            return None
        info = hit[1]
        path = getattr(info, "rigid_body", None) or getattr(info, "collision", None)
        if path is None and isinstance(info, dict):
            path = info.get("rigidBody") or info.get("collision")
        if _hit_prim_is_self(path):
            return None
        dist = getattr(info, "distance", None)
        if dist is not None:
            return float(dist)
        pos = getattr(info, "position", None)
        if pos is None and isinstance(info, dict):
            pos = info.get("position")
        if pos is not None:
            return _from_position(pos)
    return None
