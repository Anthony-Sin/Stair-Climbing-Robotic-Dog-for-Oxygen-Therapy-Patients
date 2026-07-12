"""Living-room scene variant: household furniture + a winding patient route.

Opt-in via ``--living-room`` (isaac_args) / ``SIM_LIVING_ROOM=1`` (run_living_room.bat).
It restages the SAME proven robot / patient / stair-climb stack from the default sim
(pgtt walk + blind_rl climb -- the policy pair that actually reaches the top landing)
inside a furnished living room: nine collidable household props (two sofas, coffee table,
bookshelf, armchair, TV console, potted plant, end table, floor lamp) -- every one a REAL
ArchVis mesh, no plain boxes -- sit on the flat approach floor. THREE of them stand IN the
lane, spaced ~2 m apart, and the patient walks a realistic THREE-bend slalom that weaves
AROUND them (+Y, -Y, +Y) before rejoining the stair centreline and climbing -- instead of
the old straight line / simple left-right zigzag (``--person-approach-turns``).

Design contract (matches ``final_scene``):
  * The staircase base is FIXED at x = 2.0 (``StairSpec.start_x_m``); every prop and
    every turn of the route stays in front of it (x < 2.0) so the patient only enters
    the stair lane centred (y = 0), exactly as the default path does. The per-tread +
    top-landing stair waypoints are appended by the caller (``PatientLocomotionState``),
    so the stair-climb pathing is UNCHANGED.
  * The patient spawns at (~-3.5, 0) and the robot behind it (~-3.0, 0), so there is
    ~5.5 m x ~5 m of flat living-room floor for the obstacle course.

Coordinate frame (world metres): +X forward (toward the stairs), +Y left, floor Z = 0.
Furniture is defined by its footprint CENTRE and full size; it rests on the floor
(centre Z = height/2).

This module is import-safe under the *system* Python (no ``pxr`` / Isaac / numpy at
module scope) so ``python living_room.py`` validates the layout+route geometry
(segment-to-furniture clearance) BEFORE paying for an Isaac run. The Isaac-only spawn
imports live inside ``spawn_living_room``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Furniture: axis-aligned boxes resting on the floor. cx/cy = footprint centre,
# sx/sy = full footprint size (X depth, Y width), sz = height. color is RGB 0..1.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Furniture:
    name: str
    cx: float
    cy: float
    sx: float
    sy: float
    sz: float
    color: Vec3

    @property
    def x_min(self) -> float:
        return self.cx - self.sx / 2.0

    @property
    def x_max(self) -> float:
        return self.cx + self.sx / 2.0

    @property
    def y_min(self) -> float:
        return self.cy - self.sy / 2.0

    @property
    def y_max(self) -> float:
        return self.cy + self.sy / 2.0


# Household furniture. Two "gate" pieces stand IN the centre lane (not along the
# walls) and force the patient's right-then-left weave; the rest are a real seating
# group + media wall + entry nook that frame the room. Heights and footprints are
# roughly life-sized. All are collidable (FixedCuboid) so the run is a genuine
# obstacle course. Proven policy pair: pgtt walk + blind_rl climb (see run_sim.ps1).
FURNITURE: Tuple[Furniture, ...] = (
    # A REAL, lived-in room, given ROOM to breathe (the earlier four-prop course was too cramped
    # -- the dog cut the corner INTO the first prop, wedged, and toppled). THREE compact obstacles
    # stand IN the lane (near the centreline), spaced ~2.0 m apart so the patient weaves a clean
    # slalom AROUND them and the trailing dog has room to thread the same gaps. Every prop here is
    # a REAL ArchVis mesh (see FURNITURE_USD) -- no plain boxes. Because the dog trails near the
    # centreline (it does NOT copy the patient's weave) it would otherwise drive into an in-lane
    # prop, so it steers around them with its depth obstacle-avoidance (run with
    # SIM_AVOID_OBSTACLES=1; avoidance is hard-latched OFF once the stairs are seen, so the pgtt
    # walk + blind_rl climb is unaffected). The rest frame the room along the two walls + entry
    # corners, well off the lane. The last leg is a straight, CENTRED run-up to the first riser.
    # validate_layout checks that BOTH the kinematic patient AND a simulated trailing-dog pursuit
    # path clear every prop by >= CLEARANCE_MIN_M / DOG_CLEARANCE_MIN_M.
    # --- IN-PATH obstacles (all real meshes): three compact props sit ON the lane, one per bend,
    #     each offset to the side the path curves AWAY from so its near edge stays ~0.15 m off the
    #     centreline -- room for the corner-cutting dog to pass on the open side.
    Furniture("coffee_table", -4.20, -0.52, 0.90, 0.50, 0.42, (0.42, 0.28, 0.15)), # obstacle 1: path rounds its +Y side
    Furniture("plant",        -2.20,  0.46, 0.45, 0.45, 1.15, (0.18, 0.42, 0.20)), # obstacle 2: path rounds its -Y side
    Furniture("end_table",    -0.20, -0.48, 0.55, 0.55, 0.52, (0.40, 0.26, 0.14)), # obstacle 3: path rounds its +Y side
    # --- framing along the two walls + entry corners (well off the lane; avoidance ignores them) ---
    Furniture("sofa_wall",    -3.00, -2.35, 2.30, 0.85, 0.75, (0.34, 0.30, 0.28)), # long couch along the -Y wall
    Furniture("armchair",     -4.90, -2.20, 0.90, 0.90, 0.80, (0.34, 0.42, 0.32)), # accent chair, -Y entry
    Furniture("floor_lamp",   -5.65, -2.45, 0.28, 0.28, 1.55, (0.78, 0.74, 0.62)), # floor lamp, -Y corner
    Furniture("sofa_center",  -2.60,  2.35, 1.90, 0.90, 0.75, (0.60, 0.52, 0.40)), # loveseat on the +Y wall
    Furniture("tv_console",    0.75,  2.35, 1.80, 0.40, 0.55, (0.14, 0.14, 0.16)), # TV unit, +Y wall toward the stairs
    Furniture("bookshelf",    -5.55,  2.25, 0.90, 0.45, 1.80, (0.28, 0.18, 0.10)), # bookshelf, +Y entry corner
)


# Realistic Omniverse ArchVis Residential furniture meshes, one per prop
# (paths S3-verified 2026-07-09). Referenced onto the stage when they resolve;
# each is auto-oriented (up-axis read at runtime) and uniformly scaled to FIT
# INSIDE its Furniture footprint box, then placed on the floor at the box centre.
# The COLLISION is always the same invisible footprint box as the plain-box
# variant, so the validated route + depth-avoidance envelope are provably
# unchanged -- the mesh is a purely visual skin. Any piece whose asset fails to
# resolve/compose falls back to its visible coloured box, and
# SIM_LIVING_ROOM_BOXES=1 forces the plain-box variant for the whole scene.
# Paths are relative to the ArchVis root (the sibling of the Isaac assets root,
# e.g. ".../Assets/ArchVis/...").
FURNITURE_USD = {
    "coffee_table": "ArchVis/Residential/Furniture/CoffeeTables/Midtown.usd",
    "sofa_center":  "ArchVis/Residential/Furniture/Sofas/Moline.usd",
    "sofa_wall":    "ArchVis/Residential/Furniture/Sofas/Arnold.usd",
    "bookshelf":    "ArchVis/Residential/Furniture/Bookshelves/Delmar.usd",
    "armchair":     "ArchVis/Residential/Furniture/Chairs/Armchair.usd",
    "tv_console":   "ArchVis/Residential/Furniture/MediaTables/Manchester.usd",
    "end_table":    "ArchVis/Residential/Furniture/EndTables/Ellendale.usd",
    "plant":        "ArchVis/Residential/Plants/Plant_01.usd",
    "floor_lamp":   "ArchVis/Residential/Lighting/Floor Lamps/BrassFloorLamp.usd",
}

# ArchVis assets are not version-forked (unlike Isaac/<ver>/...), so a single CDN
# fallback URL mirrors the Isaac-assets-root resolution the person/hospital loaders use.
_ARCHVIS_CDN_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"


# ---------------------------------------------------------------------------
# Winding patient route: flat legs BEFORE the stairs that weave around the gate
# furniture, then rejoin the stair centreline. The spawn point is prepended and the
# final leg is snapped to just in front of the staircase base at runtime.
# ---------------------------------------------------------------------------
# Interior legs only (spawn prepended, stair approach snapped in build_living_room_route).
ROUTE_LEGS: Tuple[Vec2, ...] = (
    (-4.20,  0.55),  # bend 1: detour +Y (left) around the coffee table in the lane
    (-2.20, -0.55),  # bend 2: detour -Y (right) around the potted plant in the lane
    (-0.20,  0.55),  # bend 3: detour +Y around the end table in the lane
    (1.00,   0.00),  # settle back onto the centreline past the last obstacle
    (1.90,   0.00),  # straight, CENTRED, head-on run-up to the stairs (snapped to the riser at runtime)
)
# NOTE 1: a genuine THREE-bend slalom (+Y, -Y, +Y) -- the patient detours AROUND one in-lane
# obstacle per bend, a clearly winding path, NOT "mostly straight". The bends are spaced ~2.0 m
# apart in X (was ~1.0 m) so the weave WAVELENGTH is ~2x the dog's ~1.0 m follow trail: the
# trailing dog tracks the weave instead of averaging it out and cutting straight through the
# props (the cramped four-bend course put a prop right on the dog's cut line -> wedge -> topple).
# The WAYPOINTS swing +-0.55 m; the follower switches target within the arrival radius (isaac_env:
# 0.18 m on the living-room flat), so the ACTUAL walked path swings ~+-0.50 m with ~50 deg corners
# -- winding, still a safe margin under the ~90 deg hairpins that lose the follower.
# NOTE 2: each in-lane prop is offset ~0.42 m to the side the path curves AWAY from, so its near
# face sits ~0.15 m off the centreline -- the corner-cutting dog passes on the open side.
# validate_layout SIMULATES both the patient walk AND a trailing-dog pursuit and asserts both
# clear every prop; the DOG also has depth avoidance (SIM_AVOID_OBSTACLES=1) as a second margin.
# NOTE 3: starts further back (PATIENT_START_X_M = -5.6) so the three bends have ~2 m of run each
# and the corners stay off the sharp-turn cliff.
# NOTE 4: the last two legs give a straight, CENTRED run-up. Without it the winding route
# delivered the trailing dog to the first riser off-centre (y~0.44, yaw ~-11 deg) and it
# ground against the riser/handrail instead of mounting (run_sim_20260710_015851_696).

# Patient spawn X (= --person-x for the living room, set by run_sim.ps1 -> run_isaac_window.ps1).
# The route's first waypoint is this spawn; starting well back (-5.6) gives the three-bend
# slalom ~2 m per leg so the corners stay gentle and the dog can track the weave.
PATIENT_START_X_M = -5.6
# Waypoint arrival radius the patient follower uses on the living-room FLAT approach (mirrors
# isaac_env.update_person_patrol). Tightened from the shipped 0.35 m so the patient traces the
# angular waypoint path instead of rounding it down to a near-straight sway. Used to SIMULATE
# the actual walked path for furniture-clearance validation.
FLAT_ARRIVE_RADIUS_M = 0.18

# Room extent (for logging / bounds validation only; the physics floor is the
# infinite default ground plane). Widened for the roomier, further-back course.
ROOM_X: Vec2 = (-6.2, 2.0)
ROOM_Y: Vec2 = (-2.9, 2.9)

# Minimum clearance from the PATIENT's WALKED path to any furniture face -- a cosmetic check
# that the kinematic patient does not visually clip a prop (the patient detours wide around the
# in-lane props, so this is generous). The DOG's clearance is the one that actually matters
# (it wedged/toppled when a prop sat on its cut line); it is checked separately, below, against
# a simulated pursuit path with DOG_CLEARANCE_MIN_M, and depth avoidance is a second margin.
CLEARANCE_MIN_M = 0.15
# Minimum clearance from the SIMULATED trailing-dog pursuit path to any IN-LANE obstacle. The
# dog half-width is ~0.15 m; require the pure-pursuit path (no avoidance) to clear each in-lane
# prop by this much so a base run does not wedge even before avoidance helps.
DOG_CLEARANCE_MIN_M = 0.12
# The in-lane obstacle names (checked against the dog path); the wall framing is off the lane.
IN_LANE_PROPS = ("coffee_table", "plant", "end_table")
# Dog follow trail distance (matches --target-distance in run_sim.ps1); sets how much the
# pure-pursuit dog path cuts the weave corners in the validator.
DOG_TRAIL_DISTANCE_M = 1.0
# Waypoints closer than this get skipped by the 0.35 m arrival threshold; keep legs
# comfortably longer so the patient visits every turn.
WAYPOINT_SPACING_MIN_M = 0.70

STAIR_START_X_M = 2.0  # must match StairSpec.start_x_m


def build_living_room_route(start_xy: Vec2, stairs=None) -> List[Vec2]:
    """Full flat waypoint route: [spawn] + weave legs, ending just in front of the
    staircase base. The per-tread + top-landing waypoints are appended by the caller
    (``PatientLocomotionState``) from the active StairSpec, exactly as the default
    straight path does -- so the stair-climb pathing is unchanged.

    ``stairs`` (the active runtime StairSpec, optional) snaps the final approach
    waypoint to ``start_x_m - 0.1`` so it aligns with the spawned treads for any preset.
    """
    start_x = float(stairs.start_x_m) if stairs is not None else STAIR_START_X_M
    legs: List[Vec2] = [(float(x), float(y)) for (x, y) in ROUTE_LEGS]
    if legs:
        legs[-1] = (start_x - 0.1, 0.0)
    route: List[Vec2] = [(float(start_xy[0]), float(start_xy[1]))]
    route.extend(legs)
    return route


# ---------------------------------------------------------------------------
# Pure-python geometry validator (no Isaac). Confirms the route clears every
# furniture box before an expensive sim run.
# ---------------------------------------------------------------------------
def _point_seg_dist(p: Vec2, a: Vec2, b: Vec2) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _orient(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segs_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    d1 = _orient(c, d, a)
    d2 = _orient(c, d, b)
    d3 = _orient(a, b, c)
    d4 = _orient(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def _seg_seg_dist(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> float:
    if _segs_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_seg_dist(a, c, d),
        _point_seg_dist(b, c, d),
        _point_seg_dist(c, a, b),
        _point_seg_dist(d, a, b),
    )


def _point_in_rect(p: Vec2, f: Furniture) -> bool:
    return f.x_min <= p[0] <= f.x_max and f.y_min <= p[1] <= f.y_max


def _seg_furniture_dist(a: Vec2, b: Vec2, f: Furniture) -> float:
    if _point_in_rect(a, f) or _point_in_rect(b, f):
        return 0.0
    corners = [
        (f.x_min, f.y_min), (f.x_max, f.y_min),
        (f.x_max, f.y_max), (f.x_min, f.y_max),
    ]
    best = float("inf")
    for i in range(4):
        e0 = corners[i]
        e1 = corners[(i + 1) % 4]
        best = min(best, _seg_seg_dist(a, b, e0, e1))
    return best


def _simulate_walked_path(waypoints: List[Vec2], arrive: float = FLAT_ARRIVE_RADIUS_M) -> List[Vec2]:
    """Replicate isaac_env.update_person_patrol's flat-approach motion to recover the ACTUAL
    path the patient walks (and the dog follows): step straight toward the current waypoint,
    switch to the next when within ``arrive``. Returns a polyline decimated to ~8 cm. The dog
    follows THIS, not the raw waypoints (which swing wider/sharper because the follower cuts
    each corner by ``arrive``), so furniture clearance is validated against it. speed/dt only
    set sample density; the path shape depends on ``arrive`` and the waypoints."""
    if len(waypoints) < 2:
        return [(float(x), float(y)) for (x, y) in waypoints]
    speed, dt = 0.35, 1.0 / 60.0
    x, y = float(waypoints[0][0]), float(waypoints[0][1])
    idx = 1
    out: List[Vec2] = [(x, y)]
    for _ in range(400000):
        tx, ty = float(waypoints[idx][0]), float(waypoints[idx][1])
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy)
        if d <= arrive:
            if idx >= len(waypoints) - 1:
                break
            idx += 1
            continue
        x += dx / d * speed * dt
        y += dy / d * speed * dt
        if math.hypot(x - out[-1][0], y - out[-1][1]) >= 0.08:
            out.append((x, y))
    last = (float(waypoints[-1][0]), float(waypoints[-1][1]))
    if math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) > 1e-6:
        out.append(last)
    return out


def _simulate_dog_path(patient_path: List[Vec2], trail: float = DOG_TRAIL_DISTANCE_M) -> List[Vec2]:
    """Simulate the trailing follow-dog as pure pursuit of the patient's CURRENT position: each
    step the dog moves toward where the patient is now, holding ~``trail`` m behind. This is the
    corner-cutting behaviour that made the dog drive INTO an in-lane prop on the old cramped
    course (it does NOT copy the weave; it cuts across the pockets). No avoidance is modelled, so
    clearing a prop here is a conservative base-case check -- runtime depth avoidance only helps.

    ``patient_path`` is the decimated walked polyline (from ``_simulate_walked_path``). Returns
    the dog polyline decimated to ~8 cm. The dog starts ``trail`` m behind the patient's first
    point along +X (the spawn offset run_sim.ps1 uses: Go2X = PersonX - 1.0)."""
    if len(patient_path) < 2:
        return list(patient_path)
    # Re-sample the patient path at a fixed time-step so the dog integrates against a moving
    # target (the walked polyline is arc-length decimated, not time-sampled).
    speed, dt = 0.35, 1.0 / 60.0
    step = speed * dt
    dense: List[Vec2] = [patient_path[0]]
    for i in range(1, len(patient_path)):
        ax, ay = dense[-1]
        bx, by = patient_path[i]
        seg = math.hypot(bx - ax, by - ay)
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            t = k / n
            dense.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    dx0, dy0 = 1.0, 0.0
    x, y = dense[0][0] - trail * dx0, dense[0][1] - trail * dy0
    out: List[Vec2] = [(x, y)]
    for px, py in dense:
        vx, vy = px - x, py - y
        d = math.hypot(vx, vy)
        if d > trail:
            x += vx / d * step
            y += vy / d * step
        if math.hypot(x - out[-1][0], y - out[-1][1]) >= 0.08:
            out.append((x, y))
    return out


def _path_max_turn_deg(path: List[Vec2]) -> float:
    """Sharpest heading change along a polyline (degrees). Reported so a layout edit can see
    the ACTUAL corner the follow dog faces (the two runs that climbed had 36 / 51 deg)."""
    worst = 0.0
    for i in range(1, len(path) - 1):
        a0 = math.atan2(path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0])
        a1 = math.atan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
        t = abs(math.degrees(a1 - a0))
        t = min(t, 360.0 - t)
        worst = max(worst, t)
    return worst


def validate_layout(*, verbose: bool = True) -> dict:
    """Assert the WALKED path clears every furniture box and legs are well-spaced.

    Clearance is checked against the simulated actual path (``_simulate_walked_path``), not the
    raw waypoints, because that is where the patient/dog actually go. Returns a summary dict;
    raises AssertionError with a specific message on the first violation.
    """
    route = build_living_room_route((PATIENT_START_X_M, 0.0))
    walked = _simulate_walked_path(route)
    dog = _simulate_dog_path(walked)
    walk_turn = _path_max_turn_deg(walked)
    walk_ymin = min(p[1] for p in walked)
    walk_ymax = max(p[1] for p in walked)
    dog_ymin = min(p[1] for p in dog)
    dog_ymax = max(p[1] for p in dog)
    # 1. Route stays in the room and every turn is in front of the stairs.
    for (x, y) in route[:-1]:
        assert x < STAIR_START_X_M, (
            f"route turn ({x:.2f},{y:.2f}) is inside the stair x-zone (>= {STAIR_START_X_M})"
        )
    for (x, y) in route:
        assert ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1], (
            f"route point ({x:.2f},{y:.2f}) is outside the room {ROOM_X} x {ROOM_Y}"
        )
    assert abs(route[-1][1]) < 1e-6, "final flat waypoint must rejoin the centreline (y=0)"

    # 2. Every furniture piece is inside the room.
    for f in FURNITURE:
        assert ROOM_X[0] <= f.x_min and f.x_max <= ROOM_X[1], f"{f.name} exceeds room X"
        assert ROOM_Y[0] <= f.y_min and f.y_max <= ROOM_Y[1], f"{f.name} exceeds room Y"
        assert f.x_max < STAIR_START_X_M + 1e-9 or f.name in (), None  # props stay off stairs

    # 3. Waypoint spacing (no skips under the 0.35 m arrival threshold).
    worst_gap = float("inf")
    for i in range(len(route) - 1):
        gap = math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        worst_gap = min(worst_gap, gap)
        assert gap >= WAYPOINT_SPACING_MIN_M, (
            f"legs {i}->{i+1} are only {gap:.2f} m apart (< {WAYPOINT_SPACING_MIN_M}); "
            f"the 0.35 m arrival threshold may skip a turn"
        )

    # 4. Clearance: every WALKED-path segment vs every furniture box (the dog follows the
    #    simulated actual path, not the raw waypoints).
    worst = (float("inf"), None, None)
    rows = []
    for i in range(len(walked) - 1):
        a, b = walked[i], walked[i + 1]
        for f in FURNITURE:
            d = _seg_furniture_dist(a, b, f)
            if d < worst[0]:
                worst = (d, i, f.name)
    # tightest clearance per furniture piece (for the printout)
    per_piece = {}
    for f in FURNITURE:
        per_piece[f.name] = min(
            _seg_furniture_dist(walked[i], walked[i + 1], f) for i in range(len(walked) - 1)
        )
    rows = sorted(((d, nm) for nm, d in per_piece.items()), key=lambda r: r[0])

    # 5. DOG clearance: the simulated trailing-pursuit path vs each IN-LANE obstacle. This is
    #    the check that actually protects against the wedge/topple -- the dog cuts the weave
    #    corners and must still clear the props it drives past (avoidance is an extra margin).
    dog_worst = (float("inf"), None)
    dog_per_piece = {}
    for f in FURNITURE:
        if f.name not in IN_LANE_PROPS:
            continue
        d = min(_seg_furniture_dist(dog[i], dog[i + 1], f) for i in range(len(dog) - 1))
        dog_per_piece[f.name] = d
        if d < dog_worst[0]:
            dog_worst = (d, f.name)

    if verbose:
        print("=== living-room layout ===")
        print(f"room: X{ROOM_X}  Y{ROOM_Y}   stairs at x={STAIR_START_X_M}")
        print(f"furniture ({len(FURNITURE)} pieces):")
        for f in FURNITURE:
            print(f"  {f.name:14s} centre=({f.cx:+.2f},{f.cy:+.2f}) "
                  f"size=({f.sx:.2f}x{f.sy:.2f}x{f.sz:.2f}) "
                  f"foot X[{f.x_min:+.2f},{f.x_max:+.2f}] Y[{f.y_min:+.2f},{f.y_max:+.2f}]")
        print(f"route ({len(route)} flat waypoints, start x={PATIENT_START_X_M}):")
        for k, (x, y) in enumerate(route):
            print(f"  wp{k}: ({x:+.2f}, {y:+.2f})")
        print(f"WALKED path (arrive={FLAT_ARRIVE_RADIUS_M} m): actual sway "
              f"[{walk_ymin:+.2f}, {walk_ymax:+.2f}] m   max corner {walk_turn:.0f} deg")
        print(f"DOG pursuit path (trail={DOG_TRAIL_DISTANCE_M} m): sway "
              f"[{dog_ymin:+.2f}, {dog_ymax:+.2f}] m")
        print(f"min waypoint spacing: {worst_gap:.2f} m  (need >= {WAYPOINT_SPACING_MIN_M})")
        print("tightest 5 walked-path/furniture clearances (patient):")
        for d, nm in rows[:5]:
            print(f"  {nm:14s}: {d:.3f} m")
        print(f"MIN PATIENT CLEARANCE: {worst[0]:.3f} m vs {worst[2]}  (need >= {CLEARANCE_MIN_M})")
        print("DOG pursuit clearance vs each in-lane obstacle:")
        for nm in IN_LANE_PROPS:
            print(f"  {nm:14s}: {dog_per_piece.get(nm, float('nan')):.3f} m")
        print(f"MIN DOG CLEARANCE: {dog_worst[0]:.3f} m vs {dog_worst[1]}  (need >= {DOG_CLEARANCE_MIN_M})")

    assert worst[0] >= CLEARANCE_MIN_M, (
        f"walked path clears '{worst[2]}' by only {worst[0]:.3f} m "
        f"(< {CLEARANCE_MIN_M}); move the prop or the waypoint"
    )
    assert dog_worst[0] >= DOG_CLEARANCE_MIN_M, (
        f"simulated follow-dog pursuit path clears in-lane prop '{dog_worst[1]}' by only "
        f"{dog_worst[0]:.3f} m (< {DOG_CLEARANCE_MIN_M}); the dog would wedge -- widen the "
        f"prop offset, the bend spacing, or the detour amplitude"
    )
    return {
        "min_clearance_m": round(worst[0], 3),
        "min_dog_clearance_m": round(dog_worst[0], 3),
        "min_waypoint_spacing_m": round(worst_gap, 3),
        "walked_sway_m": [round(walk_ymin, 2), round(walk_ymax, 2)],
        "dog_sway_m": [round(dog_ymin, 2), round(dog_ymax, 2)],
        "walked_max_corner_deg": round(walk_turn, 1),
        "furniture_count": len(FURNITURE),
        "route_waypoints": len(route),
    }


# ---------------------------------------------------------------------------
# Isaac spawn (imports Isaac lazily so the module stays system-python importable).
# ---------------------------------------------------------------------------
def _import_isaac_refs():
    """Resolve the nucleus + add_reference helpers across Isaac SDK versions
    (same branch isaac_env / final_scene use)."""
    try:
        import omni.isaac.core.utils.nucleus as nucleus_utils
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ModuleNotFoundError:
        import isaacsim.storage.native as nucleus_utils
        from isaacsim.core.utils.stage import add_reference_to_stage
    return nucleus_utils, add_reference_to_stage


def _archvis_uri_candidates(relpath, assets_root):
    """Full URIs to try for an ArchVis-relative path: the local assets root (the
    ArchVis collection is the sibling of ``.../Assets/Isaac/<ver>``) first, then the
    S3 CDN. Spaces are %20-encoded so https/omniverse URIs resolve."""
    enc = relpath.replace(" ", "%20")
    cands = []
    if assets_root:
        base = assets_root.rstrip("/")
        idx = base.rfind("/Isaac/")
        base = base[:idx] if idx != -1 else base  # ".../Assets"
        cands.append(base.rstrip("/") + "/" + enc)
    cands.append(_ARCHVIS_CDN_BASE + enc)
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_archvis_uri(relpath, assets_root, nucleus_utils):
    """First candidate confirmed by ``is_file``; else the CDN candidate to ATTEMPT
    (composition is validated by the caller, which falls back to a box on failure)."""
    cands = _archvis_uri_candidates(relpath, assets_root)
    for c in cands:
        try:
            if nucleus_utils.is_file(c):
                return c
        except Exception:
            pass
    return cands[-1] if cands else None


def _read_up_axis(uri):
    """Up-axis ('Y'/'Z') of the referenced layer. ArchVis assets are typically
    Y-up; Isaac's stage is Z-up and references are NOT auto-rotated, so we read the
    source metadata to know whether to stand the mesh up. Populates the Sdf layer
    cache, so the subsequent reference reuses this open (no double download)."""
    try:
        from pxr import Usd, UsdGeom
        s = Usd.Stage.Open(uri)
        if s is not None:
            return UsdGeom.GetStageUpAxis(s)
    except Exception:
        pass
    return "Z"


def _spawn_real_furniture(stage, f, uri, up_axis, root, add_reference_to_stage):
    """Reference the real mesh under a wrapper Xform, correct its up-axis, uniformly
    scale it to FIT INSIDE the ``f`` footprint, and seat it on the floor at the box
    centre. Returns the PLACED world-space AABB ``(min_xyz, max_xyz)`` on success (so
    the collider can be sized to exactly what the depth camera sees), or ``None`` on
    any degenerate bound so the caller falls back to a box."""
    from pxr import Usd, UsdGeom, Gf

    wrapper = f"{root}/{f.name}"
    model = f"{wrapper}/Model"
    if stage.GetPrimAtPath(wrapper).IsValid():
        stage.RemovePrim(wrapper)
    UsdGeom.Xform.Define(stage, wrapper)
    add_reference_to_stage(usd_path=uri, prim_path=model)

    model_prim = stage.GetPrimAtPath(model)
    if model_prim and model_prim.IsValid():
        try:
            model_prim.Load()  # pull payloads so the bound is computable
        except Exception:
            pass
    if not model_prim or not model_prim.IsValid() or not model_prim.GetChildren():
        return None

    wrapper_prim = stage.GetPrimAtPath(wrapper)
    xf = UsdGeom.Xformable(wrapper_prim)
    xf.ClearXformOpOrder()
    # op order [translate, rotate, scale] => point transformed by scale first, then
    # the up-axis rotation, then placement -- so scale is applied in asset-local space.
    t_op = xf.AddTranslateOp()
    r_op = xf.AddRotateXYZOp()
    s_op = xf.AddScaleOp()
    rot = Gf.Vec3f(90.0, 0.0, 0.0) if str(up_axis).upper().startswith("Y") else Gf.Vec3f(0.0, 0.0, 0.0)
    r_op.Set(rot)
    s_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    rng = bbox.ComputeWorldBound(wrapper_prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    dx = max(mx[0] - mn[0], 1e-6)
    dy = max(mx[1] - mn[1], 1e-6)
    dz = max(mx[2] - mn[2], 1e-6)
    s = min(f.sx / dx, f.sy / dy, f.sz / dz)  # uniform fit-inside (proportions kept)
    if not (0.0 < s < float("inf")):
        return None
    s_op.Set(Gf.Vec3f(s, s, s))

    bbox.Clear()
    rng2 = bbox.ComputeWorldBound(wrapper_prim).ComputeAlignedRange()
    if rng2.IsEmpty():
        return None
    mn2, mx2 = rng2.GetMin(), rng2.GetMax()
    cx_now = 0.5 * (mn2[0] + mx2[0])
    cy_now = 0.5 * (mn2[1] + mx2[1])
    dz_placed = float(mx2[2] - mn2[2])
    t_op.Set(Gf.Vec3d(float(f.cx - cx_now), float(f.cy - cy_now), float(-mn2[2])))
    # Placed AABB, snapped to the floor (base z = 0) and centred at the box (cx, cy).
    placed_min = (f.cx - 0.5 * float(mx2[0] - mn2[0]), f.cy - 0.5 * float(mx2[1] - mn2[1]), 0.0)
    placed_max = (f.cx + 0.5 * float(mx2[0] - mn2[0]), f.cy + 0.5 * float(mx2[1] - mn2[1]), dz_placed)
    return placed_min, placed_max


def _add_footprint_collider(world, stage, f, root, FixedCuboid, np, *, visible, aabb=None):
    """The collidable footprint box. When a real mesh is drawn on top we keep this
    box for physics but hide its visual (``visible=False``) and size it to the mesh's
    placed AABB (so collision matches exactly what the depth camera sees); when the
    mesh is absent it IS the prop (``visible=True``), the plain-box variant sized to
    the nominal footprint."""
    from pxr import UsdGeom

    if aabb is not None:
        (x0, y0, z0), (x1, y1, z1) = aabb
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        sx, sy, sz = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3), max(z1 - z0, 1e-3)
    else:
        cx, cy, sx, sy, sz = f.cx, f.cy, f.sx, f.sy, f.sz

    prim_path = f"{root}/{f.name}_collider" if not visible else f"{root}/{f.name}"
    name = f"living_room_{f.name}_col" if not visible else f"living_room_{f.name}"
    world.scene.add(
        FixedCuboid(
            prim_path=prim_path,
            name=name,
            position=np.array([cx, cy, sz / 2.0], dtype=float),
            scale=np.array([sx, sy, sz], dtype=float),
            color=np.array(f.color, dtype=float),
        )
    )
    if not visible:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def spawn_living_room(world) -> int:
    """Spawn the household furniture into the live Isaac scene.

    Each prop is a realistic Omniverse ArchVis mesh (``FURNITURE_USD``) skinned over
    an invisible collidable footprint box; any mesh that fails to resolve falls back
    to a visible coloured box. Set ``SIM_LIVING_ROOM_BOXES=1`` to force plain boxes.

    Called from ``isaac_env.main`` right after ``spawn_obstacles`` (before
    ``world.reset()``) when ``args.living_room`` is set. Returns the prop count.
    """
    import logging
    import os
    import numpy as np
    from sim_logging_utils import log_event
    from env import env_state

    try:
        from omni.isaac.core.objects import FixedCuboid
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedCuboid

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    force_boxes = os.environ.get("SIM_LIVING_ROOM_BOXES") == "1"
    nucleus_utils = add_reference_to_stage = None
    assets_root = None
    if not force_boxes:
        try:
            nucleus_utils, add_reference_to_stage = _import_isaac_refs()
            try:
                assets_root = nucleus_utils.get_assets_root_path()
            except Exception:
                assets_root = None
        except Exception as exc:
            log_event(
                env_state.LOGGER, logging.WARNING, "living_room_refs_unavailable",
                "Could not import Isaac reference helpers; using plain boxes",
                error=str(exc),
            )
            add_reference_to_stage = None

    root = "/World/LivingRoom"
    spawned = 0
    real = 0
    real_props = []
    for f in FURNITURE:
        used_real = False
        relpath = FURNITURE_USD.get(f.name)
        if add_reference_to_stage is not None and relpath:
            try:
                uri = _resolve_archvis_uri(relpath, assets_root, nucleus_utils)
                up_axis = _read_up_axis(uri) if uri else "Z"
                placed = _spawn_real_furniture(stage, f, uri, up_axis, root, add_reference_to_stage) if uri else None
                if placed is not None:
                    _add_footprint_collider(world, stage, f, root, FixedCuboid, np, visible=False, aabb=placed)
                    used_real = True
                    real += 1
                    real_props.append(f.name)
            except Exception as exc:
                log_event(
                    env_state.LOGGER, logging.WARNING, "living_room_real_asset_failed",
                    f"Real mesh for {f.name} failed; falling back to a box",
                    prop=f.name, uri=relpath, error=str(exc),
                )
                try:
                    if stage.GetPrimAtPath(f"{root}/{f.name}").IsValid():
                        stage.RemovePrim(f"{root}/{f.name}")
                except Exception:
                    pass
                used_real = False

        if not used_real:
            try:
                _add_footprint_collider(world, stage, f, root, FixedCuboid, np, visible=True)
            except Exception as exc:
                log_event(
                    env_state.LOGGER, logging.WARNING, "living_room_prop_failed",
                    f"Failed to spawn living-room prop {f.name}",
                    prim_path=f"{root}/{f.name}", error=str(exc),
                )
                continue
        spawned += 1

    log_event(
        env_state.LOGGER, logging.INFO, "living_room_spawned",
        f"Living-room furniture spawned ({spawned}/{len(FURNITURE)} props, "
        f"{real} realistic meshes); patient weaves a winding route around them "
        f"before the stairs",
        prop_count=spawned,
        realistic_mesh_count=real,
        realistic_props=real_props,
        boxed_props=[f.name for f in FURNITURE if f.name not in real_props],
        props=[f.name for f in FURNITURE],
        room_x=list(ROOM_X),
        room_y=list(ROOM_Y),
    )
    return spawned


if __name__ == "__main__":
    summary = validate_layout(verbose=True)
    print(f"\nOK: {summary}")
