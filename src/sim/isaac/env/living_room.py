"""Living-room scene variant: household furniture + a winding patient route.

Opt-in via ``--living-room`` (isaac_args) / ``SIM_LIVING_ROOM=1`` (run_living_room.bat).
It restages the SAME proven robot / patient / stair-climb stack from the default sim
inside a furnished living room: a handful of collidable household props (sofa, coffee
table, bookshelf, armchair, TV console, ...) sit on the flat approach floor, and the
patient walks a realistic serpentine route that weaves AROUND them before rejoining the
stair centreline and climbing -- instead of the old straight line / simple left-right
zigzag (``--person-approach-turns``).

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


# Household furniture. A few "gate" pieces sit near the centreline to force the
# serpentine weave; the rest frame the room against the side walls. Heights and
# footprints are roughly life-sized. All are collidable (FixedCuboid) so the run is
# a genuine obstacle course.
FURNITURE: Tuple[Furniture, ...] = (
    # --- gate pieces: the patient visibly curves AROUND each of these two ---
    Furniture("coffee_table", -2.30, 0.40, 0.90, 0.70, 0.42, (0.42, 0.28, 0.15)),  # wood; patient passes below it
    Furniture("sofa_center", -0.20, -0.55, 1.40, 0.95, 0.75, (0.60, 0.52, 0.40)),  # beige couch; patient passes above it
    # --- side / wall framing pieces (the route clears these comfortably) ---
    Furniture("sofa_wall", -2.55, 1.90, 2.30, 0.80, 0.75, (0.30, 0.36, 0.46)),     # blue-grey sofa on the +Y wall
    Furniture("bookshelf", -3.60, -1.95, 0.90, 0.45, 1.80, (0.28, 0.18, 0.10)),    # dark walnut on the -Y wall
    Furniture("armchair", -1.05, -2.05, 0.90, 0.90, 0.80, (0.34, 0.42, 0.32)),     # green chair, -Y side
    Furniture("tv_console", 0.60, 2.15, 2.00, 0.40, 0.55, (0.14, 0.14, 0.16)),     # charcoal TV unit, +Y wall
    Furniture("end_table", 1.10, -1.75, 0.55, 0.55, 0.50, (0.40, 0.26, 0.14)),     # wood side table, -Y side
    Furniture("plant", 1.70, 1.55, 0.42, 0.42, 1.25, (0.18, 0.42, 0.20)),          # potted plant, +Y decor
    Furniture("floor_lamp", -3.45, 1.10, 0.28, 0.28, 1.55, (0.78, 0.74, 0.62)),    # brass floor lamp, +Y decor
)


# ---------------------------------------------------------------------------
# Winding patient route: flat legs BEFORE the stairs that weave around the gate
# furniture, then rejoin the stair centreline. The spawn point is prepended and the
# final leg is snapped to just in front of the staircase base at runtime.
# ---------------------------------------------------------------------------
# Interior legs only (spawn prepended, stair approach snapped in build_living_room_route).
ROUTE_LEGS: Tuple[Vec2, ...] = (
    (-2.30, -0.95),  # veer into the lower lane, curving below the coffee table
    (-1.38, -0.95),  # hold the lower lane out to the clear gap between the table and the sofa
    (-1.38, 0.90),   # turn and cross up through the furniture-free gap into the upper lane
    (0.85, 0.90),    # travel the upper lane above the centre sofa, out past its far end
    (1.90, 0.00),    # curve back down and rejoin the stair centreline (snapped at runtime)
)

# Room extent (for logging / bounds validation only; the physics floor is the
# infinite default ground plane).
ROOM_X: Vec2 = (-4.2, 2.0)
ROOM_Y: Vec2 = (-2.6, 2.6)

# Minimum clearance required from the route CENTRELINE to any furniture face. The
# kinematic patient is driven exactly along this path, so 0.45 m still leaves the
# ~0.5 m-wide mannequin a visible gap (~0.20 m) and the ~0.31 m-wide Go2 body that
# follows it ~0.29 m -- enough to weave the gaps on a first-look run. Widen this (and
# re-space the furniture) if the follow controller is seen clipping a prop.
CLEARANCE_MIN_M = 0.45
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


def validate_layout(*, verbose: bool = True) -> dict:
    """Assert the route clears every furniture box and legs are well-spaced.

    Returns a summary dict; raises AssertionError with a specific message on the
    first violation so a layout edit gets an actionable failure.
    """
    route = build_living_room_route((-3.5, 0.0))
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

    # 4. Clearance: every route segment vs every furniture box.
    worst = (float("inf"), None, None)
    rows = []
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        for f in FURNITURE:
            d = _seg_furniture_dist(a, b, f)
            rows.append((d, i, f.name))
            if d < worst[0]:
                worst = (d, i, f.name)
    rows.sort(key=lambda r: r[0])

    if verbose:
        print("=== living-room layout ===")
        print(f"room: X{ROOM_X}  Y{ROOM_Y}   stairs at x={STAIR_START_X_M}")
        print(f"furniture ({len(FURNITURE)} pieces):")
        for f in FURNITURE:
            print(f"  {f.name:14s} centre=({f.cx:+.2f},{f.cy:+.2f}) "
                  f"size=({f.sx:.2f}x{f.sy:.2f}x{f.sz:.2f}) "
                  f"foot X[{f.x_min:+.2f},{f.x_max:+.2f}] Y[{f.y_min:+.2f},{f.y_max:+.2f}]")
        print(f"route ({len(route)} flat waypoints):")
        for k, (x, y) in enumerate(route):
            print(f"  wp{k}: ({x:+.2f}, {y:+.2f})")
        print(f"min waypoint spacing: {worst_gap:.2f} m  (need >= {WAYPOINT_SPACING_MIN_M})")
        print("tightest 5 segment/furniture clearances:")
        for d, i, nm in rows[:5]:
            print(f"  seg {i}->{i+1} vs {nm:14s}: {d:.3f} m")
        print(f"MIN CLEARANCE: {worst[0]:.3f} m at seg {worst[1]}->{worst[1]+1} "
              f"vs {worst[2]}  (need >= {CLEARANCE_MIN_M})")

    assert worst[0] >= CLEARANCE_MIN_M, (
        f"route seg {worst[1]}->{worst[1]+1} clears '{worst[2]}' by only "
        f"{worst[0]:.3f} m (< {CLEARANCE_MIN_M}); widen the gap or move the waypoint"
    )
    return {
        "min_clearance_m": round(worst[0], 3),
        "min_waypoint_spacing_m": round(worst_gap, 3),
        "furniture_count": len(FURNITURE),
        "route_waypoints": len(route),
    }


# ---------------------------------------------------------------------------
# Isaac spawn (imports Isaac lazily so the module stays system-python importable).
# ---------------------------------------------------------------------------
def spawn_living_room(world) -> int:
    """Spawn the collidable household furniture into the live Isaac scene.

    Called from ``isaac_env.main`` right after ``spawn_obstacles`` when
    ``args.living_room`` is set. Returns the number of props spawned.
    """
    import logging
    import numpy as np
    from sim_logging_utils import log_event
    from env import env_state

    try:
        from omni.isaac.core.objects import FixedCuboid
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedCuboid

    root = "/World/LivingRoom"
    spawned = 0
    for f in FURNITURE:
        try:
            world.scene.add(
                FixedCuboid(
                    prim_path=f"{root}/{f.name}",
                    name=f"living_room_{f.name}",
                    position=np.array([f.cx, f.cy, f.sz / 2.0], dtype=float),
                    scale=np.array([f.sx, f.sy, f.sz], dtype=float),
                    color=np.array(f.color, dtype=float),
                )
            )
            spawned += 1
        except Exception as exc:
            log_event(
                env_state.LOGGER, logging.WARNING, "living_room_prop_failed",
                f"Failed to spawn living-room prop {f.name}",
                prim_path=f"{root}/{f.name}", error=str(exc),
            )

    log_event(
        env_state.LOGGER, logging.INFO, "living_room_spawned",
        f"Living-room furniture spawned ({spawned}/{len(FURNITURE)} props); "
        f"patient weaves a winding route around them before the stairs",
        prop_count=spawned,
        props=[f.name for f in FURNITURE],
        room_x=list(ROOM_X),
        room_y=list(ROOM_Y),
    )
    return spawned


if __name__ == "__main__":
    summary = validate_layout(verbose=True)
    print(f"\nOK: {summary}")
