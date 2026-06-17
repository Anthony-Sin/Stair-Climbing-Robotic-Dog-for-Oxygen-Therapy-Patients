"""Tiny dependency-free polygon-mesh toolkit.

Pure Python (no numpy / pxr) so the oxygen-tank + rail visuals can be generated
into ``.usda`` without Isaac Sim. A :class:`Mesh` is just a list of points and a
list of faces (each face an index list, any number of sides -- USD supports
n-gons). Helpers build the few primitives we need:

  * :func:`box`               - axis-aligned box
  * :func:`rounded_rect_prism`- box with rounded vertical edges (the tank shell)
  * :func:`cylinder`          - capped cylinder (intake cap, buttons, rails ends)

All coordinates are in metres, matching the stage (``metersPerUnit = 1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

Pt = Tuple[float, float, float]


@dataclass
class Mesh:
    points: List[Pt] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)

    # -- construction helpers --------------------------------------------
    def add_point(self, p: Pt) -> int:
        self.points.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.points) - 1

    def add_face(self, idxs: Sequence[int]) -> None:
        self.faces.append([int(i) for i in idxs])

    def merge(self, other: "Mesh") -> "Mesh":
        """Append ``other`` into this mesh (indices are rebased)."""
        offset = len(self.points)
        self.points.extend(other.points)
        for f in other.faces:
            self.faces.append([i + offset for i in f])
        return self

    def translated(self, dx: float, dy: float, dz: float) -> "Mesh":
        m = Mesh()
        m.points = [(x + dx, y + dy, z + dz) for (x, y, z) in self.points]
        m.faces = [list(f) for f in self.faces]
        return m

    # -- queries ---------------------------------------------------------
    def extent(self) -> Tuple[Pt, Pt]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        zs = [p[2] for p in self.points]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def box(center: Pt, size: Pt) -> Mesh:
    """Axis-aligned box centred at ``center`` with full-extent ``size``."""
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    m = Mesh()
    # 8 corners
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                m.add_point((cx + sx * hx, cy + sy * hy, cz + sz * hz))
    # index helper: bit2=x, bit1=y, bit0=z over the loop order above
    def idx(sx: int, sy: int, sz: int) -> int:
        return ((0 if sx < 0 else 1) << 2) | ((0 if sy < 0 else 1) << 1) | (0 if sz < 0 else 1)

    # 6 faces, wound CCW when viewed from outside (right-hand rule -> outward)
    m.add_face([idx(1, -1, -1), idx(1, 1, -1), idx(1, 1, 1), idx(1, -1, 1)])   # +X
    m.add_face([idx(-1, -1, -1), idx(-1, -1, 1), idx(-1, 1, 1), idx(-1, 1, -1)])  # -X
    m.add_face([idx(-1, 1, -1), idx(-1, 1, 1), idx(1, 1, 1), idx(1, 1, -1)])   # +Y
    m.add_face([idx(-1, -1, -1), idx(1, -1, -1), idx(1, -1, 1), idx(-1, -1, 1)])  # -Y
    m.add_face([idx(-1, -1, 1), idx(1, -1, 1), idx(1, 1, 1), idx(-1, 1, 1)])   # +Z (top)
    m.add_face([idx(-1, -1, -1), idx(-1, 1, -1), idx(1, 1, -1), idx(1, -1, -1)])  # -Z (bottom)
    return m


def _rounded_rect_ring(a: float, b: float, r: float, seg: int, z: float) -> List[Pt]:
    """One horizontal ring of a rounded rectangle outline (CCW), at height ``z``.

    ``a``/``b`` are the half-length (X) / half-width (Y); ``r`` the corner radius;
    ``seg`` the number of segments per 90-degree corner.
    """
    r = max(0.0, min(r, min(a, b) - 1e-4))
    cx, cy = a - r, b - r  # centres of the four corner arcs
    ring: List[Pt] = []
    corners = [
        (cx, cy, 0.0),                 # top-right  : 0   -> 90
        (-cx, cy, math.pi / 2.0),      # top-left   : 90  -> 180
        (-cx, -cy, math.pi),           # bot-left   : 180 -> 270
        (cx, -cy, 3.0 * math.pi / 2.0),  # bot-right : 270 -> 360
    ]
    for ox, oy, a0 in corners:
        for k in range(seg + 1):
            ang = a0 + (math.pi / 2.0) * (k / seg)
            ring.append((ox + r * math.cos(ang), oy + r * math.sin(ang), z))
    return ring


def rounded_rect_prism(
    center: Pt,
    length_x: float,
    width_y: float,
    height_z: float,
    radius: float,
    seg: int = 6,
) -> Mesh:
    """Box with rounded vertical edges -- the concentrator shell silhouette."""
    cx, cy, cz = center
    a, b = length_x / 2.0, width_y / 2.0
    z0, z1 = cz - height_z / 2.0, cz + height_z / 2.0
    bottom = _rounded_rect_ring(a, b, radius, seg, z0)
    top = _rounded_rect_ring(a, b, radius, seg, z1)
    n = len(bottom)

    m = Mesh()
    base_bottom = 0
    for p in bottom:
        m.add_point((cx + p[0], cy + p[1], p[2]))
    base_top = len(m.points)
    for p in top:
        m.add_point((cx + p[0], cy + p[1], p[2]))

    # Side walls (CCW from outside).
    for i in range(n):
        j = (i + 1) % n
        m.add_face([base_bottom + i, base_bottom + j, base_top + j, base_top + i])

    # Caps as single n-gons (top CCW up, bottom CCW down).
    m.add_face([base_top + i for i in range(n)])
    m.add_face([base_bottom + i for i in range(n - 1, -1, -1)])
    return m


def cylinder(
    center: Pt,
    radius: float,
    height: float,
    axis: str = "z",
    seg: int = 24,
) -> Mesh:
    """Capped cylinder of ``height`` along ``axis`` ('x'|'y'|'z')."""
    cx, cy, cz = center
    h = height / 2.0
    m = Mesh()

    def place(u: float, v: float, w: float) -> Pt:
        # (u,v) span the circular cross-section; w is along the chosen axis.
        if axis == "z":
            return (cx + u, cy + v, cz + w)
        if axis == "x":
            return (cx + w, cy + u, cz + v)
        if axis == "y":
            return (cx + u, cy + w, cz + v)
        raise ValueError(f"bad axis {axis!r}")

    bottom_ring, top_ring = [], []
    for k in range(seg):
        ang = 2.0 * math.pi * (k / seg)
        u, v = radius * math.cos(ang), radius * math.sin(ang)
        bottom_ring.append(m.add_point(place(u, v, -h)))
        top_ring.append(m.add_point(place(u, v, h)))

    for k in range(seg):
        j = (k + 1) % seg
        m.add_face([bottom_ring[k], bottom_ring[j], top_ring[j], top_ring[k]])
    m.add_face([top_ring[k] for k in range(seg)])
    m.add_face([bottom_ring[k] for k in range(seg - 1, -1, -1)])
    return m
