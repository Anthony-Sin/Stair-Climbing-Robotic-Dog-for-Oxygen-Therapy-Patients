"""Clean low-poly parametric primitive mesh builders for the line-art blueprint viewer.

Every builder returns a ``Mesh`` (flat lists of positions / normals / triangle indices,
local to the primitive's own origin, ready to be transformed into a link's local frame
and embedded as a glTF primitive). Geometry only -- no UVs, no materials, no color:
the contract strips all texturing and lets the viewer apply line-art shading itself.

Primitive catalogue used by the URDF-collision-derived Go2 tier-(c) fallback:
  * ``box(sx, sy, sz)``              -- axis-aligned box, full extents, centered at origin
  * ``rounded_box(...)``             -- box with chamfered vertical edges (O2 tank silhouette)
  * ``capsule(radius, length, axis)``-- cylinder + two hemispherical caps, centered, given axis
  * ``sphere(radius)``               -- UV sphere
  * ``cylinder(radius, length, axis)``-- flat-capped cylinder (handrail posts, calflower hints)
  * ``cylinder_between(p0, p1, r)``  -- flat-capped cylinder whose end-cap CENTERS land
        exactly on p0/p1 (handrail segments -- endpoint-driven, no hand-written rotation
        matrices; added after the 2026-07-07 handrail incident where a wrong-signed
        R_y matrix inverted the sloped rail's direction)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class Mesh:
    positions: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)  # flat triangle list, 3 per tri

    def extend(self, other: "Mesh", offset: Vec3 = (0.0, 0.0, 0.0)) -> None:
        """Append another mesh's geometry in place, translated by ``offset``."""
        base = len(self.positions)
        for p in other.positions:
            self.positions.append((p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]))
        self.normals.extend(other.normals)
        self.indices.extend(i + base for i in other.indices)

    def transform(self, matrix3: List[List[float]], translation: Vec3) -> "Mesh":
        """Return a new Mesh with positions/normals transformed by a 3x3 rotation
        matrix + translation (rotation applied first, matching URDF origin semantics)."""
        out = Mesh(indices=list(self.indices))
        for p in self.positions:
            x, y, z = p
            rx = matrix3[0][0] * x + matrix3[0][1] * y + matrix3[0][2] * z
            ry = matrix3[1][0] * x + matrix3[1][1] * y + matrix3[1][2] * z
            rz = matrix3[2][0] * x + matrix3[2][1] * y + matrix3[2][2] * z
            out.positions.append((rx + translation[0], ry + translation[1], rz + translation[2]))
        for n in self.normals:
            x, y, z = n
            rx = matrix3[0][0] * x + matrix3[0][1] * y + matrix3[0][2] * z
            ry = matrix3[1][0] * x + matrix3[1][1] * y + matrix3[1][2] * z
            rz = matrix3[2][0] * x + matrix3[2][1] * y + matrix3[2][2] * z
            out.normals.append((rx, ry, rz))
        return out

    def triangle_count(self) -> int:
        return len(self.indices) // 3


def _quad(mesh: Mesh, a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3) -> None:
    """Add a quad a-b-c-d (CCW winding, viewed from the normal side) as two triangles."""
    base = len(mesh.positions)
    for p in (a, b, c, d):
        mesh.positions.append(p)
        mesh.normals.append(normal)
    mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])


def box(sx: float, sy: float, sz: float, center: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    """Axis-aligned box, full extents (sx, sy, sz), 24 verts (hard-edged normals), 12 tris."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    cx, cy, cz = center
    m = Mesh()
    # +X face
    _quad(m, (cx+hx, cy-hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx+hx, cy+hy, cz+hz), (cx+hx, cy-hy, cz+hz), (1, 0, 0))
    # -X face
    _quad(m, (cx-hx, cy+hy, cz-hz), (cx-hx, cy-hy, cz-hz), (cx-hx, cy-hy, cz+hz), (cx-hx, cy+hy, cz+hz), (-1, 0, 0))
    # +Y face
    _quad(m, (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz+hz), (cx+hx, cy+hy, cz+hz), (0, 1, 0))
    # -Y face
    _quad(m, (cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz+hz), (cx-hx, cy-hy, cz+hz), (0, -1, 0))
    # +Z face (top)
    _quad(m, (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz), (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz), (0, 0, 1))
    # -Z face (bottom)
    _quad(m, (cx-hx, cy+hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx-hx, cy-hy, cz-hz), (0, 0, -1))
    return m


def _axis_frame(axis: str) -> Tuple[int, int, int]:
    """Return (long_axis_index, u_index, v_index) for 'x'|'y'|'z'."""
    axis = axis.lower()
    if axis == "x":
        return 0, 1, 2
    if axis == "y":
        return 1, 2, 0
    if axis == "z":
        return 2, 0, 1
    raise ValueError(f"axis must be x/y/z, got {axis!r}")


def cylinder(radius: float, length: float, axis: str = "z", segments: int = 12,
             center: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    """Flat-capped cylinder of given ``length`` centered at ``center``, long axis = axis."""
    ax, u, v = _axis_frame(axis)
    half = length / 2.0
    m = Mesh()
    ring_lo: List[Vec3] = []
    ring_hi: List[Vec3] = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cu, cv = radius * math.cos(theta), radius * math.sin(theta)
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        lo[ax], hi[ax] = -half, half
        lo[u] = hi[u] = cu
        lo[v] = hi[v] = cv
        ring_lo.append(tuple(c + o for c, o in zip(lo, center)))
        ring_hi.append(tuple(c + o for c, o in zip(hi, center)))
    # Side wall
    for i in range(segments):
        j = (i + 1) % segments
        theta_i = 2.0 * math.pi * i / segments
        theta_j = 2.0 * math.pi * j / segments
        n_i = [0.0, 0.0, 0.0]
        n_j = [0.0, 0.0, 0.0]
        n_i[u], n_i[v] = math.cos(theta_i), math.sin(theta_i)
        n_j[u], n_j[v] = math.cos(theta_j), math.sin(theta_j)
        base = len(m.positions)
        m.positions.extend([ring_lo[i], ring_lo[j], ring_hi[j], ring_hi[i]])
        m.normals.extend([tuple(n_i), tuple(n_j), tuple(n_j), tuple(n_i)])
        m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    # Caps
    cap_hi_normal = [0.0, 0.0, 0.0]
    cap_hi_normal[ax] = 1.0
    cap_lo_normal = [0.0, 0.0, 0.0]
    cap_lo_normal[ax] = -1.0
    hi_center = [0.0, 0.0, 0.0]
    hi_center[ax] = half
    hi_center = tuple(c + o for c, o in zip(hi_center, center))
    lo_center = [0.0, 0.0, 0.0]
    lo_center[ax] = -half
    lo_center = tuple(c + o for c, o in zip(lo_center, center))
    base = len(m.positions)
    m.positions.append(hi_center)
    m.normals.append(tuple(cap_hi_normal))
    for i in range(segments):
        m.positions.append(ring_hi[i])
        m.normals.append(tuple(cap_hi_normal))
    for i in range(segments):
        j = (i + 1) % segments
        m.indices.extend([base, base + 1 + i, base + 1 + j])
    base = len(m.positions)
    m.positions.append(lo_center)
    m.normals.append(tuple(cap_lo_normal))
    for i in range(segments):
        m.positions.append(ring_lo[i])
        m.normals.append(tuple(cap_lo_normal))
    for i in range(segments):
        j = (i + 1) % segments
        m.indices.extend([base, base + 1 + j, base + 1 + i])
    return m


def sphere(radius: float, lat_segments: int = 8, lon_segments: int = 12,
           center: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    """UV sphere, smooth normals, ``lat_segments`` rings x ``lon_segments`` columns."""
    m = Mesh()
    verts: List[List[int]] = []
    for i in range(lat_segments + 1):
        phi = math.pi * i / lat_segments  # 0 (north pole, +Z) .. pi (south pole, -Z)
        row: List[int] = []
        for j in range(lon_segments):
            theta = 2.0 * math.pi * j / lon_segments
            nx = math.sin(phi) * math.cos(theta)
            ny = math.sin(phi) * math.sin(theta)
            nz = math.cos(phi)
            row.append(len(m.positions))
            m.positions.append((center[0] + radius * nx, center[1] + radius * ny, center[2] + radius * nz))
            m.normals.append((nx, ny, nz))
        verts.append(row)
    for i in range(lat_segments):
        for j in range(lon_segments):
            j2 = (j + 1) % lon_segments
            a, b = verts[i][j], verts[i][j2]
            c, d = verts[i + 1][j2], verts[i + 1][j]
            if i != 0:
                m.indices.extend([a, c, b])
            if i != lat_segments - 1:
                m.indices.extend([a, d, c])
    return m


def capsule(radius: float, length: float, axis: str = "z", segments: int = 12,
            lat_segments: int = 4, center: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    """Cylinder of ``length`` (the straight part, NOT including the hemispherical caps)
    plus a hemisphere cap of ``radius`` on each end, long axis = axis. This matches the
    URDF cylinder-collision convention where ``length`` is the cylindrical section only.
    """
    ax, u, v = _axis_frame(axis)
    half = length / 2.0
    m = Mesh()

    def hemisphere(sign: float) -> Mesh:
        """Hemisphere cap bulging toward +axis (sign=+1) or -axis (sign=-1), flat side
        at local-axis=0, apex at local-axis=sign*radius. Built directly in the requested
        axis frame (not built in Z-up and rotated) to avoid seam/orientation bugs."""
        hm = Mesh()
        rows: List[List[int]] = []
        for i in range(lat_segments + 1):
            # phi=0 at the flat equator (axis=0), phi=pi/2 at the apex (axis=sign*radius)
            phi = (math.pi / 2.0) * i / lat_segments
            row: List[int] = []
            ring_r = radius * math.cos(phi)
            ring_h = sign * radius * math.sin(phi)
            for j in range(segments):
                theta = 2.0 * math.pi * j / segments
                cu, cv = ring_r * math.cos(theta), ring_r * math.sin(theta)
                pos = [0.0, 0.0, 0.0]
                pos[ax] = ring_h
                pos[u] = cu
                pos[v] = cv
                nrm = [0.0, 0.0, 0.0]
                nlen = math.sqrt(ring_h * ring_h + cu * cu + cv * cv)
                if nlen > 1e-9:
                    nrm[ax], nrm[u], nrm[v] = ring_h / nlen, cu / nlen, cv / nlen
                row.append(len(hm.positions))
                hm.positions.append(tuple(pos))
                hm.normals.append(tuple(nrm))
            rows.append(row)
        for i in range(lat_segments):
            is_apex_band = (i == lat_segments - 1)  # row i+1 is the degenerate pole ring
            for j in range(segments):
                j2 = (j + 1) % segments
                a, b = rows[i][j], rows[i][j2]
                c, d = rows[i + 1][j2], rows[i + 1][j]
                # Winding chosen so the outward normal matches for sign=+1; sign=-1
                # naturally gets consistent (already outward) winding via the geometry.
                if is_apex_band:
                    # Row i+1 is the pole: c == d (both the same collapsed vertex), so
                    # emit ONE triangle per segment (a fan), not a quad-as-two-triangles
                    # (which would degenerate into a legit tri + a zero-area tri).
                    if sign > 0:
                        hm.indices.extend([a, b, c])
                    else:
                        hm.indices.extend([a, c, b])
                else:
                    if sign > 0:
                        hm.indices.extend([a, b, c])
                        hm.indices.extend([a, c, d])
                    else:
                        hm.indices.extend([a, c, b])
                        hm.indices.extend([a, d, c])
        return hm

    cyl = cylinder(radius, length, axis=axis, segments=segments)
    m.extend(cyl)
    top = hemisphere(+1.0)
    m.extend(top, offset=tuple(half if k == ax else 0.0 for k in range(3)))
    bot = hemisphere(-1.0)
    m.extend(bot, offset=tuple(-half if k == ax else 0.0 for k in range(3)))

    if center != (0.0, 0.0, 0.0):
        shifted = Mesh(indices=list(m.indices), normals=list(m.normals))
        shifted.positions = [(p[0] + center[0], p[1] + center[1], p[2] + center[2]) for p in m.positions]
        return shifted
    return m


def rounded_box(sx: float, sy: float, sz: float, corner_radius: float,
                 center: Vec3 = (0.0, 0.0, 0.0), segments: int = 6) -> Mesh:
    """Box with the 4 VERTICAL edges chamfered by a quarter-cylinder of ``corner_radius``
    (rounded on X/Y, sharp top/bottom on Z) -- used for the O2 tank body silhouette.
    Falls back to a plain box if the radius is non-positive or too large for the footprint.
    """
    if corner_radius <= 1e-6 or corner_radius * 2.0 >= min(sx, sy):
        return box(sx, sy, sz, center)

    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    r = corner_radius
    cx, cy, cz = center
    m = Mesh()

    # Inner rectangle corner centers (the axis of each rounding quarter-cylinder).
    corner_centers = [
        (hx - r, hy - r), (-(hx - r), hy - r), (-(hx - r), -(hy - r)), (hx - r, -(hy - r)),
    ]
    # Perimeter ring at bottom and top: for each of the 4 corners, sweep 90 degrees.
    def ring(z: float) -> List[Vec3]:
        pts: List[Vec3] = []
        start_angles = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
        for (ccx, ccy), a0 in zip(corner_centers, start_angles):
            for k in range(segments + 1):
                a = a0 + (math.pi / 2.0) * (k / segments)
                pts.append((cx + ccx + r * math.cos(a), cy + ccy + r * math.sin(a), cz + z))
        return pts

    bottom = ring(-hz)
    top = ring(hz)
    n = len(bottom)
    # Side wall quads with per-corner outward normals (approximate radial normal).
    for i in range(n):
        j = (i + 1) % n
        bx0, by0, _ = bottom[i]
        bx1, by1, _ = bottom[j]
        tx0, ty0, _ = top[i]
        tx1, ty1, _ = top[j]
        nx0, ny0 = bx0 - cx, by0 - cy
        nlen0 = math.sqrt(nx0 * nx0 + ny0 * ny0) or 1.0
        n0 = (nx0 / nlen0, ny0 / nlen0, 0.0)
        nx1, ny1 = bx1 - cx, by1 - cy
        nlen1 = math.sqrt(nx1 * nx1 + ny1 * ny1) or 1.0
        n1 = (nx1 / nlen1, ny1 / nlen1, 0.0)
        base = len(m.positions)
        m.positions.extend([(bx0, by0, -hz + cz), (bx1, by1, -hz + cz), (tx1, ty1, hz + cz), (tx0, ty0, hz + cz)])
        m.normals.extend([n0, n1, n1, n0])
        m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    # Top / bottom caps (fan triangulation from centroid).
    for z, normal, flip in ((hz, (0, 0, 1), False), (-hz, (0, 0, -1), True)):
        ring_pts = top if z > 0 else bottom
        base = len(m.positions)
        m.positions.append((cx, cy, cz + z))
        m.normals.append(normal)
        for p in ring_pts:
            m.positions.append(p)
            m.normals.append(normal)
        for i in range(len(ring_pts)):
            j = (i + 1) % len(ring_pts)
            if flip:
                m.indices.extend([base, base + 1 + j, base + 1 + i])
            else:
                m.indices.extend([base, base + 1 + i, base + 1 + j])
    return m


def cylinder_between(p0: Vec3, p1: Vec3, radius: float, segments: int = 12) -> Mesh:
    """Flat-capped cylinder whose two end-cap CENTERS land exactly on ``p0`` and
    ``p1``. Built as an X-axis cylinder of length |p1-p0| and rotated onto the p0->p1
    direction via quat_from_to (the shortest-arc quaternion helper already
    numerically verified in quat_math.py) -- deliberately NO hand-written rotation
    matrix: a wrong-signed R_y matrix in the original handrail code inverted the
    sloped rail's direction and shipped a rail descending INTO the staircase (see the
    2026-07-07 handrail incident in the bake report). Endpoint-driven construction
    makes the intent (\"a bar from A to B\") directly checkable: the module self-test
    and the baker's handrail self-check both verify transformed end-cap centers.
    """
    from quat_math import quat_from_to, quat_to_matrix

    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if length < 1e-9:
        raise ValueError(f"cylinder_between: degenerate segment p0={p0} p1={p1}")
    cyl = cylinder(radius, length, axis="x", segments=segments)
    rot = quat_to_matrix(quat_from_to((1.0, 0.0, 0.0), d))
    mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, (p0[2] + p1[2]) / 2.0)
    return cyl.transform(rot, mid)


def combine(*meshes: Mesh) -> Mesh:
    out = Mesh()
    for mesh in meshes:
        out.extend(mesh)
    return out
