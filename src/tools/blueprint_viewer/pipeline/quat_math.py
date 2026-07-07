"""Quaternion / rotation helpers shared by the geometry builder and animation baker.

Internal convention: quaternions are plain 4-tuples in **(w, x, y, z)** order (matching
the source data's ``quat_order: wxyz`` and the URDF/robotics convention) everywhere in
this pipeline EXCEPT at the final glTF accessor write, where ``wxyz_to_xyzw`` converts
to glTF's required (x, y, z, w) order. Keeping one order internally and converting only
at the boundary avoids the classic wxyz/xyzw mix-up bug.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)


def quat_normalize(q: Sequence[float]) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return IDENTITY_QUAT
    return (w / n, x / n, y / n, z / n)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    """Hamilton product a*b, both (w,x,y,z). Composition order: applying quat_mul(a, b)
    to a vector means "rotate by b first, then by a" (standard quaternion composition,
    same convention as matrix multiplication R_a @ R_b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return IDENTITY_QUAT
    ax, ay, az = ax / n, ay / n, az / n
    half = angle / 2.0
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def quat_from_matrix(m: Sequence[Sequence[float]]) -> Quat:
    """3x3 row-major rotation matrix -> quaternion (w,x,y,z). Standard Shepperd's method."""
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize((w, x, y, z))


def quat_to_matrix(q: Sequence[float]) -> List[List[float]]:
    w, x, y, z = quat_normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_rotate_vec(q: Sequence[float], v: Vec3) -> Vec3:
    m = quat_to_matrix(q)
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


def wxyz_to_xyzw(q: Sequence[float]) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    return (x, y, z, w)


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_length(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec_normalize(a: Vec3) -> Vec3:
    n = vec_length(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def quat_from_to(a: Vec3, b: Vec3) -> Quat:
    """Shortest-arc rotation quaternion that rotates unit-ish vector ``a`` onto
    unit-ish vector ``b`` (both are normalized internally; zero-length input returns
    IDENTITY_QUAT). Standard cross/dot construction, with the antiparallel (dot~=-1)
    special case handled by picking an arbitrary perpendicular axis (any 180-degree
    rotation axis works when a and b point exactly opposite ways)."""
    a = vec_normalize(a)
    b = vec_normalize(b)
    if a == (0.0, 0.0, 0.0) or b == (0.0, 0.0, 0.0):
        return IDENTITY_QUAT
    d = vec_dot(a, b)
    if d > 1.0 - 1e-9:
        return IDENTITY_QUAT
    if d < -1.0 + 1e-9:
        # a and b are antiparallel: any axis perpendicular to a gives a valid 180 deg
        # rotation. Pick the axis via cross with the "least aligned" world basis vector
        # to avoid a degenerate (near-zero) cross product.
        perp = vec_cross(a, (1.0, 0.0, 0.0))
        if vec_length(perp) < 1e-6:
            perp = vec_cross(a, (0.0, 1.0, 0.0))
        axis = vec_normalize(perp)
        return quat_from_axis_angle(axis, math.pi)
    axis = vec_cross(a, b)
    w = 1.0 + d
    return quat_normalize((w, axis[0], axis[1], axis[2]))


def quat_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_slerp(a: Sequence[float], b: Sequence[float], t: float) -> Quat:
    """Shortest-path slerp between two (w,x,y,z) quats, t in [0,1]."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = quat_dot(a, b)
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    d = min(1.0, max(-1.0, d))
    if d > 0.9995:
        # Nearly identical: linear-interpolate and normalize (avoids /sin(theta)~0).
        lerp = tuple(a[i] + t * (b[i] - a[i]) for i in range(4))
        return quat_normalize(lerp)
    theta0 = math.acos(d)
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s0 = math.cos(theta) - d * math.sin(theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return quat_normalize(tuple(s0 * a[i] + s1 * b[i] for i in range(4)))


def fix_quat_key_signs(quats: List[Quat]) -> List[Quat]:
    """Sign-flip consecutive keyframe quaternions so dot(q[i], q[i+1]) >= 0.

    glTF quaternion samplers LINEAR-interpolate the raw (x,y,z,w) components; because
    q and -q represent the same rotation but interpolate along different arcs, a sign
    flip between consecutive keys makes the LINEAR sampler take the "long way round"
    (a visible spin/pop). Walking the sequence and flipping each key to be closest to
    the previous one removes that discontinuity while representing the identical motion.
    """
    if not quats:
        return quats
    out = [quat_normalize(quats[0])]
    for q in quats[1:]:
        q = quat_normalize(q)
        prev = out[-1]
        if quat_dot(prev, q) < 0.0:
            q = (-q[0], -q[1], -q[2], -q[3])
        out.append(q)
    return out
