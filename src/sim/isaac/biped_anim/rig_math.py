"""Small numpy rotation helpers for the BipedRig gait adapter.

Split out of ``rig`` (Phase 2 structural move). Column-vector convention throughout
(v' = R @ v). Two helpers take a ``Gf`` matrix/quaternion as an opaque argument (they
only call methods / index it), so this module needs no pxr import.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------------- #
# Small numpy rotation helpers (column-vector convention: v' = R @ v).           #
# ----------------------------------------------------------------------------- #
def _gf_mat4_to_np(m) -> np.ndarray:
    return np.array([[float(m[i][j]) for j in range(4)] for i in range(4)], dtype=float)


def _mat3_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Column-convention rotation matrix -> quaternion (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def _quat_wxyz_to_mat3(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Quaternion (w, x, y, z) -> column-convention rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_angle_to_mat3(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation (column convention)."""
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3)
    k = axis / n
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rot_col_from_gf_mat4(m) -> np.ndarray:
    """Column-convention rotation matrix from a Gf.Matrix4d (USD is row-major,
    points transform as p' = p*M, so the column rotation is the transpose)."""
    A = _gf_mat4_to_np(m)[:3, :3]
    return A.T


def _gf_quat_to_wxyz(q) -> Tuple[float, float, float, float]:
    im = q.GetImaginary()
    return (float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))
