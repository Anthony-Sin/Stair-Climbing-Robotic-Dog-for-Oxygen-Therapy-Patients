"""Vector math + USD transform helpers for the final-scene Isaac mount.

Split out of ``isaac_mount.py`` (behavior-preserving structural move). The pure
``Vec3`` math helpers and the small local EMA (``_damp``) are dependency-free; the
two USD helpers (``_apply_transform`` / ``_set_camera_look_at``) import ``pxr``
lazily inside their bodies exactly as before, so importing this module stays cheap
under plain Python.
"""

from __future__ import annotations

import math
from typing import List

from .common import Vec3


def _asset_uri(path: str) -> str:
    return path.replace("\\", "/")


def _apply_transform(stage, prim_path: str, translate, rotate_z_deg: float, scale: float) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if abs(float(rotate_z_deg)) > 1e-9:
        xf.AddRotateZOp().Set(float(rotate_z_deg))
    if abs(float(scale) - 1.0) > 1e-9:
        xf.AddScaleOp().Set(Gf.Vec3d(float(scale), float(scale), float(scale)))


def _set_camera_look_at(stage, prim_path: str, eye, target) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xform.AddTransformOp().Set(view_matrix.GetInverse())


def _as_vec3(value) -> Vec3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vec3, amount: float) -> Vec3:
    return (v[0] * amount, v[1] * amount, v[2] * amount)


def _length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    mag = _length(v)
    if mag <= 1.0e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _round_vec3(value: Vec3) -> List[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4), round(float(value[2]), 4)]


def _damp(current, target, tau: float, dt: float):
    """Small local EMA used by the final-scene recording cameras."""
    if current is None:
        return target
    if float(tau) <= 0.0:
        return target
    alpha = 1.0 - math.exp(-max(0.0, float(dt)) / max(1.0e-6, float(tau)))
    if isinstance(target, (list, tuple)):
        return tuple(
            float(current[i]) + (float(target[i]) - float(current[i])) * alpha
            for i in range(len(target))
        )
    return float(current) + (float(target) - float(current)) * alpha
