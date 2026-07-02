"""Low-level USD xform/pose helpers for the simulated patient.

Places a prim's root translate/rotate (respecting pivot-based op stacks and orient
quaternions) and finds the first ``SkelRoot`` under a subtree. Split out of
``sim_person_actor``; imported by name from the sibling actor module and re-exported
by the ``sim_person_actor`` facade.
"""
import math
from typing import Any, Optional

import numpy as np
import omni
from pxr import Gf, Usd, UsdGeom


def _set_xform_pose(
    prim_path: str,
    position: np.ndarray,
    yaw_rad: float,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)

    translate_op = None
    rotate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        # A referenced skinned character can carry a pivot-based op stack
        # (xformOp:translate:pivot plus its !invert! pair). USD rejects Set() on an
        # inverse op, and a :pivot op is the rotation/scale pivot -- not the root
        # placement -- so drive only the PRIMARY translate/rotate ops.
        if op.IsInverseOp():
            continue
        if op.GetOpName().endswith(":pivot"):
            continue
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if rotate_op is not None:
        rotate_op.Set(
            Gf.Vec3f(
                math.degrees(roll_rad),
                math.degrees(pitch_rad),
                math.degrees(yaw_rad),
            )
        )
    elif orient_op is not None:
        orient_op.Set(_yaw_quat_for_orient_op(orient_op, yaw_rad))
    else:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))


def _yaw_quat_for_orient_op(orient_op: UsdGeom.XformOp, yaw_rad: float):
    half_yaw = yaw_rad * 0.5
    real = float(math.cos(half_yaw))
    z_imag = float(math.sin(half_yaw))

    try:
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    try:
        attr_type = str(orient_op.GetAttr().GetTypeName()).lower()
        if "quatf" in attr_type:
            return Gf.Quatf(real, 0.0, 0.0, z_imag)
    except Exception:
        pass

    return Gf.Quatd(real, 0.0, 0.0, z_imag)


def _find_first_skel_root(stage: Any, parent_path: str) -> Optional[Any]:
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return None
    for prim in Usd.PrimRange(parent):
        if prim.GetTypeName() == "SkelRoot":
            return prim
    return None
