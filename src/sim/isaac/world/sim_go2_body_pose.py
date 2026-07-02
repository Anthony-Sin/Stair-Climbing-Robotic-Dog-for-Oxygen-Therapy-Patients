"""Go2 base-link rigid-body discovery + pose/orientation math helpers.

Split out of ``sim_go2_locomotion`` (the facade re-exports these). Locates the
articulation's base RigidBodyAPI prim (kinematic-aware), extracts roll/pitch/yaw
from a world transform, and clamps scalars. No dependency back on the facade.
"""
import math
from typing import Any, List, Optional, Tuple

from pxr import Usd, UsdPhysics


def _attr_is_valid(attr: Any) -> bool:
    try:
        return attr is not None and attr.IsValid()
    except Exception:
        return False


def _prim_has_rigid_body(prim: Any) -> bool:
    try:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
    except Exception:
        pass
    try:
        return _attr_is_valid(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr())
    except Exception:
        return False


def _prim_is_kinematic(prim: Any) -> bool:
    """Return True if the prim's RigidBody is set to kinematic mode."""
    try:
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        attr = rb_api.GetKinematicEnabledAttr()
        if attr and attr.IsValid():
            return bool(attr.Get())
    except Exception:
        pass
    return False


def _candidate_rigid_body_prims(go2: Any, base_link_name: str) -> List[Any]:
    root_prim = getattr(go2, "prim", None)
    if root_prim is None:
        return []

    candidates: List[Any] = [root_prim]
    try:
        stage = root_prim.GetStage()
        root_path = str(root_prim.GetPath())
        for child_name in (base_link_name, "trunk", "base", "base_link"):
            child = stage.GetPrimAtPath(f"{root_path}/{child_name}")
            if child and child.IsValid() and child not in candidates:
                candidates.append(child)
    except Exception:
        pass

    try:
        for prim in Usd.PrimRange(root_prim):
            if prim != root_prim and prim.IsValid() and prim not in candidates:
                candidates.append(prim)
    except Exception:
        pass

    return candidates


def _find_rigid_body_api(go2: Any, base_link_name: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Find the base link prim and its RigidBodyAPI.

    For kinematic bodies (used with xform-based locomotion) we return the prim
    without trying to create/use velocity attributes — PhysX rejects velocity
    calls on kinematic bodies with a hard error.
    """
    for prim in _candidate_rigid_body_prims(go2, base_link_name):
        if not _prim_has_rigid_body(prim):
            continue
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        # Kinematic bodies: return directly — the caller uses xform for position control.
        if _prim_is_kinematic(prim):
            return prim, rb_api
        vel_attr = rb_api.GetVelocityAttr()
        angular_attr = rb_api.GetAngularVelocityAttr()
        if not _attr_is_valid(vel_attr):
            vel_attr = rb_api.CreateVelocityAttr()
        if not _attr_is_valid(angular_attr):
            angular_attr = rb_api.CreateAngularVelocityAttr()
        if _attr_is_valid(vel_attr) and _attr_is_valid(angular_attr):
            return prim, rb_api

    # Robust fallback: Search Usd.PrimRange for any prim matching base_link_name or 'trunk'
    root_prim = getattr(go2, "prim", None)
    if root_prim is not None:
        for prim in Usd.PrimRange(root_prim):
            name = prim.GetName().lower()
            if name == base_link_name.lower() or name == "trunk":
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                if _prim_is_kinematic(prim):
                    return prim, rb_api
                vel_attr = rb_api.GetVelocityAttr()
                angular_attr = rb_api.GetAngularVelocityAttr()
                if not _attr_is_valid(vel_attr):
                    rb_api.CreateVelocityAttr()
                if not _attr_is_valid(angular_attr):
                    rb_api.CreateAngularVelocityAttr()
                return prim, rb_api

    return None, None


def _extract_roll_pitch_yaw(matrix: Any) -> Tuple[float, float, float]:
    r00 = float(matrix[0][0])
    r01 = float(matrix[0][1])
    r02 = float(matrix[0][2])
    r12 = float(matrix[1][2])
    r22 = float(matrix[2][2])
    yaw = math.atan2(r01, r00)
    pitch = math.atan2(-r02, max(1e-6, math.sqrt((r00 * r00) + (r01 * r01))))
    roll = math.atan2(r12, r22)
    return roll, pitch, yaw


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
