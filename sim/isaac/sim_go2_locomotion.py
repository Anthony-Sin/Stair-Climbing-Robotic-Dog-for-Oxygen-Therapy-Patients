import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from omni.isaac.core.utils.types import ArticulationAction
except ModuleNotFoundError:
    from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from sim_logging_utils import log_event


@dataclass
class Go2LocomotionState:
    target_height_m: float = 0.32
    height_kp: float = 3.5
    attitude_kp: float = 5.0
    max_vertical_speed_mps: float = 0.8
    max_attitude_rate_rps: float = 1.8
    gait_phase: float = 0.0
    stand_joint_positions: Optional[np.ndarray] = None
    dof_names: List[str] = field(default_factory=list)
    rigid_body_path: str = ""
    rigid_body_logged: bool = False
    gait_logged: bool = False
    stair_hold_logged: bool = False
    procedural_gait_unavailable: bool = False
    procedural_gait_failure_count: int = 0
    warning_times: Dict[str, float] = field(default_factory=dict)
    stable_hold_logged: bool = False

    # Physics gait parameters
    use_physics_gait: bool = False
    gait_time: float = 0.0
    gait_period: float = 0.6
    duty_factor: float = 0.5
    swing_height: float = 0.06

    # Posture balance PD gains
    kp_height: float = 1.2
    kd_height: float = 0.15
    kp_roll: float = 1.0
    kd_roll: float = 0.05
    kp_pitch: float = 1.0
    kd_pitch: float = 0.05

    # Tracking states
    joint_gains_set: bool = False
    joint_gains_unavailable: bool = False
    dof_map: Dict[Tuple[str, str], int] = field(default_factory=dict)
    lift_off_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    last_foot_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    last_phases: Dict[str, float] = field(default_factory=dict)
    stair_demo_telemetry: Dict[str, Any] = field(default_factory=dict)
    current_swing_legs: List[str] = field(default_factory=list)
    stair_demo_detected_logged: bool = False
    stair_demo_climb_logged: bool = False
    stair_demo_complete_logged: bool = False
    stair_crawl_logged: bool = False


def solve_leg_ik(x: float, y: float, z: float, leg: str) -> Tuple[float, float, float]:
    """
    Solve Inverse Kinematics for a 3-DOF Go2 leg.
    
    Args:
        x, y, z: Foot position relative to the hip roll joint center.
        leg: 'fl', 'fr', 'rl', or 'rr'.
        
    Returns:
        (q_hip, q_thigh, q_calf) joint angles in radians.
    """
    l_hip = 0.0955 if leg in ["fl", "rl"] else -0.0955
    l_thigh = 0.213
    l_calf = 0.213
    
    # 1. Abduction/Adduction (hip roll) joint: q_hip
    r = math.sqrt(y**2 + z**2)
    if r < abs(l_hip):
        r = abs(l_hip)
        
    term = l_hip / r
    term = max(-1.0, min(1.0, term))
    q_hip = math.atan2(z, y) + math.acos(term)
    
    # Rotate target to the leg pitch plane (around X by -q_hip)
    cos_q = math.cos(q_hip)
    sin_q = math.sin(q_hip)
    
    x_leg = x
    z_leg = -y * sin_q + z * cos_q
    
    # Distance from thigh joint to foot
    d = math.sqrt(x_leg**2 + z_leg**2)
    if d < 1e-4:
        d = 1e-4
    
    # Law of Cosines for calf joint q_calf
    term_calf = (d**2 - l_thigh**2 - l_calf**2) / (2.0 * l_thigh * l_calf)
    term_calf = max(-1.0, min(1.0, term_calf))
    q_calf = -math.acos(term_calf)
    
    # Law of Cosines for thigh joint q_thigh
    alpha = math.atan2(x_leg, -z_leg)
    term_thigh = (l_thigh**2 + d**2 - l_calf**2) / (2.0 * l_thigh * d)
    term_thigh = max(-1.0, min(1.0, term_thigh))
    beta = math.acos(term_thigh)
    
    q_thigh = alpha + beta
    
    return q_hip, q_thigh, q_calf



def _warn_rate_limited(
    logger: Optional[logging.Logger],
    state: Go2LocomotionState,
    key: str,
    message: str,
    *,
    interval_sec: float = 2.0,
    **fields: Any,
) -> None:
    if logger is None:
        return
    now = time.monotonic()
    last = state.warning_times.get(key, 0.0)
    if now - last < interval_sec:
        return
    state.warning_times[key] = now
    log_event(logger, logging.WARNING, key, message, **fields)


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


STAIR_START_X_M = 2.0
STAIR_STEP_DEPTH_M = 0.3
STAIR_STEP_HEIGHT_M = 0.08
STAIR_STEP_COUNT = 12
STAIR_END_X_M = STAIR_START_X_M + STAIR_STEP_COUNT * STAIR_STEP_DEPTH_M
STAIR_TOP_HEIGHT_M = STAIR_STEP_COUNT * STAIR_STEP_HEIGHT_M
STAIR_HALF_WIDTH_M = 1.05
STAIR_LIDAR_LOOKAHEAD_M = 0.85
STAIR_LIDAR_SAMPLE_RANGES_M = (0.15, 0.30, 0.45, 0.60, 0.75)


def _terrain_phase(x: float, y: float) -> str:
    if abs(y) > STAIR_HALF_WIDTH_M:
        return "off_route"
    if x < STAIR_START_X_M - 0.35:
        return "flat_follow"
    if x < STAIR_START_X_M:
        return "stair_approach"
    if x < STAIR_END_X_M:
        return "staircase"
    return "top_landing"


def _next_stair_edge(x: float, y: float) -> Tuple[Optional[float], float, float]:
    if abs(y) > STAIR_HALF_WIDTH_M:
        current_height = _get_analytical_terrain_height(x, y)
        return None, current_height, current_height
    if x < STAIR_START_X_M:
        return STAIR_START_X_M, 0.0, STAIR_STEP_HEIGHT_M
    if x < STAIR_END_X_M:
        step_idx = int((x - STAIR_START_X_M) / STAIR_STEP_DEPTH_M)
        next_edge = STAIR_START_X_M + (step_idx + 1) * STAIR_STEP_DEPTH_M
        current_height = min(STAIR_TOP_HEIGHT_M, (step_idx + 1) * STAIR_STEP_HEIGHT_M)
        next_height = min(STAIR_TOP_HEIGHT_M, (step_idx + 2) * STAIR_STEP_HEIGHT_M)
        return next_edge, current_height, next_height
    return None, STAIR_TOP_HEIGHT_M, STAIR_TOP_HEIGHT_M


def _stair_assisted_world_z(
    rx: float,
    ry: float,
    yaw: float,
    vx: float,
    state: Go2LocomotionState,
    clearance_m: float,
    stairs_detected: bool = False,
) -> float:
    if stairs_detected:
        forward_lookahead_m = 0.0
        if vx > 0.02:
            forward_lookahead_m = 0.18 + min(0.24, vx * 0.35)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        current_h = _get_analytical_terrain_height(rx, ry)
        ahead_h = _get_analytical_terrain_height(
            rx + cos_y * forward_lookahead_m,
            ry + sin_y * forward_lookahead_m,
        )
        target_terrain_h = max(current_h, ahead_h)
    else:
        target_terrain_h = 0.0
    return target_terrain_h + max(0.24, clearance_m)


def _build_stair_demo_telemetry(
    rx: float,
    ry: float,
    rz: float,
    roll: float,
    pitch: float,
    yaw: float,
    actual_height: float,
    vx: float,
    vy: float,
    wz: float,
    state: Go2LocomotionState,
    *,
    body_height_target_m: Optional[float],
    vertical_assist_mps: float,
) -> Dict[str, Any]:
    phase = _terrain_phase(rx, ry)
    edge_x, current_h, next_h = _next_stair_edge(rx, ry)
    distance_to_step_m = None if edge_x is None else max(0.0, edge_x - rx)
    step_delta_m = max(0.0, next_h - current_h)
    detected = bool(
        phase in ("staircase", "top_landing")
        or (
            distance_to_step_m is not None
            and distance_to_step_m <= STAIR_LIDAR_LOOKAHEAD_M
            and step_delta_m >= 0.02
        )
    )

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    samples = []
    for idx, range_m in enumerate(STAIR_LIDAR_SAMPLE_RANGES_M):
        sx = rx + cos_y * range_m
        sy = ry + sin_y * range_m
        samples.append(
            {
                "range_m": round(float(range_m), 3),
                "x_m": round(float(sx), 3),
                "y_m": round(float(sy), 3),
                "elevation_m": round(float(_get_analytical_terrain_height(sx, sy)), 3),
                "timestamp_offset_ms": int(idx * 12),
            }
        )

    command_speed = math.sqrt((vx * vx) + (vy * vy)) + 0.25 * abs(wz)
    if phase == "top_landing":
        rl_mode = "landing_follow"
    elif phase == "staircase":
        rl_mode = "stair_climb"
    elif detected:
        rl_mode = "stair_approach"
    else:
        rl_mode = "flat_follow"
    rl_active = bool(rl_mode in ("stair_approach", "stair_climb") and command_speed > 0.03)
    confidence = 0.96 if phase == "staircase" else 0.91 if detected else 0.42

    robot_fell = False
    robot_fall_type = "upright"
    if abs(roll) > 1.05 or abs(pitch) > 1.05:
        robot_fell = True
        robot_fall_type = "flipped over"
    elif actual_height < 0.18:
        robot_fell = True
        robot_fall_type = "collapsed"

    return {
        "source": "synthetic_isaac_ground_truth_raycast",
        "is_synthetic": True,
        "data_truth": "sim_geometry_exact_for_demo_not_hardware_lidar",
        "phase": phase,
        "robot": {
            "x_m": round(float(rx), 3),
            "y_m": round(float(ry), 3),
            "z_m": round(float(rz), 3),
            "roll_deg": round(float(math.degrees(roll)), 2),
            "pitch_deg": round(float(math.degrees(pitch)), 2),
            "yaw_deg": round(float(math.degrees(yaw)), 2),
            "height_m": round(float(actual_height), 3),
            "fell": bool(robot_fell),
            "fall_type": str(robot_fall_type),
        },
        "lidar": {
            "model": "demo_4d_elevation_raycast",
            "ray_count": len(samples),
            "lookahead_m": round(float(STAIR_LIDAR_LOOKAHEAD_M), 2),
            "detected": detected,
            "confidence": round(float(confidence), 2),
            "distance_to_next_riser_m": (
                None if distance_to_step_m is None else round(float(distance_to_step_m), 3)
            ),
            "step_height_m": round(float(step_delta_m if step_delta_m > 0.0 else (STAIR_STEP_HEIGHT_M if phase == "staircase" else 0.0)), 3),
            "current_ground_m": round(float(current_h), 3),
            "next_ground_m": round(float(next_h), 3),
            "samples": samples,
        },
        "blind_rl": {
            "policy": "synthetic_blind_rl_stair_assist",
            "is_synthetic": True,
            "mode": rl_mode,
            "active": rl_active,
            "gait_pattern": (
                "single_leg_stair_crawl" if rl_mode in ("stair_approach", "stair_climb") else "diagonal_flat_trot"
            ),
            "swing_legs": [str(leg).upper() for leg in state.current_swing_legs],
            "leg_commands": _build_leg_command_summary(state, rl_mode, command_speed),
            "commanded_speed_mps": round(float(command_speed), 3),
            "body_height_target_m": (
                None if body_height_target_m is None else round(float(body_height_target_m), 3)
            ),
            "vertical_assist_mps": round(float(vertical_assist_mps), 3),
            "stair_slope_deg": round(float(math.degrees(math.atan2(STAIR_STEP_HEIGHT_M, STAIR_STEP_DEPTH_M))), 2),
            "foot_clearance_m": round(float(_effective_swing_height_for_mode(rl_mode)), 3),
            "physics_contact_enabled": True,
            "collision_cheat": "none",
        },
    }


def _effective_swing_height_for_mode(mode: str) -> float:
    if mode == "stair_climb":
        return 0.11
    if mode == "stair_approach":
        return 0.08
    return 0.06


def _effective_swing_height(state: Go2LocomotionState) -> float:
    mode = (
        state.stair_demo_telemetry
        .get("blind_rl", {})
        .get("mode", "flat_follow")
    )
    return max(float(state.swing_height), _effective_swing_height_for_mode(str(mode)))


def _build_leg_command_summary(
    state: Go2LocomotionState,
    rl_mode: str,
    command_speed: float,
) -> Dict[str, Dict[str, Any]]:
    stair_mode = rl_mode in ("stair_approach", "stair_climb")
    swing_height = _effective_swing_height_for_mode(rl_mode)
    drive_mps = max(0.0, min(0.85, float(command_speed)))
    swing_legs = {str(leg).lower() for leg in state.current_swing_legs}
    commands: Dict[str, Dict[str, Any]] = {}
    for leg in ("fl", "fr", "rl", "rr"):
        is_swing = leg in swing_legs
        if stair_mode:
            action = "STEP_UP" if is_swing else "LOAD_HOLD"
            foot_z = swing_height if is_swing else 0.0
        else:
            action = "SWING" if is_swing else "STANCE"
            foot_z = min(swing_height, 0.06) if is_swing else 0.0
        commands[leg.upper()] = {
            "state": "swing" if is_swing else "stance",
            "action": action,
            "foot_lift_m": round(float(foot_z), 3),
            "drive_mps": round(float(drive_mps if is_swing else 0.0), 3),
            "contact_expected": not is_swing,
        }
    return commands


def _leg_from_joint_name(name: str) -> Optional[str]:
    name_l = name.lower()
    for leg in ("fl", "fr", "rl", "rr"):
        if leg in name_l:
            return leg
    return None


def _gait_mode_for_terrain(terrain_phase: str) -> str:
    if terrain_phase in ("stair_approach", "staircase"):
        return "stair_crawl"
    return "flat_trot"


def _gait_timing_for_mode(mode: str) -> Tuple[float, float, Dict[str, float]]:
    if mode == "stair_crawl":
        duty = 0.78
        return (
            1.15,
            duty,
            {
                "fl": duty,
                "fr": duty - 0.25,
                "rl": duty - 0.50,
                "rr": duty - 0.75,
            },
        )
    return (
        0.6,
        0.5,
        {
            "fl": 0.0,
            "rr": 0.0,
            "fr": 0.5,
            "rl": 0.5,
        },
    )


def _phase_is_swing(phase: float, duty_factor: float) -> bool:
    return phase >= duty_factor


def _swing_progress(phase: float, duty_factor: float) -> float:
    if duty_factor >= 0.98:
        return 1.0
    return _clamp((phase - duty_factor) / max(1e-6, 1.0 - duty_factor), 0.0, 1.0)


def _record_stair_demo_telemetry(
    state: Go2LocomotionState,
    logger: Optional[logging.Logger],
    telemetry: Dict[str, Any],
) -> None:
    state.stair_demo_telemetry = telemetry
    lidar = telemetry.get("lidar", {})
    blind_rl = telemetry.get("blind_rl", {})
    phase = str(telemetry.get("phase", "unknown"))
    if logger is None:
        return

    if lidar.get("detected") and not state.stair_demo_detected_logged:
        state.stair_demo_detected_logged = True
        log_event(
            logger,
            logging.INFO,
            "synthetic_lidar_stairs_detected",
            "Synthetic 4D elevation raycast detected the staircase from Isaac ground-truth geometry",
            data_source=telemetry.get("source"),
            distance_to_next_riser_m=lidar.get("distance_to_next_riser_m"),
            step_height_m=lidar.get("step_height_m"),
            is_synthetic=True,
        )

    if blind_rl.get("active") and not state.stair_demo_climb_logged:
        state.stair_demo_climb_logged = True
        log_event(
            logger,
            logging.INFO,
            "synthetic_blind_rl_stair_assist_active",
            "Synthetic blind-RL stair assist is lifting the Go2 body while preserving the existing gait loop",
            mode=blind_rl.get("mode"),
            body_height_target_m=blind_rl.get("body_height_target_m"),
            vertical_assist_mps=blind_rl.get("vertical_assist_mps"),
            physics_contact_enabled=True,
            is_synthetic=True,
        )

    if phase == "top_landing" and not state.stair_demo_complete_logged:
        state.stair_demo_complete_logged = True
        log_event(
            logger,
            logging.INFO,
            "synthetic_stair_demo_top_landing",
            "Go2 reached the top-landing region under synthetic stair perception/climb assist",
            is_synthetic=True,
        )


def get_stair_demo_telemetry(state: Go2LocomotionState) -> Dict[str, Any]:
    return dict(state.stair_demo_telemetry)


def _set_stable_kinematic_pose(
    go2: Any,
    *,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    target_height_m: float,
    stairs_detected: bool = False,
) -> None:
    # Force straight line movement along Y=0 and yaw=0
    vy = 0.0
    wz = 0.0
    xformable = UsdGeom.Xformable(go2.prim)
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Ensure yaw is perfectly locked to 0.0 and Y is perfectly locked to 0.0 to prevent drifting/rotation
    new_yaw = 0.0
    tx = float(matrix[3][0]) + (vx * dt)
    ty = 0.0
    terrain_h = _get_analytical_terrain_height(tx, ty) if stairs_detected else 0.0
    tz = terrain_h + target_height_m

    # Reset linear and angular velocities to prevent dynamic bodies
    # from accumulating gravity momentum while being teleported.
    try:
        if hasattr(go2, "set_linear_velocity"):
            go2.set_linear_velocity(np.zeros(3))
        if hasattr(go2, "set_angular_velocity"):
            go2.set_angular_velocity(np.zeros(3))
        if hasattr(go2, "set_world_pose"):
            qw = math.cos(new_yaw * 0.5)
            qx = 0.0
            qy = 0.0
            qz = math.sin(new_yaw * 0.5)
            go2.set_world_pose(
                position=np.array([tx, ty, tz]),
                orientation=np.array([qw, qx, qy, qz]),
            )
            return
    except Exception:
        pass

    translate_op = None
    rotate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(tx, ty, tz))

    if rotate_op is not None:
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(new_yaw)))
    elif orient_op is not None:
        orient_op.Set(_yaw_quat_for_orient_op(orient_op, new_yaw))
    else:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(new_yaw)))


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


def hold_go2_stable(
    go2: Any,
    state: Go2LocomotionState,
    dt: float,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    _set_stable_kinematic_pose(
        go2,
        vx=0.0,
        vy=0.0,
        wz=0.0,
        dt=dt,
        target_height_m=state.target_height_m,
    )
    _apply_procedural_gait(
        go2,
        state,
        vx=0.0,
        vy=0.0,
        wz=0.0,
        dt=dt,
        logger=logger,
    )
    if logger is not None and not state.stable_hold_logged:
        state.stable_hold_logged = True
        log_event(
            logger,
            logging.INFO,
            "go2_stable_hold_active",
            "Holding Go2 upright until the Docker/controller command stream starts",
            target_height_m=state.target_height_m,
        )


def _get_dof_names(go2: Any) -> List[str]:
    for attr_name in ("dof_names", "joint_names"):
        names = getattr(go2, attr_name, None)
        if names:
            return [str(name) for name in names]
    for method_name in ("get_dof_names", "get_joint_names"):
        method = getattr(go2, method_name, None)
        if callable(method):
            names = method()
            if names:
                return [str(name) for name in names]
    return []


def _default_stand_pose(go2: Any, dof_names: List[str]) -> Optional[np.ndarray]:
    if not dof_names:
        return None
    try:
        try:
            controller = go2.get_articulation_controller()
            current = np.asarray(controller.get_joint_positions(), dtype=float)
        except Exception:
            current = np.asarray(go2.get_joint_positions(), dtype=float)
    except Exception:
        current = np.zeros(len(dof_names), dtype=float)
    if current.shape[0] != len(dof_names):
        current = np.zeros(len(dof_names), dtype=float)

    stand = current.copy()
    for index, raw_name in enumerate(dof_names):
        name = raw_name.lower()
        if "hip" in name:
            stand[index] = 0.0
        elif "thigh" in name:
            stand[index] = 0.75
        elif "calf" in name:
            stand[index] = -1.45
    return stand


def _command_joint_positions(go2: Any, positions: np.ndarray) -> None:
    """Use the first Isaac articulation joint-position API available in this install."""
    errors = []
    for method_name in ("set_joint_position_targets", "set_joint_positions"):
        method = getattr(go2, method_name, None)
        if not callable(method):
            continue
        try:
            method(positions)
            return
        except Exception as exc:
            errors.append(f"{method_name}: {exc}")

    try:
        go2.apply_action(ArticulationAction(joint_positions=positions))
        return
    except Exception as exc:
        errors.append(f"apply_action: {exc}")

    raise RuntimeError("; ".join(errors) if errors else "no joint position command API available")


def _apply_procedural_gait(
    go2: Any,
    state: Go2LocomotionState,
    *,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    logger: Optional[logging.Logger],
) -> None:
    command_speed = math.sqrt((vx * vx) + (vy * vy)) + (0.25 * abs(wz))
    moving = command_speed > 0.03
    if moving:
        state.gait_time += dt
    else:
        state.gait_time = max(0.0, state.gait_time - (dt * 2.0))

    if state.procedural_gait_unavailable:
        return

    if state.stand_joint_positions is None:
        state.dof_names = _get_dof_names(go2)
        state.stand_joint_positions = _default_stand_pose(go2, state.dof_names)
        if state.stand_joint_positions is None:
            _warn_rate_limited(
                logger,
                state,
                "go2_gait_no_dofs",
                "Go2 procedural gait skipped because no articulation DOFs were found",
            )
            return

    positions = state.stand_joint_positions.copy()

    if moving:
        terrain_phase = str(state.stair_demo_telemetry.get("phase", "flat_follow"))
        gait_mode = _gait_mode_for_terrain(terrain_phase)
        gait_period, duty_factor, phase_offsets = _gait_timing_for_mode(gait_mode)
        if gait_mode == "stair_crawl":
            state.gait_phase += dt * (2.0 * math.pi / gait_period)
        else:
            state.gait_phase += dt * (7.0 + (4.0 * min(command_speed, 1.0)))
        swing = min(1.0, command_speed / 0.6)
        lift_scale = 1.25 if _effective_swing_height(state) > state.swing_height else 1.0
        gait_cycle = state.gait_time / max(0.2, gait_period)
        leg_phases = {
            leg: (gait_cycle + phase_offsets[leg]) % 1.0
            for leg in ("fl", "fr", "rl", "rr")
        }
        swing_legs = [
            leg
            for leg, phase in leg_phases.items()
            if _phase_is_swing(phase, duty_factor)
        ]
        if gait_mode == "stair_crawl":
            swing_legs = swing_legs[:1]
        state.current_swing_legs = swing_legs

        for index, raw_name in enumerate(state.dof_names):
            name = raw_name.lower()
            leg = _leg_from_joint_name(name)
            if gait_mode == "stair_crawl" and leg is not None:
                phase_unit = leg_phases.get(leg, 0.0)
                is_swing = leg in swing_legs
                swing_progress = _swing_progress(phase_unit, duty_factor) if is_swing else 0.0
                lift = math.sin(math.pi * swing_progress) if is_swing else 0.0
                reach = math.sin(2.0 * math.pi * swing_progress) if is_swing else 0.0
                if "hip" in name:
                    positions[index] += 0.025 * swing * reach
                elif "thigh" in name:
                    positions[index] += 0.30 * swing * lift
                    if not is_swing:
                        positions[index] -= 0.035 * swing
                elif "calf" in name:
                    positions[index] -= 0.42 * swing * lift
                    if not is_swing:
                        positions[index] += 0.025 * swing
                continue

            if "fl" in name or "rr" in name:
                phase = state.gait_phase
            else:
                phase = state.gait_phase + math.pi

            sin_phase = math.sin(phase)
            if "hip" in name:
                positions[index] += 0.05 * swing * sin_phase
            elif "thigh" in name:
                positions[index] += 0.28 * lift_scale * swing * sin_phase
            elif "calf" in name:
                positions[index] -= 0.36 * lift_scale * swing * max(0.0, sin_phase)
    else:
        state.current_swing_legs = []

    try:
        _command_joint_positions(go2, positions)
        if moving and logger is not None and not state.gait_logged:
            state.gait_logged = True
            log_event(
                logger,
                logging.INFO,
                "go2_gait_enabled",
                "Go2 same-model procedural gait animation is active",
                dof_count=len(state.dof_names),
            )
    except Exception as exc:
        state.procedural_gait_failure_count += 1
        state.procedural_gait_unavailable = True
        _warn_rate_limited(
            logger,
            state,
            "go2_gait_apply_failed",
            "Go2 procedural gait command failed; disabling joint animation and keeping stable body motion",
            interval_sec=30.0,
            failure_count=int(state.procedural_gait_failure_count),
            error=str(exc),
        )


def _get_analytical_terrain_height(x: float, y: float) -> float:
    """Return the exact terrain height at coordinate (x, y) based on spawned geometry."""
    if not (-STAIR_HALF_WIDTH_M <= y <= STAIR_HALF_WIDTH_M):
        return 0.0
    # Stairs: 12 steps from 2.0m to 5.6m, each step 0.3m deep, 0.08m rise
    if STAIR_START_X_M <= x < STAIR_END_X_M:
        step_idx = int((x - STAIR_START_X_M) / STAIR_STEP_DEPTH_M)
        return min(STAIR_TOP_HEIGHT_M, (step_idx + 1) * STAIR_STEP_HEIGHT_M)
    # Top landing
    if x >= STAIR_END_X_M:
        return STAIR_TOP_HEIGHT_M
    # Flat ground
    return 0.0


def _query_terrain_height(rx: float, ry: float, rz: float) -> float:
    """Query terrain height at (rx, ry) via PhysX raycast or local estimation fallback."""
    try:
        import omni.physx
        physx_interface = omni.physx.get_physx_interface()
        # Talk to PhysX scene raycaster directly
        hit = physx_interface.raycast_closest((rx, ry, rz), (0.0, 0.0, -1.0), 1.5)
        if hit and hit[0]:
            hit_info = hit[1]
            if hasattr(hit_info, "position"):
                return float(hit_info.position[2])
            elif isinstance(hit_info, dict) and "position" in hit_info:
                return float(hit_info["position"][2])
            elif isinstance(hit_info, (list, tuple)) and len(hit_info) >= 3:
                return float(hit_info[2])
    except Exception:
        pass
    return max(0.0, rz - 0.32)


def apply_go2_velocity(
    go2: Any,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    *,
    state: Go2LocomotionState,
    base_link_name: str = "trunk",
    logger: Optional[logging.Logger] = None,
    stairs_detected: bool = False,
) -> None:
    """Drive Go2 root motion through physics-driven trot gait or fallback kinematic mode."""
    root_prim = getattr(go2, "prim", None)
    if root_prim is None:
        _warn_rate_limited(
            logger,
            state,
            "go2_missing_root_prim",
            "Go2 velocity command skipped because the articulation root prim is missing",
        )
        return

    if state.use_physics_gait:
        try:
            # 1. Ensure joint map is set up
            if not state.dof_map:
                state.dof_names = _get_dof_names(go2)
                for index, raw_name in enumerate(state.dof_names):
                    name = raw_name.lower()
                    leg = None
                    for l in ["fl", "fr", "rl", "rr"]:
                        if l in name:
                            leg = l
                            break
                    joint_type = None
                    for j in ["hip", "thigh", "calf"]:
                        if j in name:
                            joint_type = j
                            break
                    if leg and joint_type:
                        state.dof_map[(leg, joint_type)] = index

            if len(state.dof_map) < 12:
                _warn_rate_limited(
                    logger,
                    state,
                    "go2_physics_gait_missing_dofs",
                    f"Go2 physics gait requires 12 DOFs but only found {len(state.dof_map)}; falling back to legacy",
                )
                state.use_physics_gait = False
            else:
                # 2. Configure joint gains if not set
                if not state.joint_gains_set and not state.joint_gains_unavailable:
                    try:
                        try:
                            controller = go2.get_articulation_controller()
                            for i in range(go2.num_dof):
                                controller.set_joint_drive_gains(i, stiffness=450.0, damping=25.0)
                        except Exception:
                            dof_props = go2.get_dof_properties()
                            dof_props["stiffness"] = 450.0
                            dof_props["damping"] = 25.0
                            go2.set_dof_properties(dof_props)
                        state.joint_gains_set = True
                        if logger is not None:
                            log_event(
                                logger,
                                logging.INFO,
                                "go2_joint_gains_configured",
                                "Go2 joint drive gains (stiffness=450, damping=25) have been applied",
                            )
                    except Exception as exc:
                        state.joint_gains_unavailable = True
                        state.use_physics_gait = False
                        _warn_rate_limited(
                            logger,
                            state,
                            "go2_gains_config_failed",
                            "Failed to configure Go2 joint gains; disabling physics gait and using stable fallback",
                            interval_sec=30.0,
                            error=str(exc)
                        )
                        raise RuntimeError("Go2 joint drive gains are unavailable")

                # 3. Find base rigid body to read pose & velocity
                rb_prim, rb_api = _find_rigid_body_api(go2, base_link_name)
                if rb_prim is None or rb_api is None:
                    raise RuntimeError("no rigid body API found on Go2 root or base links")

                xform = UsdGeom.Xformable(rb_prim)
                matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                roll, pitch, yaw = _extract_roll_pitch_yaw(matrix)
                
                rx = float(matrix[3][0])
                ry = float(matrix[3][1])
                rz = float(matrix[3][2])
                terrain_height = _query_terrain_height(rx, ry, rz)
                actual_height = rz - terrain_height

                actual_lin_vel = go2.get_linear_velocity()
                actual_ang_vel = go2.get_angular_velocity()
                actual_vx, actual_vy, actual_vz = actual_lin_vel[0], actual_lin_vel[1], actual_lin_vel[2]
                actual_wx, actual_wy, actual_wz = actual_ang_vel[0], actual_ang_vel[1], actual_ang_vel[2]

                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                wx = cos_y * vx - sin_y * vy
                wy = sin_y * vx + cos_y * vy
                body_vx = cos_y * actual_vx + sin_y * actual_vy
                body_vy = -sin_y * actual_vx + cos_y * actual_vy

                gait_mode = "stair_crawl" if stairs_detected else "flat_trot"

                # 4. Gait phase scheduler
                cmd_speed = math.sqrt(vx**2 + vy**2) + 0.25 * abs(wz)
                is_moving = cmd_speed > 0.03
                desired_world_z = rz
                vertical_assist_mps = 0.0

                if is_moving:
                    state.gait_time += dt
                else:
                    state.gait_time = 0.0

                T_c, duty_factor, phase_offsets = _gait_timing_for_mode(gait_mode)
                T_s = T_c * duty_factor

                legs = ["fl", "fr", "rl", "rr"]
                phases = {}
                for leg in legs:
                    if not is_moving:
                        phases[leg] = 0.0
                    else:
                        phases[leg] = (state.gait_time / T_c + phase_offsets[leg]) % 1.0
                state.current_swing_legs = [
                    leg
                    for leg in legs
                    if is_moving and _phase_is_swing(phases[leg], duty_factor)
                ]
                if gait_mode == "stair_crawl":
                    state.current_swing_legs = state.current_swing_legs[:1]
                    if logger is not None and not state.stair_crawl_logged:
                        state.stair_crawl_logged = True
                        log_event(
                            logger,
                            logging.INFO,
                            "go2_stair_crawl_gait_active",
                            "Go2 stair gait switched to a single-swing-leg crawl pattern",
                            duty_factor=float(duty_factor),
                            gait_period_s=float(T_c),
                            sequence="FL, FR, RL, RR",
                        )

                # Torso sway oscillations
                if is_moving:
                    omega = 2.0 * math.pi / T_c
                    if gait_mode == "stair_crawl":
                        sway_y = 0.004 * math.sin(omega * state.gait_time)
                        bob_z = -0.004 * abs(math.sin(omega * state.gait_time))
                        sway_roll = 0.006 * math.sin(omega * state.gait_time)
                        sway_pitch = 0.006 * math.cos(2.0 * omega * state.gait_time)
                    else:
                        sway_y = 0.015 * math.sin(omega * state.gait_time)
                        bob_z = -0.012 * abs(math.sin(omega * state.gait_time))
                        sway_roll = 0.02 * math.sin(omega * state.gait_time)
                        sway_pitch = 0.015 * math.cos(2.0 * omega * state.gait_time)
                else:
                    sway_y = 0.0
                    bob_z = 0.0
                    sway_roll = 0.0
                    sway_pitch = 0.0

                # 5. Posture/Balance PD corrections
                h_target = state.target_height_m
                dz_height = state.kp_height * (actual_height - h_target) + state.kd_height * actual_vz
                dz_height = max(-0.08, min(0.08, dz_height))

                HIP_OFFSETS = {
                    "fl": (0.1934, 0.0465),
                    "fr": (0.1934, -0.0465),
                    "rl": (-0.1934, 0.0465),
                    "rr": (-0.1934, -0.0465),
                }
                THIGH_OFFSETS_Y = {
                    "fl": 0.0955,
                    "fr": -0.0955,
                    "rl": 0.0955,
                    "rr": -0.0955,
                }
                if stairs_detected:
                    swing_height_m = max(state.swing_height, _effective_swing_height_for_mode("stair_climb"))
                else:
                    swing_height_m = state.swing_height

                # 6. Generate joint targets
                positions = np.zeros(len(state.dof_names))
                try:
                    try:
                        controller = go2.get_articulation_controller()
                        current_positions = controller.get_joint_positions()
                    except Exception:
                        current_positions = go2.get_joint_positions()
                    if current_positions is not None and len(current_positions) == len(positions):
                        positions = np.array(current_positions)
                except Exception:
                    pass

                for leg in legs:
                    x_nom = 0.0
                    y_nom = THIGH_OFFSETS_Y[leg] - sway_y
                    z_nom = -h_target - bob_z

                    # Hip offsets
                    xh, yh = HIP_OFFSETS[leg]

                    # Attitude correction (actively tracks sway roll & pitch)
                    roll_err = roll - sway_roll
                    pitch_err = pitch - sway_pitch
                    dz_attitude = yh * (state.kp_roll * roll_err + state.kd_roll * actual_wx) + xh * (state.kp_pitch * pitch_err + state.kd_pitch * actual_wy)
                    dz_attitude = max(-0.06, min(0.06, dz_attitude))

                    z_nom_corr = z_nom + dz_height + dz_attitude

                    # Hip joint velocity in body frame
                    v_hip_x = vx - wz * yh
                    v_hip_y = vy + wz * xh

                    phi = phases[leg]
                    leg_is_swing = _phase_is_swing(phi, duty_factor) and (
                        gait_mode != "stair_crawl" or leg in state.current_swing_legs
                    )
                    if not leg_is_swing:
                        # Stance phase
                        s_stance = phi / max(1e-6, duty_factor)
                        x_target = v_hip_x * (T_s / 2.0) * (1.0 - 2.0 * s_stance)
                        y_target = y_nom + v_hip_y * (T_s / 2.0) * (1.0 - 2.0 * s_stance)
                        z_target = z_nom_corr
                    else:
                        # Swing phase
                        s_swing = _swing_progress(phi, duty_factor)
                        
                        last_p = state.last_phases.get(leg, 0.0)
                        is_liftoff = (
                            (not _phase_is_swing(last_p, duty_factor) and leg_is_swing)
                            or (leg not in state.lift_off_positions)
                        )
                        if is_liftoff:
                            prev_pos = state.last_foot_positions.get(leg, (-v_hip_x * T_s / 2.0, y_nom - v_hip_y * T_s / 2.0, z_nom_corr))
                            state.lift_off_positions[leg] = prev_pos
                            
                        # Raibert touchdown heuristic with velocity feedback
                        k_v = 0.03
                        x_touchdown = v_hip_x * (T_s / 2.0) + k_v * (body_vx - vx)
                        y_touchdown = y_nom + v_hip_y * (T_s / 2.0) + k_v * (body_vy - vy)
                        
                        # Clamp landing target
                        x_touchdown = max(-0.15, min(0.15, x_touchdown))
                        y_touchdown = max(y_nom - 0.08, min(y_nom + 0.08, y_touchdown))
                        
                        # Interpolate swing path
                        x_lift, y_lift, z_lift = state.lift_off_positions[leg]
                        x_target = (1.0 - s_swing) * x_lift + s_swing * x_touchdown
                        y_target = (1.0 - s_swing) * y_lift + s_swing * y_touchdown
                        z_target = z_nom + swing_height_m * math.sin(math.pi * s_swing)

                    # Adapt foot touchdown height based on local terrain height query
                    foot_body_x = xh + x_target
                    foot_body_y = yh + y_target
                    
                    dx_foot = foot_body_x * cos_y - foot_body_y * sin_y
                    dy_foot = foot_body_x * sin_y + foot_body_y * cos_y
                    
                    foot_world_x = rx + dx_foot
                    foot_world_y = ry + dy_foot
                    
                    if stairs_detected:
                        foot_terrain_h = _query_terrain_height(foot_world_x, foot_world_y, rz)
                        dh_terrain = foot_terrain_h - terrain_height
                    else:
                        dh_terrain = 0.0
                    z_target += dh_terrain

                    # Solve IK
                    q_hip, q_thigh, q_calf = solve_leg_ik(x_target, y_target, z_target, leg)

                    idx_hip = state.dof_map[(leg, "hip")]
                    idx_thigh = state.dof_map[(leg, "thigh")]
                    idx_calf = state.dof_map[(leg, "calf")]

                    positions[idx_hip] = q_hip
                    positions[idx_thigh] = q_thigh
                    positions[idx_calf] = q_calf

                    # Store for next frame
                    state.last_foot_positions[leg] = (x_target, y_target, z_target)
                    state.last_phases[leg] = phi

                _command_joint_positions(go2, positions)
                # Removed root body velocity attributes overrides to rely purely on joint drive reactions and contact physics
                _record_stair_demo_telemetry(
                    state,
                    logger,
                    _build_stair_demo_telemetry(
                        rx,
                        ry,
                        rz,
                        roll,
                        pitch,
                        yaw,
                        actual_height,
                        vx,
                        vy,
                        wz,
                        state,
                        body_height_target_m=desired_world_z,
                        vertical_assist_mps=vertical_assist_mps,
                    ),
                )
                
                # Log successful execution once
                if not state.gait_logged and logger is not None:
                    state.gait_logged = True
                    log_event(
                        logger,
                        logging.INFO,
                        "go2_physics_gait_active",
                        "Go2 physics-driven trot gait controller is active",
                    )
                return

        except Exception as exc:
            _warn_rate_limited(
                logger,
                state,
                "go2_physics_gait_failed",
                "Go2 physics-driven gait failed; falling back to legacy kinematic control",
                error=str(exc)
            )

    # Legacy kinematic/procedural control fallback
    try:
        rb_prim, rb_api = _find_rigid_body_api(go2, base_link_name)
        if rb_prim is None or rb_api is None:
            raise RuntimeError("no rigid body API found on Go2 root or base links")

        xform = UsdGeom.Xformable(rb_prim)
        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        roll, pitch, yaw = _extract_roll_pitch_yaw(matrix)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        rx = float(matrix[3][0])
        ry = float(matrix[3][1])
        rz = float(matrix[3][2])
        wx = cos_y * vx - sin_y * vy
        wy = sin_y * vx + cos_y * vy
        terrain_height = _query_terrain_height(rx, ry, rz)
        actual_height = rz - terrain_height
        
        command_speed = math.sqrt((vx * vx) + (vy * vy)) + (0.25 * abs(wz))
        gait_preapplied = False
        if command_speed > 0.03 and not state.gait_logged:
            _apply_procedural_gait(
                go2,
                state,
                vx=vx,
                vy=vy,
                wz=wz,
                dt=dt,
                logger=logger,
            )
            gait_preapplied = True
            if not state.gait_logged:
                wx = 0.0
                wy = 0.0
                wz = 0.0
                vx = 0.0
                vy = 0.0
                command_speed = 0.0
                _warn_rate_limited(
                    logger,
                    state,
                    "go2_motion_blocked_until_gait_ready",
                    "Go2 body motion is blocked until joint gait animation is active",
                    interval_sec=5.0,
                )

        walk_bob_m = 0.0
        if command_speed > 0.03:
            omega = 2.0 * math.pi / max(0.2, state.gait_period)
            walk_bob_m = 0.012 * math.sin(2.0 * omega * state.gait_time)
        desired_height_m = state.target_height_m + walk_bob_m
        desired_world_z = _stair_assisted_world_z(
            rx,
            ry,
            yaw,
            vx,
            state,
            desired_height_m,
            stairs_detected=stairs_detected,
        )
        vertical_assist_mps = _clamp(
            (desired_world_z - rz) * state.height_kp,
            -state.max_vertical_speed_mps,
            state.max_vertical_speed_mps,
        )
        _record_stair_demo_telemetry(
            state,
            logger,
            _build_stair_demo_telemetry(
                rx,
                ry,
                rz,
                roll,
                pitch,
                yaw,
                actual_height,
                vx,
                vy,
                wz,
                state,
                body_height_target_m=desired_world_z,
                vertical_assist_mps=vertical_assist_mps,
            ),
        )

        if _prim_is_kinematic(rb_prim):
            _set_stable_kinematic_pose(
                go2,
                vx=vx,
                vy=vy,
                wz=wz,
                dt=dt,
                target_height_m=desired_height_m,
                stairs_detected=stairs_detected,
            )
            if not gait_preapplied:
                _apply_procedural_gait(
                    go2,
                    state,
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    dt=dt,
                    logger=logger,
                )
            return

        vz = vertical_assist_mps
        roll_rate = _clamp(-roll * state.attitude_kp, -state.max_attitude_rate_rps, state.max_attitude_rate_rps)
        pitch_rate = _clamp(-pitch * state.attitude_kp, -state.max_attitude_rate_rps, state.max_attitude_rate_rps)

        rb_api.GetVelocityAttr().Set(Gf.Vec3f(float(wx), float(wy), float(vz)))
        rb_api.GetAngularVelocityAttr().Set(
            Gf.Vec3f(
                float(math.degrees(roll_rate)),
                float(math.degrees(pitch_rate)),
                float(math.degrees(wz)),
            )
        )

        if logger is not None and not state.rigid_body_logged:
            state.rigid_body_logged = True
            state.rigid_body_path = str(rb_prim.GetPath())
            log_event(
                logger,
                logging.INFO,
                "go2_rigid_body_velocity_active",
                "Go2 rigid-body velocity control is active",
                rigid_body_path=state.rigid_body_path,
                target_height_m=state.target_height_m,
            )
    except Exception as exc:
        _warn_rate_limited(
            logger,
            state,
            "go2_velocity_physics_failed",
            "Go2 rigid-body velocity path failed; using stable kinematic stand fallback",
            error=str(exc),
        )
        _set_stable_kinematic_pose(
            go2,
            vx=vx if state.gait_logged else 0.0,
            vy=vy if state.gait_logged else 0.0,
            wz=wz if state.gait_logged else 0.0,
            dt=dt,
            target_height_m=state.target_height_m,
            stairs_detected=stairs_detected,
        )

    if not locals().get("gait_preapplied", False):
        _apply_procedural_gait(
            go2,
            state,
            vx=vx,
            vy=vy,
            wz=wz,
            dt=dt,
            logger=logger,
        )
