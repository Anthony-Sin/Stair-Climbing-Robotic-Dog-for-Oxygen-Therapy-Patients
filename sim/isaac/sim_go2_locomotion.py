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


# Locomotion is driven by the RL policy in rl_locomotion_policy.py. This module
# now only owns the stair-demo perception/telemetry and the analytical terrain
# model used to build that telemetry from the robot's measured body pose.


@dataclass
class Go2LocomotionState:
    # Telemetry/bookkeeping state shared with the stair-demo reporter. The RL
    # locomotion policy (rl_locomotion_policy.py) drives the joints; this struct
    # only carries perception/telemetry fields and per-run logging latches.
    target_height_m: float = 0.30
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

    # Posture balance PD gains. Roll/pitch gains drive per-foot height offsets that
    # right the body; the previous values (1.0 / 0.05) produced only ~2 cm of
    # correction at a 30 deg tilt and could not stop a sideways rollover at follow
    # speed, so they are stiffer here.
    kp_height: float = 1.2
    kd_height: float = 0.15
    kp_roll: float = 3.0
    kd_roll: float = 0.15
    kp_pitch: float = 3.0
    kd_pitch: float = 0.15

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
    stair_visual_deferred_logged: bool = False
    fallback_body_motion_disabled_logged: bool = False

    # Diagnostics: most-recent measured base velocity (m/s). diag_body_* is the
    # measured velocity rotated into the robot's heading frame, so diag_body_vx > 0
    # means the body is actually translating forward. Compared against the
    # commanded vx this reveals whether the gait produces forward thrust or the
    # body is recoiling/slipping backward.
    diag_body_vx: float = 0.0
    diag_body_vy: float = 0.0
    diag_cmd_vx: float = 0.0


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
            "body_height_assist_enabled": False,
            "anti_tip_assist_enabled": False,
        },
    }


def _effective_swing_height_for_mode(mode: str) -> float:
    if mode == "stair_climb":
        return 0.11
    if mode == "stair_approach":
        return 0.08
    return 0.06


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
            "Synthetic blind-RL stair telemetry is active; body-height and anti-tip assist are disabled",
            mode=blind_rl.get("mode"),
            body_height_target_m=blind_rl.get("body_height_target_m"),
            vertical_assist_mps=blind_rl.get("vertical_assist_mps"),
            physics_contact_enabled=True,
            body_height_assist_enabled=False,
            anti_tip_assist_enabled=False,
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


def _record_passive_body_telemetry(
    go2: Any,
    state: Go2LocomotionState,
    *,
    base_link_name: str,
    logger: Optional[logging.Logger],
    vx: float,
    vy: float,
    wz: float,
) -> None:
    rb_prim, _ = _find_rigid_body_api(go2, base_link_name)
    if rb_prim is None:
        return
    try:
        matrix = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        roll, pitch, yaw = _extract_roll_pitch_yaw(matrix)
        rx = float(matrix[3][0])
        ry = float(matrix[3][1])
        rz = float(matrix[3][2])
        # Diagnostic: measured base velocity rotated into the heading frame so
        # diag_body_vx > 0 means the robot is actually moving forward (compared
        # against the commanded vx in the fall_diag log).
        try:
            lin = go2.get_linear_velocity()
            cy, sy = math.cos(yaw), math.sin(yaw)
            state.diag_body_vx = float(cy * lin[0] + sy * lin[1])
            state.diag_body_vy = float(-sy * lin[0] + cy * lin[1])
            state.diag_cmd_vx = float(vx)
        except Exception:
            pass
        terrain_height = _query_terrain_height(rx, ry, rz)
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
                rz - terrain_height,
                vx,
                vy,
                wz,
                state,
                body_height_target_m=None,
                vertical_assist_mps=0.0,
            ),
        )
    except Exception as exc:
        _warn_rate_limited(
            logger,
            state,
            "go2_passive_telemetry_failed",
            "Go2 passive telemetry update failed",
            interval_sec=10.0,
            error=str(exc),
        )


# Public entry point for the RL locomotion loop: rebuild the stair-demo telemetry
# from the robot's measured body pose each control step (the policy moves the
# joints; this only observes).
record_go2_telemetry = _record_passive_body_telemetry


