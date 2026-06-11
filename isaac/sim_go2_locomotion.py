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
    for prim in _candidate_rigid_body_prims(go2, base_link_name):
        if not _prim_has_rigid_body(prim):
            continue
        rb_api = UsdPhysics.RigidBodyAPI(prim)
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


def _set_stable_kinematic_pose(
    go2: Any,
    *,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    target_height_m: float,
) -> None:
    xformable = UsdGeom.Xformable(go2.prim)
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    _, _, yaw = _extract_roll_pitch_yaw(matrix)
    new_yaw = yaw + (wz * dt)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    tx = float(matrix[3][0]) + ((cos_y * vx - sin_y * vy) * dt)
    ty = float(matrix[3][1]) + ((sin_y * vx + cos_y * vy) * dt)
    tz = target_height_m

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
        state.gait_phase += dt * (7.0 + (4.0 * min(command_speed, 1.0)))
        swing = min(1.0, command_speed / 0.6)
        for index, raw_name in enumerate(state.dof_names):
            name = raw_name.lower()
            if "fl" in name or "rr" in name:
                phase = state.gait_phase
            else:
                phase = state.gait_phase + math.pi

            sin_phase = math.sin(phase)
            if "hip" in name:
                positions[index] += 0.05 * swing * sin_phase
            elif "thigh" in name:
                positions[index] += 0.28 * swing * sin_phase
            elif "calf" in name:
                positions[index] -= 0.36 * swing * max(0.0, sin_phase)

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
    if not (-1.05 <= y <= 1.05):
        return 0.0
    # Stairs: 2.0 to 3.5m
    if 2.0 <= x < 3.5:
        step_idx = int((x - 2.0) / 0.3)
        return min(0.40, (step_idx + 1) * 0.08)
    # Flat ground
    return 0.0


def _query_terrain_height(rx: float, ry: float, rz: float) -> float:
    """Query terrain height at (rx, ry) via PhysX raycast or analytical fallback."""
    try:
        import omni.physx
        physx_interface = omni.physx.get_physx_interface()
        # Raycast straight down from the robot's Z position.
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
    return _get_analytical_terrain_height(rx, ry)


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
                                controller.set_joint_drive_gains(i, stiffness=80.0, damping=2.0)
                        except Exception:
                            dof_props = go2.get_dof_properties()
                            dof_props["stiffness"] = 80.0
                            dof_props["damping"] = 2.0
                            go2.set_dof_properties(dof_props)
                        state.joint_gains_set = True
                        if logger is not None:
                            log_event(
                                logger,
                                logging.INFO,
                                "go2_joint_gains_configured",
                                "Go2 joint drive gains (stiffness=80, damping=2) have been applied",
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
                body_vx = cos_y * actual_vx + sin_y * actual_vy
                body_vy = -sin_y * actual_vx + cos_y * actual_vy

                # 4. Trot phase scheduler
                cmd_speed = math.sqrt(vx**2 + vy**2) + 0.25 * abs(wz)
                is_moving = cmd_speed > 0.03

                if is_moving:
                    state.gait_time += dt
                else:
                    state.gait_time = 0.0

                T_c = state.gait_period
                T_s = T_c * state.duty_factor

                legs = ["fl", "fr", "rl", "rr"]
                phases = {}
                for leg in legs:
                    if not is_moving:
                        phases[leg] = 0.0
                    else:
                        if leg in ["fl", "rr"]:
                            phases[leg] = (state.gait_time / T_c) % 1.0
                        else:
                            phases[leg] = (state.gait_time / T_c + 0.5) % 1.0

                # Torso sway oscillations
                if is_moving:
                    omega = 2.0 * math.pi / T_c
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
                    if phi < 0.5:
                        # Stance phase
                        s_stance = phi / 0.5
                        x_target = v_hip_x * (T_s / 2.0) * (1.0 - 2.0 * s_stance)
                        y_target = y_nom + v_hip_y * (T_s / 2.0) * (1.0 - 2.0 * s_stance)
                        z_target = z_nom_corr
                    else:
                        # Swing phase
                        s_swing = (phi - 0.5) / 0.5
                        
                        last_p = state.last_phases.get(leg, 0.0)
                        is_liftoff = (last_p < 0.5 <= phi) or (leg not in state.lift_off_positions)
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
                        z_target = z_nom + state.swing_height * math.sin(math.pi * s_swing)

                    # Adapt foot touchdown height based on local terrain height query
                    foot_body_x = xh + x_target
                    foot_body_y = yh + y_target
                    
                    dx_foot = foot_body_x * cos_y - foot_body_y * sin_y
                    dy_foot = foot_body_x * sin_y + foot_body_y * cos_y
                    
                    foot_world_x = rx + dx_foot
                    foot_world_y = ry + dy_foot
                    
                    foot_terrain_h = _query_terrain_height(foot_world_x, foot_world_y, rz)
                    dh_terrain = foot_terrain_h - terrain_height
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
        if rx >= 1.86 and vx > 0.0:
            vx = 0.0
            if logger is not None and not state.stair_hold_logged:
                state.stair_hold_logged = True
                log_event(
                    logger,
                    logging.INFO,
                    "go2_stair_hold_active",
                    "Holding Go2 before the first stair because stair-climbing gait is not enabled yet",
                    hold_x_m=float(rx),
                )
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

        vz = _clamp(
            (desired_height_m - actual_height) * state.height_kp,
            -state.max_vertical_speed_mps,
            state.max_vertical_speed_mps,
        )
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
