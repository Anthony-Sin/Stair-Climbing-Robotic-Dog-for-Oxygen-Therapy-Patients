"""Passive Go2 body-telemetry recorder: rebuilds the stair-demo overlay each step.

Split out of ``sim_go2_locomotion`` (the facade re-exports these). Reads the
robot's measured body pose, queries terrain height, and records the synthetic
stair-demo HUD telemetry. ``record_go2_telemetry`` is the public per-step entry
point for the RL locomotion loop (the policy moves the joints; this only observes).
"""
import logging
import math
from typing import Any, Dict, Optional

from pxr import Usd, UsdGeom

from sim_logging_utils import log_event

from world.sim_go2_state import Go2LocomotionState, _warn_rate_limited
from world.sim_go2_body_pose import _extract_roll_pitch_yaw, _find_rigid_body_api
from world.sim_go2_stairs import _build_stair_demo_telemetry


def _record_stair_demo_telemetry(
    state: Go2LocomotionState,
    logger: Optional[logging.Logger],
    telemetry: Dict[str, Any],
) -> None:
    state.stair_demo_telemetry = telemetry
    locomotion = telemetry.get("locomotion", {})
    phase = str(telemetry.get("phase", "unknown"))
    if logger is None:
        return

    if locomotion.get("active") and not state.stair_demo_climb_logged:
        state.stair_demo_climb_logged = True
        log_event(
            logger,
            logging.INFO,
            "synthetic_locomotion_stair_assist_active",
            "Synthetic locomotion stair telemetry is active; body-height and anti-tip assist are disabled",
            mode=locomotion.get("mode"),
            body_height_target_m=locomotion.get("body_height_target_m"),
            vertical_assist_mps=locomotion.get("vertical_assist_mps"),
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
