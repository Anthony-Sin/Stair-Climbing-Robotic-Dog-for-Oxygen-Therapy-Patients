"""Trajectory analysis and pass/fail verdict computation.

Moved verbatim from analyze_run.py as part of a pure structural split.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from analyze_run_constants import STAIR_START_X_M


# ---------------------------------------------------------------------------
# Trajectory analysis — authoritative from fall_diag
# ---------------------------------------------------------------------------
def analyze_trajectory(trajectory: List[Dict], config: Dict) -> Dict:
    """Derive pass/fail metrics from the physics trajectory."""
    if not trajectory:
        return {
            "max_height_m": None,
            "max_x_m": None,
            "reached_stair_base": False,
            "stair_climb_started": False,
            "stair_top_reached": False,
            "fall_detected": False,
            "fall_time_sec": None,
            "fall_position": None,
            "steps_climbed_estimate": 0,
            "stair_detect_count": 0,
            "stair_action_active_count": 0,
        }

    step_h = config.get("step_height_m") or 0.08
    step_d = config.get("step_depth_m") or 0.30
    step_count = config.get("step_count") or 12
    start_x = float(STAIR_START_X_M)
    end_x = start_x + step_count * step_d

    max_h = max(p["h"] for p in trajectory)
    max_x = max(p["x"] for p in trajectory)

    reached_stair_base = max_x >= start_x
    # Climbing started: robot is above the first riser AND inside the stair X zone
    stair_climb_started = any(
        p["x"] >= start_x and p["h"] >= step_h * 0.8
        for p in trajectory
    )
    stair_top_reached = any(p["x"] >= end_x for p in trajectory)

    # Fall: look for the first frame where fell=True
    fall_detected = any(p["fell"] for p in trajectory)
    fall_time_sec = None
    fall_position = None
    for p in trajectory:
        if p["fell"]:
            fall_time_sec = p["t"]
            fall_position = {"x": p["x"], "y": p["y"], "h": p["h"], "roll_deg": p["roll"], "pitch_deg": p["pitch"]}
            break

    # Estimate steps climbed from max height
    steps_climbed = 0
    if step_h > 0:
        steps_climbed = min(step_count, int(max_h / step_h))

    stair_detect_count = sum(1 for p in trajectory if p.get("stairs_detected"))
    stair_action_active_count = sum(1 for p in trajectory if p.get("stairs_action_active"))

    return {
        "max_height_m": round(max_h, 3),
        "max_x_m": round(max_x, 3),
        "reached_stair_base": reached_stair_base,
        "stair_climb_started": stair_climb_started,
        "stair_top_reached": stair_top_reached,
        "fall_detected": fall_detected,
        "fall_time_sec": round(fall_time_sec, 3) if fall_time_sec is not None else None,
        "fall_position": fall_position,
        "steps_climbed_estimate": steps_climbed,
        "stair_detect_count": stair_detect_count,
        "stair_action_active_count": stair_action_active_count,
        "stair_end_x_m": round(end_x, 3),
    }


# ---------------------------------------------------------------------------
# Pass/fail checklist
# ---------------------------------------------------------------------------
def build_pass_fail(config: Dict, traj_analysis: Dict, outcome: Dict, mask_info: Dict) -> Dict:
    """Boolean checklist of key success criteria for AI review."""
    exit_reason = outcome.get("exit_reason", "")

    robot_no_fall = not traj_analysis.get("fall_detected", True) and exit_reason != "robot_fell"
    reached_stairs = traj_analysis.get("reached_stair_base", False)
    started_climb = traj_analysis.get("stair_climb_started", False)
    reached_top = traj_analysis.get("stair_top_reached", False)

    # O2 payload — only meaningful if attached
    o2_attached = config.get("o2_attached", False)
    o2_retained: Optional[bool] = None
    if o2_attached:
        # Presence of 'strap_break' or similar would indicate loss;
        # absence of such events + no fall = retained (conservative).
        o2_retained = robot_no_fall  # best we can infer without explicit strap event

    # Person masking — should fire if not disabled
    mask_expected = config.get("person_mask_enabled", True)
    mask_fired = mask_info.get("person_masking_active", False)

    return {
        "robot_did_not_fall": robot_no_fall,
        "robot_reached_stair_base": reached_stairs,
        "robot_started_climbing": started_climb,
        "robot_reached_stair_top": reached_top,
        "o2_payload_retained": o2_retained,
        "person_depth_mask_fired": mask_fired if mask_expected else None,
        # unknown = run was interrupted or still in progress, not a clean exit
        "run_completed_without_timeout": (
            None if exit_reason == "unknown"
            else exit_reason != "timeout"
        ),
        "stair_climb_action_engaged": traj_analysis.get("stair_action_active_count", 0) > 0,
    }
