"""Builders for the run-dir events perf_tracker.extract_metrics consumes.

Pure dict builders -- no I/O -- so the exact field names/shapes the parser expects
are pinned in one place and unit-tested. The two streams that matter:
  * status.jsonl  : setup/start (timestamp, locomotion_mode) + docker/start (command
                    string the arg-parser reads) + summary/complete (-> outcome).
  * isaac_env.jsonl: the fall_diag physics stream -- {event:{action:"fall_diag"}, sim:
                    {x, h, pitch, roll, action_norm, policy_cmd}}. pitch/roll DEGREES.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def fall_diag_event(
    *,
    x: Optional[float],
    h: Optional[float],
    pitch_deg: float,
    roll_deg: float,
    policy_cmd: Sequence[float],
    action_norm: Optional[float],
) -> Dict[str, Any]:
    """One physics-stream sample. x/h may be None on real hardware (no global pose);
    the run still classifies as REAL on fall_diag step count + pitch/roll/action_norm."""
    sim: Dict[str, Any] = {
        "x": None if x is None else float(x),
        "h": None if h is None else float(h),
        "pitch": float(pitch_deg),
        "roll": float(roll_deg),
        "policy_cmd": [float(v) for v in policy_cmd],
    }
    if action_norm is not None:
        sim["action_norm"] = float(action_norm)
    return {"event": {"action": "fall_diag"}, "sim": sim}


def status_setup_event(timestamp: str, *, locomotion_mode: str = "pgtt") -> Dict[str, Any]:
    return {
        "stage": "setup", "state": "start", "timestamp": timestamp,
        "locomotion_mode": locomotion_mode, "parkour_heading_mode": "vision",
        "sim2real_validation_cam": False,
    }


def status_docker_event(command: str) -> Dict[str, Any]:
    """The launch command string; extract_metrics parses follow-backend/gains/etc. from it."""
    return {"stage": "docker", "state": "start", "command": command}


def status_summary_event() -> Dict[str, Any]:
    return {"stage": "summary", "state": "complete"}


def stair_demo_report(
    *, exit_reason: str, motion_elapsed_sec: float, final_x_m: Optional[float]
) -> Dict[str, Any]:
    return {
        "exit_reason": exit_reason,
        "motion_elapsed_sim_sec": float(motion_elapsed_sec),
        "stair_demo": {"robot": {"x_m": None if final_x_m is None else float(final_x_m)}},
    }
