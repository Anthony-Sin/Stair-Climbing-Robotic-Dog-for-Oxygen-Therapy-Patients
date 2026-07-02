"""Structured extraction of scene config, trajectory, events, and outcome.

Moved verbatim from analyze_run.py as part of a pure structural split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from analyze_run_constants import STAIR_START_X_M
from analyze_run_io import _action, _level, _sim, _ts


# ---------------------------------------------------------------------------
# Extract structured data from JSONL
# ---------------------------------------------------------------------------
def extract_scene_config(events: List[Dict]) -> Dict:
    """Pull run configuration from bootstrap and stair-preset events."""
    config: Dict[str, Any] = {
        "scene_type": "default",
        "stair_preset": None,
        "step_count": None,
        "step_height_m": None,
        "step_depth_m": None,
        "top_height_m": None,
        "start_x_m": STAIR_START_X_M,
        "handrail": None,
        "realism_profile": "perfect_env",
        "sim2real_validation_cam": False,
        "obs_noise": False,
        "domain_rand": False,
        "parkour_depth_noise_mult": 0.0,
        "obs_latency_steps": 0,
        "o2_attached": False,
        "parkour_heading_mode": "vision",
        "person_mask_enabled": True,
        "person_mask_fill": "terrain",
        "fall_recovery": False,
        "physics_hz": 200,
        "self_test_walk": False,
    }

    for ev in events:
        a = _action(ev)
        s = _sim(ev)

        if a == "isaac_env_bootstrap":
            config["scene_type"] = "final_scene" if s.get("final_scene") else "default"
            config["physics_hz"] = s.get("physics_hz", 200)
            config["self_test_walk"] = bool(s.get("self_test_walk", False))

        elif a == "stair_preset_configured":
            config["stair_preset"] = s.get("preset")
            config["step_count"] = s.get("step_count")
            config["step_height_m"] = s.get("step_height_m")
            config["step_depth_m"] = s.get("step_depth_m")
            config["top_height_m"] = s.get("top_height_m")
            config["handrail"] = s.get("handrail")

        elif a == "realism_profile":
            config["realism_profile"] = (
                "real_sim_env" if s.get("sim2real_validation_cam") else "perfect_env"
            )
            config["sim2real_validation_cam"] = bool(s.get("sim2real_validation_cam", False))
            config["obs_noise"] = bool(s.get("obs_noise", False))
            config["domain_rand"] = bool(s.get("domain_rand", False))
            config["parkour_depth_noise_mult"] = float(s.get("parkour_depth_noise_mult", 0.0))
            config["obs_latency_steps"] = int(s.get("obs_latency_steps", 0))

        elif a == "robot_config":
            config["o2_attached"] = bool(s.get("o2_attached", False))

    return config


def extract_trajectory(events: List[Dict], max_samples: int = 200) -> List[Dict]:
    """Extract a sampled robot trajectory from fall_diag events.

    fall_diag fires at control-loop rate (200 Hz physics / render_every steps),
    so we downsample to at most max_samples points for readability.
    """
    raw: List[Dict] = []
    for ev in events:
        if _action(ev) == "fall_diag":
            s = _sim(ev)
            raw.append({
                "t": s.get("t", 0.0),
                "x": s.get("x", 0.0),
                "y": s.get("y", 0.0),
                "h": s.get("h", 0.0),   # height above terrain
                "roll": s.get("roll", 0.0),
                "pitch": s.get("pitch", 0.0),
                "yaw": s.get("yaw", 0.0),
                "vx": s.get("vx", 0.0),
                "wz": s.get("wz", 0.0),
                "fell": bool(s.get("fell", False)),
                "stairs_detected": bool(s.get("stairs_detected", False)),
                "stairs_action_active": bool(s.get("stairs_action_active", False)),
            })

    if len(raw) <= max_samples:
        return raw

    # Uniform downsampling
    step = len(raw) / max_samples
    indices = {int(i * step) for i in range(max_samples)}
    indices.add(len(raw) - 1)
    return [raw[i] for i in sorted(indices)]


def extract_key_events(events: List[Dict]) -> List[Dict]:
    """Extract the handful of events that tell the story of a run."""
    KEY_ACTIONS = {
        "world_ready",
        "scene_motion_started",
        "scene_motion_waiting_for_controller",
        "controller_command_stream_started",
        "robot_stair_climb_visible",
        "patient_stair_phase_started",
        "parkour_person_depth_masked",
        "synthetic_locomotion_stair_assist_active",
        "evaluation_exit",
        "fall_recovery_triggered",
        "fall_recovery_complete",
    }
    out: List[Dict] = []
    for ev in events:
        a = _action(ev)
        if a in KEY_ACTIONS:
            entry: Dict[str, Any] = {
                "action": a,
                "timestamp": _ts(ev),
            }
            s = _sim(ev)
            if s:
                entry["data"] = s
            out.append(entry)
    return out


def extract_warnings(events: List[Dict]) -> List[str]:
    """Collect all WARNING and ERROR level log messages."""
    out: List[str] = []
    for ev in events:
        if _level(ev) in ("WARNING", "ERROR"):
            msg = ev.get("message", "")
            a = _action(ev)
            out.append(f"[{_level(ev)}] {a}: {msg}")
    return out


def extract_mask_events(events: List[Dict]) -> Dict:
    """Count how many times person masking fired and if it was active."""
    count = sum(1 for ev in events if _action(ev) == "parkour_person_depth_masked")
    return {"mask_event_count": count, "person_masking_active": count > 0}


def extract_outcome_from_jsonl(events: List[Dict]) -> Dict:
    """Pull the definitive run outcome from the evaluation_exit event."""
    for ev in reversed(events):
        if _action(ev) == "evaluation_exit":
            s = _sim(ev)
            return {
                "exit_reason": s.get("reason", "unknown"),
                "final_height_m": s.get("robot_height_m"),
                "final_roll_rad": s.get("roll_rad"),
                "final_pitch_rad": s.get("pitch_rad"),
                "motion_elapsed_sec": s.get("motion_elapsed_sim_sec"),
                "stair_phase_at_exit": s.get("stair_phase"),
                "recoveries_used": s.get("recoveries_used", 0),
            }
    return {"exit_reason": "unknown"}


# ---------------------------------------------------------------------------
# Final scene analysis extras
# ---------------------------------------------------------------------------
def extract_final_scene_info(events: List[Dict]) -> Optional[Dict]:
    """Additional analysis for --final-scene runs (patient route, hospital env)."""
    is_final = any(
        _action(ev) == "isaac_env_bootstrap" and _sim(ev).get("final_scene")
        for ev in events
    )
    if not is_final:
        return None

    patient_trajectory_events = [
        ev for ev in events if _action(ev) == "patient_trajectory"
    ]

    patient_stair_started = any(
        _action(ev) == "patient_stair_phase_started" for ev in events
    )

    route_waypoints_logged = len(patient_trajectory_events)

    hospital_env_loaded = any(
        _action(ev) == "world_ready" for ev in events
    )

    return {
        "hospital_env_loaded": hospital_env_loaded,
        "patient_stair_phase_started": patient_stair_started,
        "patient_trajectory_event_count": route_waypoints_logged,
        "note": (
            "Final-scene: patient walks a multi-leg hospital lobby route before the stairs. "
            "patient_stair_phase_started = patient reached and began climbing the staircase."
        ),
    }
