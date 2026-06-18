"""
analyze_run.py — AI-ready run analyzer for Isaac Sim stair-climbing runs.

Works for BOTH the default scene (stairs-only corridor) and the --final-scene
(hospital environment with multi-leg patient route).

Usage:
    python analyze_run.py                     # auto-detects from log/latest_run.txt
    python analyze_run.py <run_dir>           # path to a specific run_sim_* folder
    python analyze_run.py --json-only         # skip contact-sheet image generation

Outputs (all written to <run_dir>/reports/):
    ai_analysis.json              — machine-readable summary for LLM review
    ai_report.md                  — human+AI readable markdown report
    contact_sheet_all_cameras.jpg — frame grid: all 4 videos + verification PNGs

Data sources and their reliability:
    fall_diag events (debug/isaac_env.jsonl)  → REAL physics trajectory  [AUTHORITATIVE]
    evaluation_exit event                     → REAL exit reason + final state  [AUTHORITATIVE]
    stair_demo_report.json                    → SYNTHETIC last-frame snapshot  [LABEL/HUD ONLY]
    evaluation_summary.txt                    → human-readable summary of above
    videos/                                   → visual record of the run

NOTE (from CLAUDE.md): stair_demo_report.json and evaluation_summary.txt derive
phase/locomotion labels from Isaac ground-truth geometry, NOT from sensor data.
Judge real motion and falls from the fall_diag trajectory in isaac_env.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None   # type: ignore[assignment]
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAIR_START_X_M = 2.0   # fixed across all presets (see isaac_env.py)
VIDEO_NAMES = ("scene_view.mp4", "topdown.mp4", "lidar_preview.mp4", "opencv_preview.mp4")
VERIFICATION_PNGS = ("verification_start.png", "verification_end.png")
THUMB_W, THUMB_H = 384, 216
LABEL_HEIGHT = 22
FRAMES_PER_VIDEO = 4   # start, 1/3, 2/3, end


# ---------------------------------------------------------------------------
# Resolve run directory
# ---------------------------------------------------------------------------
def resolve_run_dir(arg: Optional[str] = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    log_root = repo_root / "log"

    if arg:
        p = Path(arg)
        # Try as-is first, then relative to repo root, then relative to log dir
        candidates = [p]
        if not p.is_absolute():
            candidates += [repo_root / p, log_root / p]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"Run dir not found: {arg!r} (tried {[str(c) for c in candidates]})")

    # Auto-detect from latest_run.txt
    latest_file = log_root / "latest_run.txt"
    if latest_file.exists():
        run_name = latest_file.read_text().strip()
        if run_name:
            candidate = log_root / run_name
            if candidate.is_dir():
                return candidate

    # Fall back: most recent run_sim_* directory
    candidates = sorted(log_root.glob("run_sim_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No run directory found. Pass one explicitly or ensure {log_root} has run_sim_* folders."
    )


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    events: List[Dict] = []
    if not path.exists():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _action(event: Dict) -> str:
    return event.get("event", {}).get("action", "")


def _sim(event: Dict) -> Dict:
    return event.get("sim", {})


def _ts(event: Dict) -> str:
    return event.get("@timestamp", "")


def _level(event: Dict) -> str:
    return event.get("level", "")


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
# Parse text/JSON reports
# ---------------------------------------------------------------------------
def parse_evaluation_summary(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def parse_stair_demo_report(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
# Contact-sheet image generation
# ---------------------------------------------------------------------------
def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Add a dark banner with label text at top of frame."""
    h, w = frame.shape[:2]
    out = np.zeros((h + LABEL_HEIGHT, w, 3), dtype=np.uint8)
    out[:LABEL_HEIGHT, :] = (30, 30, 30)
    cv2.putText(
        out, label, (6, LABEL_HEIGHT - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    out[LABEL_HEIGHT:, :] = frame
    return out


def _thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def extract_video_frames(path: Path, n: int = FRAMES_PER_VIDEO) -> List[Tuple[np.ndarray, str]]:
    """Return (bgr_frame, label) tuples sampled evenly from a video."""
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 1:
        cap.release()
        return []

    indices = []
    if n == 1:
        indices = [0]
    else:
        for i in range(n):
            idx = int(round(i * (total - 1) / (n - 1)))
            indices.append(max(0, min(total - 1, idx)))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            pct = int(100 * idx / max(1, total - 1))
            frames.append((_thumb(frame), f"{path.name} [{pct}%]"))
    cap.release()
    return frames


def load_png_as_bgr(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img


def build_contact_sheet(run_dir: Path) -> Optional[np.ndarray]:
    """Build a contact sheet from all cameras and verification PNGs."""
    if not _CV2_AVAILABLE:
        return None
    videos_dir = run_dir / "videos"
    reports_dir = run_dir / "reports"

    all_cells: List[np.ndarray] = []

    # -- Verification PNGs first (wide scene overview) --
    for png_name in VERIFICATION_PNGS:
        img = load_png_as_bgr(reports_dir / png_name)
        if img is not None:
            cell = _label_frame(_thumb(img), png_name)
            all_cells.append(cell)

    # -- Video frames for each camera --
    for vid_name in VIDEO_NAMES:
        vid_path = videos_dir / vid_name
        frames = extract_video_frames(vid_path, n=FRAMES_PER_VIDEO)
        for frame_bgr, label in frames:
            all_cells.append(_label_frame(frame_bgr, label))

    if not all_cells:
        return None

    # Normalise all cells to same height
    cell_h = THUMB_H + LABEL_HEIGHT
    cell_w = THUMB_W
    for i, cell in enumerate(all_cells):
        if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
            all_cells[i] = cv2.resize(cell, (cell_w, cell_h))

    # Layout: 4 cells per row (1 row per video + verification row)
    COLS = 4
    while len(all_cells) % COLS:
        all_cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(all_cells), COLS):
        rows.append(np.concatenate(all_cells[i:i + COLS], axis=1))
    sheet = np.concatenate(rows, axis=0)
    return sheet


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


# ---------------------------------------------------------------------------
# Build the full AI analysis JSON
# ---------------------------------------------------------------------------
def build_ai_analysis(run_dir: Path) -> Dict[str, Any]:
    debug_dir = run_dir / "debug"
    reports_dir = run_dir / "reports"

    events = load_jsonl(debug_dir / "isaac_env.jsonl")
    controller_events = load_jsonl(debug_dir / "sim_robot_controller.jsonl")

    config = extract_scene_config(events)
    trajectory = extract_trajectory(events)
    key_events = extract_key_events(events)
    warnings = extract_warnings(events)
    mask_info = extract_mask_events(events)
    outcome = extract_outcome_from_jsonl(events)
    traj_analysis = analyze_trajectory(trajectory, config)

    stair_demo_report = parse_stair_demo_report(reports_dir / "stair_demo_report.json")
    evaluation_summary_txt = parse_evaluation_summary(reports_dir / "evaluation_summary.txt")

    final_scene_info = extract_final_scene_info(events)

    pass_fail = build_pass_fail(config, traj_analysis, outcome, mask_info)

    # Videos present
    videos_dir = run_dir / "videos"
    video_files = {
        name: (videos_dir / name).exists()
        for name in VIDEO_NAMES
    }
    png_files = {
        name: (reports_dir / name).exists()
        for name in VERIFICATION_PNGS
    }

    # Controller events summary
    controller_summary: Dict[str, Any] = {}
    if controller_events:
        controller_summary["event_count"] = len(controller_events)
        controller_summary["event_types"] = list({
            _action(ev) for ev in controller_events
        })

    return {
        "run_id": run_dir.name,
        "data_authority_note": (
            "fall_diag events in isaac_env.jsonl = authoritative physics trajectory. "
            "stair_demo_report.json/evaluation_summary.txt = HUD labels from Isaac ground-truth "
            "geometry — NOT sensor data, NOT hardware-confirmed. Use JSONL for fall/climb judgement."
        ),
        "scene_config": config,
        "outcome": outcome,
        "trajectory_analysis": traj_analysis,
        "pass_fail": pass_fail,
        "person_masking": mask_info,
        "key_events": key_events,
        "final_scene": final_scene_info,
        "warnings": warnings[:50],   # cap at 50
        "warning_count": len(warnings),
        "synthetic_report": {
            "note": "Values below are HUD/geometry labels. Do not use for climb judgement.",
            "evaluation_summary_text": evaluation_summary_txt,
            "stair_demo_report": stair_demo_report,
        },
        "video_files": video_files,
        "verification_pngs": png_files,
        "controller_events": controller_summary,
        "trajectory_sample": trajectory,   # full sampled trajectory
    }


# ---------------------------------------------------------------------------
# Build markdown report
# ---------------------------------------------------------------------------
def build_markdown_report(analysis: Dict) -> str:
    cfg = analysis["scene_config"]
    outcome = analysis["outcome"]
    ta = analysis["trajectory_analysis"]
    pf = analysis["pass_fail"]
    mask = analysis["person_masking"]
    fs = analysis["final_scene"]
    warnings = analysis["warnings"]

    def tick(val: Optional[bool]) -> str:
        if val is None:
            return "⬜ N/A"
        return "✅ PASS" if val else "❌ FAIL"

    lines: List[str] = []

    lines.append(f"# Run Analysis: {analysis['run_id']}")
    lines.append("")
    lines.append(f"> **Data authority**: `fall_diag` JSONL = authoritative physics.  "
                 f"`stair_demo_report.json` = synthetic HUD labels only.")
    lines.append("")

    # ---- Config ----
    lines.append("## Scene Configuration")
    lines.append(f"- **Scene type**: `{cfg['scene_type']}`")
    lines.append(f"- **Stair preset**: `{cfg['stair_preset']}` — "
                 f"{cfg.get('step_count')} steps × {cfg.get('step_height_m')} m rise / "
                 f"{cfg.get('step_depth_m')} m run → top height {cfg.get('top_height_m')} m")
    lines.append(f"- **Stair base X**: {STAIR_START_X_M} m, "
                 f"**top X**: {ta.get('stair_end_x_m')} m")
    lines.append(f"- **Handrail**: {cfg.get('handrail')}")
    lines.append(f"- **Realism profile**: `{cfg['realism_profile']}`")
    if cfg["sim2real_validation_cam"]:
        lines.append(f"  - obs_noise={cfg['obs_noise']}, "
                     f"domain_rand={cfg['domain_rand']}, "
                     f"depth_noise_mult={cfg['parkour_depth_noise_mult']}, "
                     f"obs_latency_steps={cfg['obs_latency_steps']}")
    lines.append(f"- **O2 payload attached**: {cfg['o2_attached']}")
    lines.append(f"- **Parkour heading mode**: `{cfg.get('parkour_heading_mode', 'vision')}`")
    lines.append(f"- **Person depth mask**: enabled={cfg.get('person_mask_enabled', True)}, "
                 f"fill=`{cfg.get('person_mask_fill', 'terrain')}`")
    lines.append("")

    # ---- Outcome ----
    lines.append("## Run Outcome")
    lines.append(f"- **Exit reason**: `{outcome.get('exit_reason', 'unknown')}`")
    lines.append(f"- **Motion elapsed (sim sec)**: {outcome.get('motion_elapsed_sec')} s")
    lines.append(f"- **Stair phase at exit**: `{outcome.get('stair_phase_at_exit')}`")
    if outcome.get("final_height_m") is not None:
        lines.append(f"- **Final robot height**: {outcome['final_height_m']:.3f} m")
    if outcome.get("recoveries_used", 0) > 0:
        lines.append(f"- **Fall recoveries used**: {outcome['recoveries_used']}")
    lines.append("")

    # ---- Trajectory metrics ----
    lines.append("## Trajectory Metrics (Physics, Authoritative)")
    lines.append(f"- **Max height reached**: {ta.get('max_height_m')} m")
    lines.append(f"- **Max X reached**: {ta.get('max_x_m')} m "
                 f"(stair base={STAIR_START_X_M} m, top={ta.get('stair_end_x_m')} m)")
    lines.append(f"- **Steps climbed (estimate)**: {ta.get('steps_climbed_estimate')} "
                 f"of {cfg.get('step_count')}")
    if ta.get("fall_detected"):
        fp = ta.get("fall_position", {})
        lines.append(f"- **Fall detected at**: t={ta.get('fall_time_sec')} s, "
                     f"x={fp.get('x')} m, h={fp.get('h')} m, "
                     f"roll={fp.get('roll_deg')}°, pitch={fp.get('pitch_deg')}°")
    lines.append(f"- **Stair detect events in trajectory**: {ta.get('stair_detect_count')}")
    lines.append(f"- **Stair action active events**: {ta.get('stair_action_active_count')}")
    lines.append(f"- **Person mask events fired**: {mask.get('mask_event_count', 0)}")
    lines.append("")

    # ---- Pass/fail ----
    lines.append("## Pass/Fail Checklist")
    lines.append(f"- {tick(pf.get('robot_did_not_fall'))} Robot did not fall")
    lines.append(f"- {tick(pf.get('robot_reached_stair_base'))} Robot reached stair base (x ≥ {STAIR_START_X_M} m)")
    lines.append(f"- {tick(pf.get('robot_started_climbing'))} Robot started climbing")
    lines.append(f"- {tick(pf.get('robot_reached_stair_top'))} Robot reached stair top")
    lines.append(f"- {tick(pf.get('stair_climb_action_engaged'))} Stair-climb action engaged (policy sent stair cmd)")
    lines.append(f"- {tick(pf.get('person_depth_mask_fired'))} Person depth mask fired")
    lines.append(f"- {tick(pf.get('o2_payload_retained'))} O2 payload retained" +
                 ("" if cfg.get("o2_attached") else " (N/A — not attached)"))
    lines.append(f"- {tick(pf.get('run_completed_without_timeout'))} Run completed without timeout")
    lines.append("")

    # ---- Final scene extras ----
    if fs:
        lines.append("## Final Scene (Hospital)")
        lines.append(f"- Hospital env loaded: {fs.get('hospital_env_loaded')}")
        lines.append(f"- Patient started stair phase: {fs.get('patient_stair_phase_started')}")
        lines.append(f"- Patient trajectory events logged: {fs.get('patient_trajectory_event_count')}")
        lines.append("")

    # ---- Key event timeline ----
    lines.append("## Key Event Timeline")
    for ev in analysis.get("key_events", []):
        data_str = ""
        d = ev.get("data", {})
        if d:
            brief = {k: v for k, v in d.items() if k not in ("log_path",)}
            data_str = " — " + json.dumps(brief, separators=(",", ":"))[:120]
        ts = ev.get("timestamp", "")[:19]
        lines.append(f"- `{ts}` **{ev['action']}**{data_str}")
    lines.append("")

    # ---- Warnings ----
    if warnings:
        lines.append(f"## Warnings ({len(warnings)} total, showing first {min(len(warnings), 20)})")
        for w in warnings[:20]:
            lines.append(f"- {w}")
        lines.append("")

    # ---- Video files ----
    lines.append("## Available Artifacts")
    for name, present in analysis.get("video_files", {}).items():
        lines.append(f"- {'✅' if present else '❌'} `videos/{name}`")
    for name, present in analysis.get("verification_pngs", {}).items():
        lines.append(f"- {'✅' if present else '❌'} `reports/{name}`")
    lines.append(f"- ✅ `reports/ai_analysis.json`")
    lines.append(f"- ✅ `reports/ai_report.md`")
    contact_mark = "✅" if _CV2_AVAILABLE else "⬜ (cv2 not installed)"
    lines.append(f"- {contact_mark} `reports/contact_sheet_all_cameras.jpg`")
    lines.append("")

    # ---- Synthetic report note ----
    lines.append("## Synthetic Report (HUD Labels Only — Do Not Use for Climb Judgement)")
    lines.append("```")
    lines.append(analysis["synthetic_report"].get("evaluation_summary_text", "(missing)"))
    lines.append("```")
    lines.append("")
    lines.append(
        "> NOTE: The text above reflects `stair_demo_phase`, `locomotion_mode`, and "
        "`robot_summary` from Isaac ground-truth geometry labels, NOT from sensor data "
        "or the RL policy output. See `trajectory_analysis` above for physics-authoritative results."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an Isaac Sim run for AI review.")
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Path to a run_sim_* folder (default: auto-detect from latest_run.txt).",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Skip contact-sheet image generation (no cv2 video decoding).",
    )
    args = parser.parse_args()

    try:
        run_dir = resolve_run_dir(args.run_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing run: {run_dir}")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build analysis ----
    analysis = build_ai_analysis(run_dir)

    # ---- Write ai_analysis.json ----
    json_path = reports_dir / "ai_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"  Written: {json_path}")

    # ---- Write ai_report.md ----
    md_path = reports_dir / "ai_report.md"
    md_path.write_text(build_markdown_report(analysis), encoding="utf-8")
    print(f"  Written: {md_path}")

    # ---- Build contact sheet ----
    if not args.json_only:
        if not _CV2_AVAILABLE:
            print("  WARNING: cv2 not available — skipping contact sheet. "
                  "Run: pip install opencv-python", file=sys.stderr)
        else:
            sheet = build_contact_sheet(run_dir)
            if sheet is not None:
                sheet_path = reports_dir / "contact_sheet_all_cameras.jpg"
                ok = cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    print(f"  Written: {sheet_path} ({sheet.shape[1]}x{sheet.shape[0]})")
                else:
                    print(f"  WARNING: cv2.imwrite failed for contact sheet", file=sys.stderr)
            else:
                print("  No frames extracted for contact sheet (no videos found).", file=sys.stderr)

    # ---- Print pass/fail summary to console ----
    pf = analysis["pass_fail"]
    ta = analysis["trajectory_analysis"]
    outcome = analysis["outcome"]
    print()
    print(f"=== PASS/FAIL SUMMARY ===")
    print(f"  Exit reason       : {outcome.get('exit_reason', 'unknown')}")
    print(f"  Motion elapsed    : {outcome.get('motion_elapsed_sec')} s")
    print(f"  Max height reached: {ta.get('max_height_m')} m")
    print(f"  Max X reached     : {ta.get('max_x_m')} m")
    print(f"  Steps climbed est : {ta.get('steps_climbed_estimate')}")
    print(f"  Fell              : {ta.get('fall_detected')}")
    print(f"  Reached stairs    : {pf.get('robot_reached_stair_base')}")
    print(f"  Started climbing  : {pf.get('robot_started_climbing')}")
    print(f"  Reached top       : {pf.get('robot_reached_stair_top')}")
    print(f"  Mask fired        : {pf.get('person_depth_mask_fired')}")
    if analysis["scene_config"].get("o2_attached"):
        print(f"  O2 payload retained: {pf.get('o2_payload_retained')}")
    print()


if __name__ == "__main__":
    main()
