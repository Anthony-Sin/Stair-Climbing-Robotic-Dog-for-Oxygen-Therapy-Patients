"""Build the AI analysis JSON and the human/AI-readable markdown report.

Moved verbatim from analyze_run.py as part of a pure structural split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyze_run_constants import (
    STAIR_START_X_M,
    VERIFICATION_PNGS,
    VIDEO_NAMES,
    _CV2_AVAILABLE,
)
from analyze_run_extract import (
    extract_final_scene_info,
    extract_key_events,
    extract_mask_events,
    extract_outcome_from_jsonl,
    extract_scene_config,
    extract_trajectory,
    extract_warnings,
)
from analyze_run_io import (
    _action,
    load_jsonl,
    parse_evaluation_summary,
    parse_stair_demo_report,
)
from analyze_run_metrics import analyze_trajectory, build_pass_fail


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
