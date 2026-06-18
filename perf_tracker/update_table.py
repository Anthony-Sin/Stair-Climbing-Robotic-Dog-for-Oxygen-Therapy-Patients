#!/usr/bin/env python3
"""
update_table.py

Extract performance metrics from a single run_sim_* folder and upsert a row
into two persistent files in perf_tracker/data/:

  perf_tracker/data/performance_table.csv   -- sorted data table (best max_x_m first)
  perf_tracker/data/performance_table.jsonl -- one JSON object per run, for programmatic use

Usage (called by run_sim.ps1 via WSL):
    python3 perf_tracker/update_table.py <run_folder_path> [--git-branch <branch>]

Data sources inside each run folder:
  logs/status.jsonl                            -- launcher timeline + docker command (all controller args)
  reports/stair_demo_report.json               -- exit_reason, motion time, final robot pose
  debug/isaac_env.jsonl                        -- fall_diag physics stream, stair preset, climb events
  debug/debug_trace/vision_main_trace.jsonl    -- per-frame controller timing, YOLO detections, latency
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"

sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Column schema -- order defines CSV column order.
# Grouped: identity | outcome | trajectory | dynamics | scene | physical | args | features | trace
# ---------------------------------------------------------------------------
COLUMNS = [
    # --- Identity ---
    "run_id",
    "timestamp",
    "git_branch",
    "git_commit_sha",       # 7-char short SHA at run time
    "git_commit_msg",       # first line of commit message
    # --- Outcome ---
    "outcome",              # robot_fell | completed | timeout | docker_failed | unknown
    "exit_reason",          # raw exit_reason from stair_demo_report.json
    "sim_gate_state",       # complete | failed (did patient reach top?)
    "motion_elapsed_sec",
    "stair_phase_sec",
    # --- Physics trajectory (from fall_diag stream -- authoritative) ---
    "stair_climb_reached",          # bool: robot entered staircase phase
    "patient_stair_phase_reached",  # bool: patient reached base of stairs
    "final_x_m",                    # last recorded x position
    "max_x_m",                      # furthest forward the robot got
    "final_y_m",                    # lateral drift at end
    "final_height_m",               # body height at last sample
    "final_pitch_deg",
    "final_roll_deg",
    "final_yaw_deg",                # heading at end (deg)
    "fall_type",                    # upright | side | nose_down
    # --- Dynamics stats ---
    "fall_diag_steps",              # total physics samples logged
    "mean_vx_cmd_mps",              # mean commanded forward speed
    "mean_yaw_cmd_rps",             # mean commanded yaw rate
    "max_abs_pitch_deg",            # worst nose-down/up excursion
    "max_abs_roll_deg",             # worst lean excursion
    "mean_action_norm",             # policy output magnitude (measure of effort)
    "max_action_norm",              # peak policy output (spikes = instability)
    "stair_slope_deg",              # staircase slope from locomotion report
    "person_mask_count",            # depth mask events fired this run
    # --- Scene config ---
    "stair_preset",
    "step_count",
    "step_height_m",
    "step_depth_m",
    # --- Physical config (from robot_config event in isaac_env.jsonl) ---
    "go2_trunk_mass_kg",
    "o2_attached",
    "o2_tank_mass_kg",
    "o2_rail_mass_kg",
    "o2_total_payload_kg",
    "o2_length_m",
    "o2_width_m",
    "o2_height_m",
    "o2_mount_x_m",
    "o2_mount_y_m",
    "o2_mount_z_m",
    "o2_com_shift_x_mm",
    "o2_com_shift_z_mm",
    "o2_pitch_torque_nm",
    "o2_strap_break_n",
    # --- Controller args (what was being tested) ---
    "locomotion_mode",
    "parkour_heading_mode",
    "follow_backend",
    "trans_x_max",
    "trans_x_tolerance",
    "trans_x_alpha",
    "kp",
    "kd",
    "target_distance",
    "sim_latency_ms",
    "stair_speed_scale",
    "stair_forward_floor",
    "stair_near_distance",
    "stair_latch_frames",
    "sim2real_validation_cam",
    # --- Features active / inactive ---
    "obstacle_stop_enabled",
    "parkour_person_mask_enabled",
    # --- Controller-side trace (debug/debug_trace/vision_main_trace.jsonl) ---
    "yolo_total_frames",            # total frames processed by controller
    "yolo_detect_count",            # frames where YOLO detected a person
    "yolo_detect_pct",              # detection rate (%)
    "mean_frame_latency_ms",        # mean total loop time per frame
    "mean_pose_infer_ms",           # mean pose inference time per frame
    "mean_yaw_err_deg",             # mean bearing error to person (when detected)
    "stall_count",                  # frames flagged as stalled
    "person_lost_count",            # times person detection dropped (True->False transition)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _parse_arg(cmd: str, flag: str) -> Optional[str]:
    """Extract the value after --flag in a command string."""
    m = re.search(r"(?:^|\s)" + re.escape(flag) + r"\s+([^\s\"']+)", cmd)
    return m.group(1) if m else None


def _flag_absent(cmd: str, flag: str) -> bool:
    """Return True if --flag does NOT appear in the command string."""
    return not bool(re.search(r"(?:^|\s)" + re.escape(flag) + r"(?:\s|$)", cmd))


def _get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _get_git_info() -> Tuple[str, str]:
    """Return (short_sha, first_line_of_message) for current HEAD."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h|%s"],
            capture_output=True, text=True, timeout=5,
        )
        raw = result.stdout.strip()
        if "|" in raw:
            sha, msg = raw.split("|", 1)
            return sha.strip(), msg.strip()
        return raw.strip(), ""
    except Exception:
        return "", ""


def _extract_trace(run_folder: Path) -> Dict[str, Any]:
    """Extract per-frame controller stats from debug/debug_trace/vision_main_trace.jsonl."""
    trace_path = run_folder / "debug" / "debug_trace" / "vision_main_trace.jsonl"
    if not trace_path.exists():
        return {}

    total_frames = 0
    detect_frames = 0
    latencies: List[float] = []
    infer_times: List[float] = []
    yaw_errs: List[float] = []
    stall_count = 0
    person_lost = 0
    prev_detected: Optional[bool] = None

    try:
        with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if ev.get("event") != "frame_timing":
                    continue

                total_frames += 1
                data = ev.get("data") or {}

                stage_ms = data.get("stage_ms") or {}
                total_loop = _safe_float(stage_ms.get("total_loop"))
                pose_infer = _safe_float(stage_ms.get("pose_infer"))
                if total_loop is not None:
                    latencies.append(total_loop)
                if pose_infer is not None:
                    infer_times.append(pose_infer)

                if data.get("stall_suspected"):
                    stall_count += 1

                debug_info = data.get("debug_info") or {}
                detected = bool(debug_info.get("person_detected", False))
                if detected:
                    detect_frames += 1
                if prev_detected is True and not detected:
                    person_lost += 1
                prev_detected = detected

                yaw_err = _safe_float(debug_info.get("rotation_error_deg"))
                if yaw_err is not None and detected:
                    yaw_errs.append(abs(yaw_err))

    except Exception:
        pass

    if total_frames == 0:
        return {}

    result: Dict[str, Any] = {
        "yolo_total_frames": total_frames,
        "yolo_detect_count": detect_frames,
        "yolo_detect_pct": round(detect_frames / total_frames * 100, 1),
        "stall_count": stall_count,
        "person_lost_count": person_lost,
    }
    if latencies:
        result["mean_frame_latency_ms"] = round(sum(latencies) / len(latencies), 1)
    if infer_times:
        result["mean_pose_infer_ms"] = round(sum(infer_times) / len(infer_times), 1)
    if yaw_errs:
        result["mean_yaw_err_deg"] = round(sum(yaw_errs) / len(yaw_errs), 2)

    return result


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_metrics(run_folder: Path, git_branch: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: None for col in COLUMNS}
    row["run_id"] = run_folder.name
    row["git_branch"] = git_branch if git_branch else _get_git_branch()

    # Git identity
    sha, msg = _get_git_info()
    row["git_commit_sha"] = sha or None
    row["git_commit_msg"] = msg or None

    # ------------------------------------------------------------------
    # 1. logs/status.jsonl -- launcher timeline
    # ------------------------------------------------------------------
    status_path = run_folder / "logs" / "status.jsonl"
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stage = ev.get("stage", "")
                state = ev.get("state", "")

                if stage == "setup" and state == "start":
                    row["timestamp"] = ev.get("timestamp")
                    row["sim2real_validation_cam"] = bool(ev.get("sim2real_validation_cam", False))
                    row["parkour_heading_mode"] = ev.get("parkour_heading_mode", "vision")
                    row["locomotion_mode"] = ev.get("locomotion_mode", "parkour")

                elif stage == "docker" and state == "start":
                    cmd = ev.get("command", "")
                    row["follow_backend"] = _parse_arg(cmd, "--follow-backend") or "pid"
                    row["trans_x_max"] = _safe_float(_parse_arg(cmd, "--trans-x-max"))
                    row["trans_x_tolerance"] = _safe_float(_parse_arg(cmd, "--trans-x-tolerance"))
                    row["trans_x_alpha"] = _safe_float(_parse_arg(cmd, "--trans-x-alpha"))
                    row["kp"] = _safe_float(_parse_arg(cmd, "--kp"))
                    row["kd"] = _safe_float(_parse_arg(cmd, "--kd"))
                    row["target_distance"] = _safe_float(_parse_arg(cmd, "--target-distance"))
                    row["sim_latency_ms"] = _safe_float(_parse_arg(cmd, "--sim-latency-ms"))
                    row["stair_speed_scale"] = _safe_float(_parse_arg(cmd, "--stair-speed-scale"))
                    row["stair_forward_floor"] = _safe_float(_parse_arg(cmd, "--stair-forward-floor"))
                    row["stair_near_distance"] = _safe_float(_parse_arg(cmd, "--stair-near-distance"))
                    row["stair_latch_frames"] = _safe_int(_parse_arg(cmd, "--stairs-latch-frames"))
                    row["obstacle_stop_enabled"] = _flag_absent(cmd, "--no-obstacle-stop")
                    row["parkour_person_mask_enabled"] = _flag_absent(cmd, "--no-parkour-person-mask")

                elif stage == "sim_gate":
                    row["sim_gate_state"] = state

                elif stage == "summary" and state == "complete":
                    if row.get("outcome") is None:
                        row["outcome"] = "completed"

    # ------------------------------------------------------------------
    # 2. reports/stair_demo_report.json -- final robot state + outcome
    # ------------------------------------------------------------------
    report_path = run_folder / "reports" / "stair_demo_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            row["exit_reason"] = report.get("exit_reason")
            row["motion_elapsed_sec"] = _safe_float(report.get("motion_elapsed_sim_sec"))
            row["stair_phase_sec"] = _safe_float(report.get("robot_stair_phase_sim_sec"))

            demo = report.get("stair_demo", {})
            robot = demo.get("robot", {})
            if robot:
                row["final_x_m"] = _safe_float(robot.get("x_m"))
                row["final_y_m"] = _safe_float(robot.get("y_m"))
                row["final_height_m"] = _safe_float(robot.get("height_m"))
                row["final_pitch_deg"] = _safe_float(robot.get("pitch_deg"))
                row["final_roll_deg"] = _safe_float(robot.get("roll_deg"))
                row["final_yaw_deg"] = _safe_float(robot.get("yaw_deg"))
                row["fall_type"] = robot.get("fall_type")

            loco = demo.get("locomotion", {})
            if loco:
                row["stair_slope_deg"] = _safe_float(loco.get("stair_slope_deg"))

            er = row.get("exit_reason")
            if er == "robot_fell":
                row["outcome"] = "robot_fell"
            elif er == "patient_reached_top":
                row["outcome"] = "completed"
            elif er == "timeout":
                row["outcome"] = "timeout"
            elif er and row.get("outcome") is None:
                row["outcome"] = er

        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3. debug/isaac_env.jsonl -- physics stream
    # ------------------------------------------------------------------
    isaac_path = run_folder / "debug" / "isaac_env.jsonl"
    fall_diags: List[Dict] = []
    stair_climb_reached = False
    stair_preset_seen = False
    person_mask_count = 0
    patient_stair_reached = False

    if isaac_path.exists():
        with open(isaac_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                action = ""
                event_block = ev.get("event")
                if isinstance(event_block, dict):
                    action = event_block.get("action", "")
                sim = ev.get("sim") or {}

                if action == "fall_diag":
                    fall_diags.append(sim)

                elif action == "stair_preset_configured" and not stair_preset_seen:
                    row["stair_preset"] = sim.get("preset")
                    row["step_count"] = _safe_int(sim.get("step_count"))
                    row["step_height_m"] = _safe_float(sim.get("step_height_m"))
                    row["step_depth_m"] = _safe_float(sim.get("step_depth_m"))
                    stair_preset_seen = True

                elif action == "robot_config":
                    row["go2_trunk_mass_kg"] = _safe_float(sim.get("go2_trunk_mass_kg"))
                    row["o2_attached"] = sim.get("o2_attached")
                    row["o2_tank_mass_kg"] = _safe_float(sim.get("o2_tank_mass_kg"))
                    row["o2_rail_mass_kg"] = _safe_float(sim.get("o2_rail_mass_kg"))
                    row["o2_total_payload_kg"] = _safe_float(sim.get("o2_total_payload_kg"))
                    row["o2_length_m"] = _safe_float(sim.get("o2_length_m"))
                    row["o2_width_m"] = _safe_float(sim.get("o2_width_m"))
                    row["o2_height_m"] = _safe_float(sim.get("o2_height_m"))
                    row["o2_mount_x_m"] = _safe_float(sim.get("o2_mount_x_m"))
                    row["o2_mount_y_m"] = _safe_float(sim.get("o2_mount_y_m"))
                    row["o2_mount_z_m"] = _safe_float(sim.get("o2_mount_z_m"))
                    row["o2_com_shift_x_mm"] = _safe_float(sim.get("o2_com_shift_x_mm"))
                    row["o2_com_shift_z_mm"] = _safe_float(sim.get("o2_com_shift_z_mm"))
                    row["o2_pitch_torque_nm"] = _safe_float(sim.get("o2_pitch_torque_nm"))
                    row["o2_strap_break_n"] = _safe_float(sim.get("o2_strap_break_n"))

                elif action == "robot_stair_climb_visible":
                    stair_climb_reached = True

                elif action == "parkour_person_depth_masked":
                    person_mask_count += 1

                elif action == "patient_stair_phase_started":
                    patient_stair_reached = True

    row["stair_climb_reached"] = stair_climb_reached
    row["patient_stair_phase_reached"] = patient_stair_reached
    row["person_mask_count"] = person_mask_count if person_mask_count > 0 else None
    row["fall_diag_steps"] = len(fall_diags)

    if fall_diags:
        last = fall_diags[-1]
        row["final_x_m"] = _safe_float(last.get("x"))
        row["final_height_m"] = _safe_float(last.get("h"))
        row["final_pitch_deg"] = _safe_float(last.get("pitch"))
        row["final_roll_deg"] = _safe_float(last.get("roll"))

        xs = [_safe_float(d.get("x")) for d in fall_diags if d.get("x") is not None]
        row["max_x_m"] = max(xs) if xs else None

        pitches = [abs(_safe_float(d.get("pitch"), 0)) for d in fall_diags if d.get("pitch") is not None]
        row["max_abs_pitch_deg"] = max(pitches) if pitches else None

        rolls = [abs(_safe_float(d.get("roll"), 0)) for d in fall_diags if d.get("roll") is not None]
        row["max_abs_roll_deg"] = max(rolls) if rolls else None

        vxs: List[float] = []
        yws: List[float] = []
        for d in fall_diags:
            pc = d.get("policy_cmd")
            if isinstance(pc, list):
                if len(pc) > 0:
                    v = _safe_float(pc[0])
                    if v is not None:
                        vxs.append(v)
                if len(pc) > 2:
                    w = _safe_float(pc[2])
                    if w is not None:
                        yws.append(w)
        row["mean_vx_cmd_mps"] = round(sum(vxs) / len(vxs), 4) if vxs else None
        row["mean_yaw_cmd_rps"] = round(sum(yws) / len(yws), 4) if yws else None

        anorms = [_safe_float(d.get("action_norm")) for d in fall_diags if d.get("action_norm") is not None]
        if anorms:
            row["mean_action_norm"] = round(sum(anorms) / len(anorms), 4)
            row["max_action_norm"] = round(max(anorms), 4)

    # ------------------------------------------------------------------
    # 4. debug/debug_trace/vision_main_trace.jsonl -- controller trace
    # ------------------------------------------------------------------
    row.update(_extract_trace(run_folder))

    # ------------------------------------------------------------------
    # Round floats for legibility
    # ------------------------------------------------------------------
    for k in [
        "final_x_m", "max_x_m", "final_y_m", "final_height_m",
        "final_pitch_deg", "final_roll_deg", "final_yaw_deg",
        "max_abs_pitch_deg", "max_abs_roll_deg",
        "mean_action_norm", "max_action_norm",
        "step_height_m", "step_depth_m",
        "motion_elapsed_sec", "stair_phase_sec",
        "mean_vx_cmd_mps", "mean_yaw_cmd_rps",
        "trans_x_max", "trans_x_tolerance", "trans_x_alpha",
        "kp", "kd", "target_distance", "sim_latency_ms",
        "stair_speed_scale", "stair_forward_floor", "stair_near_distance",
        "stair_slope_deg",
        "go2_trunk_mass_kg", "o2_tank_mass_kg", "o2_rail_mass_kg", "o2_total_payload_kg",
        "o2_length_m", "o2_width_m", "o2_height_m",
        "o2_mount_x_m", "o2_mount_y_m", "o2_mount_z_m",
        "o2_com_shift_x_mm", "o2_com_shift_z_mm",
        "o2_pitch_torque_nm", "o2_strap_break_n",
    ]:
        if row.get(k) is not None:
            row[k] = round(float(row[k]), 3)

    if row.get("outcome") is None:
        row["outcome"] = "unknown"

    return row


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _sort_key(r: Dict) -> tuple:
    """Best runs first: max_x_m desc, then timestamp desc."""
    try:
        x = float(r.get("max_x_m") or 0)
    except (TypeError, ValueError):
        x = 0.0
    ts = str(r.get("timestamp") or "")
    return (-x, ts)


def upsert_csv(table_path: Path, row: Dict[str, Any]) -> None:
    existing: List[Dict] = []
    if table_path.exists():
        with open(table_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("run_id") != row["run_id"]:
                    existing.append(r)

    new_row = {col: ("" if row.get(col) is None else str(row[col])) for col in COLUMNS}
    existing.append(new_row)
    existing.sort(key=_sort_key)

    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)


def upsert_jsonl(jsonl_path: Path, row: Dict[str, Any]) -> None:
    lines: List[str] = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") != row["run_id"]:
                        lines.append(line)
                except json.JSONDecodeError:
                    lines.append(line)

    lines.append(json.dumps(row, default=str))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract run metrics and update performance table.")
    parser.add_argument("run_folder", help="Path to the run_sim_* folder")
    parser.add_argument("--git-branch", default=None, help="Git branch name (passed from launcher)")
    args = parser.parse_args()

    run_folder = Path(args.run_folder).resolve()
    if not run_folder.is_dir():
        print(f"[perf_table] ERROR: run folder not found: {run_folder}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table_csv = DATA_DIR / "performance_table.csv"
    table_jsonl = DATA_DIR / "performance_table.jsonl"

    print(f"[perf_table] Extracting: {run_folder.name}", flush=True)
    row = extract_metrics(run_folder, git_branch=args.git_branch)

    print(
        f"[perf_table] outcome={str(row['outcome']):40s} "
        f"max_x={row['max_x_m']}m  "
        f"stair_climb={row['stair_climb_reached']}  "
        f"steps={row['fall_diag_steps']}  "
        f"sha={row['git_commit_sha']}  "
        f"branch={row['git_branch']}",
        flush=True,
    )

    upsert_csv(table_csv, row)
    upsert_jsonl(table_jsonl, row)

    print(f"[perf_table] -> {table_csv}", flush=True)
    print(f"[perf_table] -> {table_jsonl}", flush=True)

    try:
        from charts import generate_all
        generate_all(table_jsonl, DATA_DIR / "charts")
    except Exception as exc:
        print(f"[perf_table] Charts skipped: {exc}", flush=True)

    print(f"[perf_table] data dir: {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
