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
import sys

# ---------------------------------------------------------------------------
# Facade re-exports — every previously top-level name is preserved here so that
# `python analyze_run.py` and `from analyze_run import <name>` keep working
# exactly as before. Behaviour is unchanged; the bodies now live in sibling
# modules. Do NOT reorder or "clean up" these imports.
# ---------------------------------------------------------------------------
from analyze_run_constants import (  # noqa: F401
    FRAMES_PER_VIDEO,
    LABEL_HEIGHT,
    STAIR_START_X_M,
    THUMB_H,
    THUMB_W,
    VERIFICATION_PNGS,
    VIDEO_NAMES,
    _CV2_AVAILABLE,
    cv2,
    np,
)
from analyze_run_contact_sheet import (  # noqa: F401
    _label_frame,
    _thumb,
    build_contact_sheet,
    extract_video_frames,
    load_png_as_bgr,
)
from analyze_run_extract import (  # noqa: F401
    extract_final_scene_info,
    extract_key_events,
    extract_mask_events,
    extract_outcome_from_jsonl,
    extract_scene_config,
    extract_trajectory,
    extract_warnings,
)
from analyze_run_io import (  # noqa: F401
    _action,
    _level,
    _sim,
    _ts,
    load_jsonl,
    parse_evaluation_summary,
    parse_stair_demo_report,
    resolve_run_dir,
)
from analyze_run_metrics import (  # noqa: F401
    analyze_trajectory,
    build_pass_fail,
)
from analyze_run_report import (  # noqa: F401
    build_ai_analysis,
    build_markdown_report,
)


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
