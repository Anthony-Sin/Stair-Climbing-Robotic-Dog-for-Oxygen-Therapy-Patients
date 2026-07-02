"""Episode discovery + aggregation for the stair-sweep presenter.

Split out of ``sweep_present.py`` (single-responsibility): turns a manifest / run-dir list /
``--auto`` glob into enriched episode dicts (parsed metrics + scene config + hero clip), the
per-riser summary row, and the BEST-hero selector.
"""
import glob
import json
import os
import sys

# Sibling import (analyze_climb.py lives next to this file) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_climb  # noqa: E402

from sweep_constants import (
    LOG_DIR, DEFAULT_STEP_DEPTH_M, DEFAULT_STEP_COUNT, DEFAULT_TOP_EDGE_X,
)
from sweep_helpers import (
    have, riser_short, verdict_short, pick_video, ffprobe_nframes, scan_scene_events,
)


# ---------------------------------------------------------------------------
# episode discovery + aggregation
# ---------------------------------------------------------------------------
def _resolve_run_dir(path):
    if os.path.isabs(path) and os.path.isdir(path):
        return path
    cand = os.path.join(LOG_DIR, path)
    return cand if os.path.isdir(cand) else path


def discover_episodes(args):
    """Return a list of {height, label, run_dir, waypoint} dicts from the chosen source."""
    eps = []
    if args.manifest:
        with open(args.manifest, encoding="utf-8-sig") as f:
            man = json.load(f)
        eps_raw = man.get("episodes", man) if isinstance(man, dict) else man
        if isinstance(eps_raw, dict):     # PowerShell ConvertTo-Json collapses a 1-elem array
            eps_raw = [eps_raw]
        for e in eps_raw:
            rd = _resolve_run_dir(str(e.get("run_dir", "")))
            eps.append({
                "height": e.get("height"), "label": e.get("label"),
                "run_dir": rd, "waypoint": e.get("waypoint"),
            })
    elif args.run_dirs:
        heights = args.heights or [None] * len(args.run_dirs)
        for rd, h in zip(args.run_dirs, heights):
            eps.append({"height": h, "label": None, "run_dir": _resolve_run_dir(rd), "waypoint": None})
    else:  # --auto
        cands = sorted(glob.glob(os.path.join(LOG_DIR, "run_sim_*")), key=os.path.getmtime, reverse=True)
        for rd in cands[: args.auto]:
            eps.append({"height": None, "label": None, "run_dir": rd, "waypoint": None})
    return eps


def build_episode(ep):
    """Enrich a discovered episode with parsed metrics, scene config and its hero clip."""
    rd = ep["run_dir"]
    jsonl = os.path.join(rd, "debug", "isaac_env.jsonl")
    result = analyze_climb.analyze_run(rd) if os.path.isdir(rd) else {"rows": [], "stats": None}
    scene = scan_scene_events(jsonl)
    stats = result.get("stats")
    rows = result.get("rows", [])

    height = ep.get("height")
    if height is None:
        height = scene.get("step_height_m")
    video, vdur = pick_video(rd) if os.path.isdir(rd) else (None, None)

    verdict = stats["verdict"] if stats else "NO DATA (no fall_diag rows)"
    waypoint = ep.get("waypoint") or scene.get("waypoint_status") or "unknown"
    # "completion time" preference: the waypoint-reached event, else on-stairs climb time, else span.
    real_time = scene.get("waypoint_time")
    if real_time is None and stats:
        real_time = stats.get("climb_time_s") or stats.get("total_t")

    return {
        "height": height,
        "label": ep.get("label"),
        "label_short": riser_short(height),
        "run_dir": rd,
        "run_leaf": os.path.basename(rd.rstrip("/\\")),
        "rows": rows,
        "stats": stats,
        "verdict": verdict,
        "verdict_short": verdict_short(verdict),
        "waypoint": waypoint,
        "real_time_s": real_time,
        "video": video,
        "video_dur_s": vdur,
        "video_frames": ffprobe_nframes(video) if (video and have("ffprobe")) else None,
        "step_depth_m": scene.get("step_depth_m") or DEFAULT_STEP_DEPTH_M,
        "step_count": scene.get("step_count") or DEFAULT_STEP_COUNT,
        "top_height_m": scene.get("top_height_m"),
    }


def summary_row(ep):
    s = ep.get("stats") or {}
    return {
        "riser_m": ep["height"],
        "label_short": ep["label_short"],
        "label": ep.get("label") or "",
        "verdict": ep["verdict"],
        "verdict_short": ep["verdict_short"],
        "steps_climbed": round(s["steps_climbed"], 2) if s.get("steps_climbed") is not None else None,
        "max_x_m": round(s["max_x"], 3) if s.get("max_x") is not None else None,
        "top_edge_x_m": DEFAULT_TOP_EDGE_X,
        "max_tilt_deg": round(s["max_tilt"], 1) if s.get("max_tilt") is not None else None,
        "min_h_on_m": round(s["min_h_on"], 3) if s.get("min_h_on") is not None else None,
        "mean_pitch_on_deg": round(s["pitch_mean_on"], 1) if s.get("pitch_mean_on") is not None else None,
        "climb_time_s": round(s["climb_time_s"], 1) if s.get("climb_time_s") is not None else None,
        "real_time_s": round(ep["real_time_s"], 1) if isinstance(ep.get("real_time_s"), (int, float)) else None,
        "waypoint": ep["waypoint"],
        "step_depth_m": ep["step_depth_m"],
        "step_count": ep["step_count"],
        "top_height_m": ep.get("top_height_m"),
        "video": ep["video"] or "",
        "video_dur_s": round(ep["video_dur_s"], 2) if ep.get("video_dur_s") else None,
        "video_frames": ep.get("video_frames"),
        "run_dir": ep["run_dir"],
    }


def pick_hero(eps, mode):
    """Index of the episode to badge as BEST (or None)."""
    scored = [(i, e) for i, e in enumerate(eps) if e.get("stats")]
    if not scored:
        return None
    if mode == "steepest":
        return max(scored, key=lambda ie: ie[1].get("height") or 0)[0]
    if mode == "highest_clean":
        clean = [ie for ie in scored if ie[1]["verdict_short"] == "CLEAN"]
        if clean:
            return max(clean, key=lambda ie: ie[1].get("height") or 0)[0]
        # fall through to best
    # best: most steps climbed, tie-break by verdict rank then riser
    rank = {"CLEAN": 3, "INCOMPLETE": 2, "COLLIDED": 1, "FELL": 0, "NO REACH": 0, "NO DATA": -1}

    def key(ie):
        e = ie[1]
        return (
            e["stats"].get("steps_climbed") or 0,
            rank.get(e["verdict_short"], -1),
            e.get("height") or 0,
        )
    return max(scored, key=key)[0]
