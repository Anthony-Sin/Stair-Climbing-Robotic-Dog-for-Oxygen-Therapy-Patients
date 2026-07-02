"""Follow-mode reach classification + JSONL scan for the person-follow presenter.

Split out of ``follow_present.py`` (single-responsibility): the follow-mode reach-color
palette, the ``reach_short`` normalizer, and ``scan_follow_events`` (the follow-mode replacement
for ``sweep_present.scan_scene_events``, reading stair config + approach-time + reach-status
out of a run's ``debug/isaac_env.jsonl``).
"""
import json
import os

from analyze_climb import STAIR_BASE_X


# ---------------------------------------------------------------------------
# follow-mode reach colors (parallel to VERDICT_COLORS in sweep_present)
# ---------------------------------------------------------------------------
REACH_COLORS = {
    "REACHED TOP":   "#2e7d32",   # green -- made it
    "DID NOT REACH": "#546e7a",   # blue-grey -- stopped before top
    "FELL":          "#c62828",   # red -- tipped over
    "INCOMPLETE":    "#9e9e9e",   # grey
    "NO DATA":       "#bdbdbd",   # light grey
}


def reach_short(s):
    if not s:
        return "NO DATA"
    u = s.upper()
    if "REACHED TOP" in u:
        return "REACHED TOP"
    if "FELL" in u or "FLIPPED" in u or "COLLAPSED" in u:
        return "FELL"
    if "DID NOT" in u or "NO REACH" in u:
        return "DID NOT REACH"
    if "INCOMPLETE" in u:
        return "INCOMPLETE"
    return "NO DATA"


# ---------------------------------------------------------------------------
# follow-specific JSONL scan (replaces scan_scene_events for follow mode)
# ---------------------------------------------------------------------------
def scan_follow_events(jsonl):
    """Read stair config + follow-specific metrics from a run's JSONL."""
    out = {
        "step_height_m": None, "step_depth_m": None, "step_count": None,
        "top_height_m": None,
        "approach_time_s": None,  # sim-time when robot first reached x >= STAIR_BASE_X
        "reach_status": None,     # REACHED TOP / DID NOT REACH / FELL
    }
    if not os.path.exists(jsonl):
        return out
    first_approach = None
    max_x = None
    for line in open(jsonl, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        sim = d.get("sim", {}) or {}
        if isinstance(ev, dict):
            act = ev.get("action")
            if act == "stair_preset_configured":
                for k in ("step_height_m", "step_depth_m", "step_count", "top_height_m"):
                    if out[k] is None and sim.get(k) is not None:
                        out[k] = sim.get(k)
        # Track per-step metrics for approach_time and reach_status
        t = d.get("t")
        x = d.get("x") or sim.get("x")
        fall_type = d.get("fall_type") or sim.get("fall_type")
        if fall_type in ("flipped", "collapsed_low") and out["reach_status"] is None:
            out["reach_status"] = "FELL"
        if isinstance(x, (int, float)):
            if max_x is None or x > max_x:
                max_x = x
            if isinstance(t, (int, float)) and x >= STAIR_BASE_X and first_approach is None:
                first_approach = t
    if out["reach_status"] is None:
        if max_x is not None and max_x >= 5.8:
            out["reach_status"] = "REACHED TOP"
        elif max_x is not None:
            out["reach_status"] = "DID NOT REACH"
    out["approach_time_s"] = first_approach
    return out
