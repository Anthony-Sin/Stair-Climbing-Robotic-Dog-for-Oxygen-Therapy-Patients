"""Small stateless helpers for the stair-sweep presenter: logging, ffprobe wrappers,
label/verdict formatting, hero-clip selection, and the scene-event JSONL scan.

Split out of ``sweep_present.py`` (single-responsibility). Pure/IO-light utilities shared
by episode aggregation, graphs, and the montage builder.
"""
import json
import os
import shutil
import subprocess

from sweep_constants import RISER_SHORT, VIDEO_PREFERENCE


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[sweep_present] {msg}", flush=True)


def have(exe):
    return shutil.which(exe) is not None


def riser_short(h):
    if h is None:
        return "run"
    key = min(RISER_SHORT, key=lambda k: abs(k - h)) if RISER_SHORT else None
    if key is not None and abs(key - h) < 0.005:
        return RISER_SHORT[key]
    return f"~{h * 39.3701:.0f}in"


def verdict_short(v):
    if not v:
        return "NO DATA"
    v = v.upper()
    if v.startswith("CLEAN"):
        return "CLEAN"
    if v.startswith("COLLIDED"):
        return "COLLIDED"
    if v.startswith("FELL"):
        return "FELL"
    if v.startswith("DID NOT REACH"):
        return "NO REACH"
    if v.startswith("INCOMPLETE"):
        return "INCOMPLETE"
    return "NO DATA"


def fmt_time(t):
    return f"{t:.1f}s" if isinstance(t, (int, float)) else "n/a"


def _write_caption(path, *lines):
    # newline="\n" is load-bearing: Windows text mode would emit \r\n, and ffmpeg drawtext
    # renders the stray \r as an extra blank line (pushing line 2 out of the caption bar).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def _ffprobe(path, entries, stream=False):
    cmd = ["ffprobe", "-v", "error", "-of", "default=nk=1:nw=1", "-show_entries", entries]
    if stream:
        cmd[3:3] = ["-select_streams", "v:0", "-count_packets"]
    cmd.append(path)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout.strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def ffprobe_duration(path):
    v = _ffprobe(path, "format=duration")
    try:
        d = float(v)
        return d if d > 0 else None
    except (TypeError, ValueError):
        return None


def ffprobe_nframes(path):
    v = _ffprobe(path, "stream=nb_read_packets", stream=True)
    try:
        n = int(v)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def pick_video(run_dir):
    """Return the best usable hero clip for a run, or None (validated via ffprobe)."""
    vids = os.path.join(run_dir, "videos")
    for name in VIDEO_PREFERENCE:
        p = os.path.join(vids, name)
        if os.path.exists(p):
            d = ffprobe_duration(p) if have("ffprobe") else None
            if d and d > 0.05:
                return p, d
            log(f"WARNING: {p} exists but is empty/unreadable (0-frame) -- skipping it")
    return None, None


def scan_scene_events(jsonl):
    """Read the one-off scene/waypoint events from a run's JSONL (defaults when absent)."""
    out = {
        "step_height_m": None, "step_depth_m": None, "step_count": None,
        "top_height_m": None, "waypoint_status": None, "waypoint_time": None,
    }
    if not os.path.exists(jsonl):
        return out
    for line in open(jsonl, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        if not isinstance(ev, dict):
            continue
        act = ev.get("action")
        # NOTE: the structured logger nests the payload under `sim`, not `event` (which only
        # carries {"action": ...}). For stair_preset_configured `sim` is the stair config dict
        # (preset/step_*); for fall_diag/waypoint events `sim` is the telemetry snapshot (t,x,...).
        sim = d.get("sim", {}) or {}
        if act == "stair_preset_configured":
            for k in ("step_height_m", "step_depth_m", "step_count", "top_height_m"):
                if out[k] is None and sim.get(k) is not None:
                    out[k] = sim.get(k)
        elif act == "robot_reached_stair_waypoint" and out["waypoint_status"] is None:
            out["waypoint_status"] = "PASS (upright, held 2s)"
            out["waypoint_time"] = sim.get("t")
        elif act == "stair_waypoint_collision" and out["waypoint_status"] is None:
            out["waypoint_status"] = "FAIL (collided)"
            out["waypoint_time"] = sim.get("t")
    return out
