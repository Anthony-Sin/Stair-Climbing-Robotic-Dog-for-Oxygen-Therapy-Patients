"""Build a slide-ready presentation pack from a stair-height sweep.

`run_stair_sweep.ps1` drives the blind-RL climb policy up several riser heights (one episode
per height) and leaves each episode in its own ``log/run_sim_<stamp>/`` (videos + a
``debug/isaac_env.jsonl`` physics ground-truth stream). This tool turns that scatter into one
folder you can drop straight into slides:

  <out>/
    sweep_summary.csv / sweep_summary.json     the data section (one row per riser)
    graphs/g1_climb_profile.png ... g5_dashboard.png
    stats_card.png                             leaderboard card (also montage cell 6)
    stair_sweep_montage.mp4                    2x3 grid: 5 riser videos + stats card,
                                               speed-normalized so they all finish together
    clips/clip_<riser>.mp4                     each riser standalone (labeled + speed-normalized)

It reads ONLY recorded artifacts, so it is runnable standalone on existing logs (no Isaac):

    python sweep_present.py --manifest log/stair_sweep_<stamp>/manifest.json
    python sweep_present.py --auto 5
    python sweep_present.py --run-dirs <d1> <d2> ... --heights 0.1 0.125 0.15

The honest climb verdict (CLEAN / COLLIDED / FELL ...) is computed by analyze_climb.analyze_run
so this tool and the live sim share ONE source of truth for the thresholds.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

# Sibling import (analyze_climb.py lives next to this file) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_climb  # noqa: E402
from analyze_climb import STAIR_BASE_X, STEP_RUN, FALL_TILT_DEG, COLLAPSE_H_M  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO_ROOT, "log")

# Commercial footprint the sweep holds constant (only the riser changes). Used as defaults
# when the per-run stair_preset_configured event cannot be parsed -- these match the preset
# run_stair_sweep.ps1 hard-codes, so the fallback is correct for this sweep.
DEFAULT_STEP_DEPTH_M = STEP_RUN     # 0.305
DEFAULT_STEP_COUNT = 14
DEFAULT_TOP_EDGE_X = 6.27           # forward x of the top step edge (constant across risers)

# Candidate hero clips per episode, best first. scene_view = cinematic chase (hero shot);
# topdown = autofit overview; follow_view = tracking chase.
VIDEO_PREFERENCE = ("scene_view.mp4", "topdown.mp4", "follow_view.mp4")

# Short human label per riser (m -> tag). Falls back to inches if unmatched.
RISER_SHORT = {
    0.100: '4"  gentle rise',
    0.125: '5"  hospital std',
    0.150: '6"  ADA maximum',
    0.178: '7"  IBC standard',
    0.198: '7.75"  comm. max',
}

VERDICT_COLORS = {
    "CLEAN": "#2e7d32",
    "COLLIDED": "#ef6c00",
    "FELL": "#c62828",
    "NO REACH": "#546e7a",
    "INCOMPLETE": "#9e9e9e",
    "NO DATA": "#bdbdbd",
}

# montage geometry
CANVAS_W, CANVAS_H = 1920, 1080
COLS, ROWS = 2, 3
CELL_W, CELL_H = CANVAS_W // COLS, CANVAS_H // ROWS   # 960 x 360
CAPTION_BAR_H = 64


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


# ---------------------------------------------------------------------------
# graphs (matplotlib, Agg)
# ---------------------------------------------------------------------------
def _assign_colors(eps):
    try:
        import matplotlib.cm as cm
        n = max(1, len(eps))
        cols = cm.plasma([0.05 + 0.85 * i / max(1, n - 1) for i in range(n)]) if n > 1 else cm.plasma([0.5])
        for e, c in zip(eps, cols):
            e["color"] = tuple(c)
    except Exception:
        for e in eps:
            e["color"] = (0.2, 0.4, 0.8, 1.0)


def _plot_climb_profile(ax, eps):
    plotted = 0
    for e in eps:
        pts = [(r.get("x"), r.get("h")) for r in e.get("rows", [])
               if r.get("x") is not None and r.get("h") is not None and r.get("x") >= -1.0]
        if not pts:
            continue
        xs, hs = zip(*pts)
        lab = f"{e['height']:.3f} m  {e['verdict_short']}" if e.get("height") is not None else e["verdict_short"]
        ax.plot(xs, hs, color=e.get("color"), lw=2.0, label=lab)
        ax.scatter([xs[-1]], [hs[-1]], color=e.get("color"), s=28, zorder=5, edgecolors="white", linewidths=0.6)
        plotted += 1
    ax.axvline(STAIR_BASE_X, ls="--", color="0.5", lw=1.2)
    ax.text(STAIR_BASE_X + 0.05, 0.03, "stair base", color="0.4", fontsize=8)
    ax.axhspan(0.22, 0.6, color="#2e7d32", alpha=0.06)
    ax.axhline(COLLAPSE_H_M, ls=":", color="#c62828", lw=1.0)
    ax.set_xlabel("forward distance  x (m)")
    ax.set_ylabel("body height above terrain (m)")
    ax.set_title("Climb stability: body height vs forward progress")
    ax.set_ylim(0, 0.6)
    if plotted:
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)


def _verdict_bar_colors(rows):
    return [VERDICT_COLORS.get(r["verdict_short"], "#9e9e9e") for r in rows]


def _xt(rows):
    return [f"{r['riser_m']:.3f}\n{r['label_short']}" if r.get("riser_m") is not None else r["label_short"]
            for r in rows]


def _plot_steps(ax, rows):
    xs = range(len(rows))
    vals = [r.get("steps_climbed") or 0 for r in rows]
    ax.bar(xs, vals, color=_verdict_bar_colors(rows))
    full = rows[0].get("step_count") or DEFAULT_STEP_COUNT
    ax.axhline(full, ls="--", color="0.5", lw=1.0)
    ax.text(len(rows) - 0.5, full + 0.2, f"full staircase ({full})", ha="right", color="0.4", fontsize=8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_xt(rows), fontsize=8)
    ax.set_ylabel("step-runs climbed past base")
    ax.set_title("How far up the stairs (steps climbed)")


def _plot_stability(ax, rows):
    xs = range(len(rows))
    vals = [r.get("max_tilt_deg") or 0 for r in rows]
    ax.bar(xs, vals, color=_verdict_bar_colors(rows))
    ax.axhspan(0, 18, color="#2e7d32", alpha=0.08)
    ax.axhline(FALL_TILT_DEG, ls="--", color="#c62828", lw=1.2)
    ax.text(len(rows) - 0.5, FALL_TILT_DEG + 1, f"fall line ({FALL_TILT_DEG:.0f}°)", ha="right",
            color="#c62828", fontsize=8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_xt(rows), fontsize=8)
    ax.set_ylabel("max body tilt (deg)")
    ax.set_title("Stability: peak tilt vs fall threshold")


def _plot_reach(ax, rows):
    xs = range(len(rows))
    vals = [r.get("max_x_m") or 0 for r in rows]
    ax.bar(xs, vals, color=_verdict_bar_colors(rows))
    ax.axhline(STAIR_BASE_X, ls=":", color="0.5", lw=1.0)
    ax.text(len(rows) - 0.5, STAIR_BASE_X + 0.05, "stair base", ha="right", color="0.4", fontsize=8)
    top = rows[0].get("top_edge_x_m") or DEFAULT_TOP_EDGE_X
    ax.axhline(top, ls="--", color="0.5", lw=1.0)
    ax.text(len(rows) - 0.5, top + 0.05, "top edge", ha="right", color="0.4", fontsize=8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_xt(rows), fontsize=8)
    ax.set_ylabel("max forward reach  x (m)")
    ax.set_title("Forward reach vs stair geometry")


def generate_graphs(eps, rows, graphs_dir):
    """Write the per-metric PNGs + a dashboard. Returns the list of written paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable ({exc}); skipping graphs")
        return []
    os.makedirs(graphs_dir, exist_ok=True)
    _assign_colors(eps)
    written = []

    def _save(fig, name):
        p = os.path.join(graphs_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(p)

    singles = [
        ("g1_climb_profile.png", lambda ax: _plot_climb_profile(ax, eps)),
        ("g2_steps_vs_riser.png", lambda ax: _plot_steps(ax, rows)),
        ("g3_stability_vs_riser.png", lambda ax: _plot_stability(ax, rows)),
        ("g4_reach_vs_riser.png", lambda ax: _plot_reach(ax, rows)),
    ]
    for name, fn in singles:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        try:
            fn(ax)
            _save(fig, name)
        except Exception as exc:
            log(f"WARNING: chart {name} failed: {exc}")
            plt.close(fig)

    # dashboard 2x2
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.4))
        _plot_climb_profile(axes[0][0], eps)
        _plot_steps(axes[0][1], rows)
        _plot_stability(axes[1][0], rows)
        _plot_reach(axes[1][1], rows)
        fig.suptitle("Stair-Climb Sweep — blind-RL climb policy + O2 payload", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, "g5_dashboard.png")
    except Exception as exc:
        log(f"WARNING: dashboard failed: {exc}")
    return written


def generate_stats_card(rows, meta, out_path, best_idx=None):
    """Render the leaderboard stats card (also embedded as montage cell 6)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable ({exc}); skipping stats card")
        return None

    BG      = "#0d1117"   # near-black canvas
    HEADER  = "#161b22"   # slightly lighter header panel
    ROW_ALT = "#0d1117"
    ROW_EVN = "#101419"
    GOLD    = "#f0c040"
    TEXT    = "#e6edf3"
    MUTED   = "#8b949e"
    ACCENT  = "#58a6ff"

    n = len(rows)
    card_h = 3.4 + n * 0.58   # taller for more rows
    fig = plt.figure(figsize=(12.8, card_h), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── title strip ──────────────────────────────────────────────────────────
    title_ax = fig.add_axes([0.0, 1 - 0.52 / card_h, 1.0, 0.52 / card_h])
    title_ax.set_facecolor(HEADER)
    title_ax.axis("off")
    title_ax.text(0.022, 0.72, "STAIR-CLIMB SWEEP", color=TEXT,
                  fontsize=17, fontweight="bold", va="center",
                  transform=title_ax.transAxes)
    title_ax.text(0.022, 0.22, "blind-RL policy  ·  commercial stairs  ·  +O₂ payload",
                  color=MUTED, fontsize=10, va="center",
                  transform=title_ax.transAxes)
    branch = meta.get("git_branch", "")
    commit = meta.get("git_commit", "")
    ref = f"{branch}@{commit}" if branch and commit else (branch or commit)
    if ref:
        title_ax.text(0.978, 0.5, ref, color=MUTED, fontsize=9, va="center",
                      ha="right", transform=title_ax.transAxes,
                      fontfamily="monospace")

    # ── column headers ────────────────────────────────────────────────────────
    COLS_DEF = [
        ("RISER",   0.060, "left"),
        ("CODE",    0.175, "left"),
        ("VERDICT", 0.360, "left"),
        ("STEPS",   0.590, "center"),
        ("PROGRESS",0.680, "left"),
        ("TIME",    0.930, "right"),
    ]
    hdr_top  = 1 - 0.52 / card_h
    hdr_h    = 0.38 / card_h
    hdr_ax   = fig.add_axes([0.0, hdr_top - hdr_h, 1.0, hdr_h])
    hdr_ax.set_facecolor("#1c2128")
    hdr_ax.axis("off")
    for lbl, xf, ha in COLS_DEF:
        hdr_ax.text(xf, 0.5, lbl, color=ACCENT, fontsize=8.5, fontweight="bold",
                    va="center", ha=ha, transform=hdr_ax.transAxes)

    # ── rows ──────────────────────────────────────────────────────────────────
    row_top = hdr_top - hdr_h
    row_h   = (row_top - 0.28 / card_h) / max(1, n)
    step_count = (rows[0].get("step_count") or DEFAULT_STEP_COUNT) if rows else DEFAULT_STEP_COUNT

    for i, r in enumerate(rows):
        is_best  = (best_idx is not None and i == best_idx)
        row_bg   = ROW_EVN if i % 2 == 0 else ROW_ALT
        y0 = row_top - (i + 1) * row_h

        row_ax = fig.add_axes([0.0, y0, 1.0, row_h])
        row_ax.set_facecolor(row_bg)
        row_ax.axis("off")

        # gold left border for best
        if is_best:
            row_ax.add_patch(FancyBboxPatch((0, 0), 0.004, 1.0,
                                            boxstyle="square,pad=0",
                                            facecolor=GOLD, edgecolor="none",
                                            transform=row_ax.transAxes, clip_on=False, zorder=5))

        vc_hex  = VERDICT_COLORS.get(r["verdict_short"], "#9e9e9e")
        riser   = f"{r['riser_m']:.3f} m" if r.get("riser_m") is not None else "-"
        steps   = r.get("steps_climbed") or 0
        steps_s = f"{steps:.1f}" if r.get("steps_climbed") is not None else "-"
        t       = r.get("real_time_s") or r.get("climb_time_s")
        t_s     = fmt_time(t)

        # riser column
        mark_col = GOLD if is_best else TEXT
        row_ax.text(0.060, 0.5, ("▶ " if is_best else "   ") + riser,
                    color=mark_col, fontsize=11,
                    fontweight="bold" if is_best else "normal",
                    va="center", ha="left", transform=row_ax.transAxes)

        # code column
        row_ax.text(0.175, 0.5, r["label_short"], color=MUTED,
                    fontsize=9, va="center", ha="left", transform=row_ax.transAxes)

        # verdict pill
        pill_x, pill_y, pill_w, pill_h = 0.358, 0.18, 0.175, 0.64
        row_ax.add_patch(FancyBboxPatch((pill_x, pill_y), pill_w, pill_h,
                                        boxstyle="round,pad=0.01",
                                        facecolor=vc_hex, edgecolor="none",
                                        transform=row_ax.transAxes, clip_on=True))
        row_ax.text(pill_x + pill_w / 2, 0.5, r["verdict_short"],
                    color="white", fontsize=9.5, fontweight="bold",
                    va="center", ha="center", transform=row_ax.transAxes)

        # steps number
        row_ax.text(0.590, 0.5, steps_s, color=TEXT,
                    fontsize=11, va="center", ha="center",
                    transform=row_ax.transAxes)

        # progress bar
        bar_x, bar_w_max = 0.640, 0.26
        bar_h_f = 0.28
        bar_y = (1 - bar_h_f) / 2
        # background track
        row_ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w_max, bar_h_f,
                                        boxstyle="round,pad=0.005",
                                        facecolor="#30363d", edgecolor="none",
                                        transform=row_ax.transAxes, clip_on=True))
        frac = min(1.0, steps / step_count) if step_count else 0
        if frac > 0.01:
            row_ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w_max * frac, bar_h_f,
                                            boxstyle="round,pad=0.005",
                                            facecolor=vc_hex, edgecolor="none",
                                            transform=row_ax.transAxes, clip_on=True))
        row_ax.text(bar_x + bar_w_max + 0.012, 0.5,
                    f"/{step_count}", color=MUTED, fontsize=8,
                    va="center", ha="left", transform=row_ax.transAxes)

        # time column
        row_ax.text(0.965, 0.5, t_s, color=MUTED,
                    fontsize=10, va="center", ha="right",
                    transform=row_ax.transAxes)

    # ── footer ────────────────────────────────────────────────────────────────
    ftr_h = 0.28 / card_h
    ftr_ax = fig.add_axes([0.0, 0.0, 1.0, ftr_h])
    ftr_ax.set_facecolor("#161b22")
    ftr_ax.axis("off")
    speeds = [r["video_dur_s"] / meta["montage_seconds"] for r in rows if r.get("video_dur_s")]
    avg = (sum(speeds) / len(speeds)) if speeds else None
    foot = f"video compressed to ~{meta['montage_seconds']:.0f}s per clip"
    if avg:
        foot += f"  ·  ≈{avg:.0f}× real-time avg"
    ftr_ax.text(0.022, 0.5, foot, color=MUTED, fontsize=8.5,
                va="center", transform=ftr_ax.transAxes)
    n_pass = sum(1 for r in rows if r.get("verdict_short") == "CLEAN")
    ftr_ax.text(0.978, 0.5, f"{n_pass}/{n} clean climbs",
                color=GOLD if n_pass > 0 else MUTED, fontsize=9.5, fontweight="bold",
                va="center", ha="right", transform=ftr_ax.transAxes)

    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# montage (pure filtergraph builder + ffmpeg orchestration)
# ---------------------------------------------------------------------------
def build_filtergraph(tiles, duration, fps=30, cell_w=CELL_W, cell_h=CELL_H,
                      canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """Compose the -filter_complex string for the montage. PURE: paths in, string out.

    Each tile is a dict with:
      kind:        "video" | "image" | "placeholder"
      input_index: ffmpeg -i index for video/image (None for placeholder)
      x, y:        top-left of the cell on the canvas
      m:           setpts multiplier (video only; <1 speeds up)
      pad:         seconds of cloned-tail padding (video only)
      fit:         "cover" | "letterbox" (video only)
      caption:     relative textfile name, or None
      best:        bool -- draw the BEST badge/border
      font:        relative font filename for drawtext
    Font + caption paths are relative (ffmpeg is run with cwd=the montage workdir) so no
    Windows drive-colon ever reaches the filtergraph parser.
    """
    parts = [f"color=c=black:s={canvas_w}x{canvas_h}:r={fps}:d={duration:.3f}[base]"]
    labels = []
    for i, t in enumerate(tiles):
        chain = []
        src = ""
        if t["kind"] == "video":
            src = f"[{t['input_index']}:v]"
            chain.append(f"setpts=PTS*{t['m']:.6f}")
            if t.get("fit") == "letterbox":
                chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease")
                chain.append(f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black")
            else:
                chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase")
                chain.append(f"crop={cell_w}:{cell_h}")
            if t.get("pad", 0) > 0.01:
                chain.append(f"tpad=stop_mode=clone:stop_duration={t['pad']:.3f}")
            chain.append(f"fps={fps}")
        elif t["kind"] == "image":
            src = f"[{t['input_index']}:v]"
            chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease")
            chain.append(f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:white")
            chain.append(f"fps={fps}")
        else:  # placeholder
            chain.append(f"color=c=0x141414:s={cell_w}x{cell_h}:r={fps}:d={duration:.3f}")

        if t.get("best"):
            chain.append(f"drawbox=x=0:y=0:w={cell_w}:h={cell_h}:color=gold@0.95:t=8")
        if t.get("caption"):
            font = t.get("font", "font.ttf")
            y0 = cell_h - CAPTION_BAR_H
            chain.append(f"drawbox=x=0:y={y0}:w={cell_w}:h={CAPTION_BAR_H}:color=black@0.55:t=fill")
            chain.append(
                f"drawtext=fontfile={font}:textfile={t['caption']}:fontcolor=white:"
                f"fontsize=25:x=16:y={y0 + 7}:line_spacing=6"
            )
        if t.get("best"):
            font = t.get("font", "font.ttf")
            chain.append("drawbox=x=0:y=0:w=118:h=34:color=gold@0.95:t=fill")
            chain.append(f"drawtext=fontfile={font}:text=BEST:fontcolor=black:fontsize=24:x=20:y=4")

        out_lbl = f"c{i}"
        parts.append(f"{src}{','.join(chain)}[{out_lbl}]")
        labels.append(out_lbl)

    prev = "base"
    for i, t in enumerate(tiles):
        last = i == len(tiles) - 1
        out_lbl = "out" if last else f"o{i}"
        parts.append(f"[{prev}][{labels[i]}]overlay=x={t['x']}:y={t['y']}:shortest=0[{out_lbl}]")
        prev = out_lbl
    return ";".join(parts)


def _compute_speeds(durs, target, mode):
    """Return (multipliers, pads) per duration so tiles finish at `target` seconds."""
    valid = [d for d in durs if d]
    if not valid:
        return [None] * len(durs), [0.0] * len(durs)
    if mode == "uniform":
        ref = max(valid)
        ms = [(target / ref) if d else None for d in durs]
    else:  # sync: every clip finishes at target
        ms = [(target / d) if d else None for d in durs]
    pads = []
    for d, m in zip(durs, ms):
        sped = d * m if (d and m) else 0.0
        pads.append(max(0.0, target - sped))
    return ms, pads


def render_montage(eps, stats_card, out_path, montage_seconds, mode, fit):
    """Build the 2x3 grid montage with ffmpeg. Returns out_path or None."""
    if not have("ffmpeg"):
        log("WARNING: ffmpeg not found on PATH; skipping montage")
        return None
    work = os.path.join(os.path.dirname(out_path), "_montage")
    os.makedirs(work, exist_ok=True)

    # resolve a font into the workdir (relative reference dodges drive-colon escaping)
    font_rel = "font.ttf"
    font_src = next((p for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
                                 r"C:\Windows\Fonts\segoeui.ttf") if os.path.exists(p)), None)
    if font_src:
        shutil.copyfile(font_src, os.path.join(work, font_rel))
    else:
        log("WARNING: no system font found; captions may not render")

    # fixed 5 riser cells (ascending) + stats card in cell 5
    grid = list(eps[:5]) + [None] * (5 - len(eps))
    durs = [e["video_dur_s"] if (e and e.get("video")) else None for e in grid]
    ms, pads = _compute_speeds(durs, montage_seconds, mode)

    inputs = []   # ffmpeg -i file list, in input-index order
    tiles = []
    for i, e in enumerate(grid):
        x, y = (i % COLS) * CELL_W, (i // COLS) * CELL_H
        if e and e.get("video"):
            cap = os.path.join(work, f"cap{i}.txt")
            h = e.get("height")
            l1 = (f"{h:.3f} m   {e['label_short']}" if h is not None else e["label_short"])
            l2 = f"{e['verdict_short']}   |   {fmt_time(e.get('real_time_s'))}"
            _write_caption(cap, l1, l2)
            idx = len(inputs)
            inputs.append(os.path.abspath(e["video"]))
            tiles.append({"kind": "video", "input_index": idx, "x": x, "y": y,
                          "m": ms[i] or 1.0, "pad": pads[i], "fit": fit,
                          "caption": f"cap{i}.txt", "best": bool(e.get("_best")),
                          "font": font_rel})
        else:
            cap = os.path.join(work, f"cap{i}.txt")
            label = (f"{e['height']:.3f} m" if (e and e.get("height") is not None) else "(no run)")
            note = "no video recorded" if e else ""
            _write_caption(cap, *([label, note] if note else [label]))
            tiles.append({"kind": "placeholder", "input_index": None, "x": x, "y": y,
                          "caption": f"cap{i}.txt", "font": font_rel})

    # stats card -> cell 5
    sx, sy = (5 % COLS) * CELL_W, (5 // COLS) * CELL_H
    if stats_card and os.path.exists(stats_card):
        idx = len(inputs)
        inputs.append(os.path.abspath(stats_card))
        tiles.append({"kind": "image", "input_index": idx, "x": sx, "y": sy})
    else:
        tiles.append({"kind": "placeholder", "input_index": None, "x": sx, "y": sy})

    fg = build_filtergraph(tiles, montage_seconds)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    n_video = sum(1 for t in tiles if t["kind"] == "video")
    for p in inputs[:n_video]:
        cmd += ["-i", p]
    for t in tiles:           # image inputs (stats card) need looping for the full duration
        if t["kind"] == "image":
            cmd += ["-loop", "1", "-t", f"{montage_seconds:.3f}", "-i", inputs[t["input_index"]]]
    cmd += ["-filter_complex", fg, "-map", "[out]", "-r", "30", "-t", f"{montage_seconds:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-movflags", "+faststart", os.path.abspath(out_path)]

    log(f"rendering montage ({n_video} clips + stats card) -> {out_path}")
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ERROR: ffmpeg montage failed (exit {r.returncode}):\n{r.stderr.strip()[:1500]}")
        return None
    return out_path


def render_clip(ep, out_path, montage_seconds, fit, work):
    """Render one riser's standalone labeled + speed-normalized clip."""
    if not (have("ffmpeg") and ep.get("video")):
        return None
    font_rel = "font.ttf"   # already copied by render_montage into a sibling work dir; ensure here too
    font_src = next((p for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf") if os.path.exists(p)), None)
    if font_src and not os.path.exists(os.path.join(work, font_rel)):
        shutil.copyfile(font_src, os.path.join(work, font_rel))
    d = ep["video_dur_s"]
    m = (montage_seconds / d) if d else 1.0
    pad = max(0.0, montage_seconds - (d * m if d else 0.0))
    h = ep.get("height")
    cap_name = f"clipcap_{ep['run_leaf']}.txt"
    l1 = (f"{h:.3f} m   {ep['label_short']}" if h is not None else ep["label_short"])
    l2 = f"{ep['verdict_short']}   |   {fmt_time(ep.get('real_time_s'))}"
    _write_caption(os.path.join(work, cap_name), l1, l2)
    tile = {"kind": "video", "input_index": 0, "x": 0, "y": 0, "m": m, "pad": pad,
            "fit": fit, "caption": cap_name, "best": False, "font": font_rel}
    fg = build_filtergraph([tile], montage_seconds, cell_w=1280, cell_h=720,
                           canvas_w=1280, canvas_h=720)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", os.path.abspath(ep["video"]),
           "-filter_complex", fg, "-map", "[out]", "-r", "30", "-t", f"{montage_seconds:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-movflags", "+faststart", os.path.abspath(out_path)]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ERROR: clip {out_path} failed: {r.stderr.strip()[:600]}")
        return None
    return out_path


# ---------------------------------------------------------------------------
# data section
# ---------------------------------------------------------------------------
def _git(args):
    try:
        return subprocess.run(["git", "-C", REPO_ROOT] + args, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def write_summary(rows, meta, out_dir):
    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    json_path = os.path.join(out_dir, "sweep_summary.json")
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "episodes": rows}, f, indent=2)
    return csv_path, json_path


def print_table(rows):
    log("data section (per riser):")
    print(f"  {'riser':>8}  {'code':<14}  {'verdict':<11}  {'steps':>6}  {'time':>7}  waypoint")
    for r in rows:
        riser = f"{r['riser_m']:.3f}" if r.get("riser_m") is not None else "-"
        steps = f"{r['steps_climbed']:.1f}" if r.get("steps_climbed") is not None else "-"
        t = r.get("real_time_s") or r.get("climb_time_s")
        print(f"  {riser:>8}  {r['label_short']:<14}  {r['verdict_short']:<11}  {steps:>6}  "
              f"{fmt_time(t):>7}  {r.get('waypoint', '')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build a presentation pack from a stair-height sweep.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", help="sweep manifest.json (episodes: [{height,label,run_dir,waypoint}])")
    src.add_argument("--run-dirs", nargs="+", help="explicit run_sim_* dirs (with --heights)")
    src.add_argument("--auto", type=int, metavar="N", help="use the newest N run_sim_* dirs")
    ap.add_argument("--heights", nargs="+", type=float, help="riser heights matching --run-dirs")
    ap.add_argument("--out", help="output folder (default: <manifest dir>/presentation or log/sweep_presentation)")
    ap.add_argument("--montage-seconds", type=float, default=20.0, help="target length all clips finish at")
    ap.add_argument("--montage-mode", choices=("sync", "uniform"), default="sync",
                    help="sync: each clip finishes at target; uniform: one factor, relative speed preserved")
    ap.add_argument("--hero", choices=("best", "highest_clean", "steepest"), default="best")
    ap.add_argument("--fit", choices=("cover", "letterbox"), default="cover")
    ap.add_argument("--no-montage", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args(argv)
    if not (args.manifest or args.run_dirs or args.auto):
        args.auto = 5
    return args


def default_out(args):
    if args.out:
        return args.out
    if args.manifest:
        return os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "presentation")
    return os.path.join(LOG_DIR, "sweep_presentation")


def run(args):
    discovered = discover_episodes(args)
    if not discovered:
        log("no episodes found; nothing to do")
        return 1
    eps = [build_episode(e) for e in discovered]
    # order by riser ascending (None last) for stable grid + charts
    eps.sort(key=lambda e: (e.get("height") is None, e.get("height") or 0))

    best_idx = pick_hero(eps, args.hero)
    for i, e in enumerate(eps):
        e["_best"] = (i == best_idx)

    rows = [summary_row(e) for e in eps]

    out_dir = default_out(args)
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "montage_seconds": args.montage_seconds,
        "montage_mode": args.montage_mode,
        "hero_mode": args.hero,
        "best_riser_m": eps[best_idx]["height"] if best_idx is not None else None,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git(["rev-parse", "--short", "HEAD"]),
        "n_episodes": len(eps),
    }

    csv_path, json_path = write_summary(rows, meta, out_dir)
    print_table(rows)
    log(f"wrote {csv_path}")
    log(f"wrote {json_path}")

    produced = {"summary_csv": csv_path, "summary_json": json_path}

    if not args.no_graphs:
        graphs = generate_graphs(eps, rows, os.path.join(out_dir, "graphs"))
        produced["graphs"] = graphs
        log(f"wrote {len(graphs)} graph(s) to {os.path.join(out_dir, 'graphs')}")

    stats_card = generate_stats_card(rows, meta, os.path.join(out_dir, "stats_card.png"), best_idx)
    if stats_card:
        produced["stats_card"] = stats_card
        log(f"wrote {stats_card}")

    if not args.no_montage:
        montage = render_montage(eps, stats_card,
                                 os.path.join(out_dir, "stair_sweep_montage.mp4"),
                                 args.montage_seconds, args.montage_mode, args.fit)
        if montage:
            produced["montage"] = montage
            log(f"wrote {montage}")

    if not args.no_clips and have("ffmpeg"):
        clips_dir = os.path.join(out_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        work = os.path.join(out_dir, "_montage")
        os.makedirs(work, exist_ok=True)
        made = []
        for e in eps:
            if not e.get("video"):
                continue
            tag = f"{e['height']:.3f}".replace(".", "p") if e.get("height") is not None else e["run_leaf"]
            p = render_clip(e, os.path.join(clips_dir, f"clip_{tag}.mp4"),
                            args.montage_seconds, args.fit, work)
            if p:
                made.append(p)
        produced["clips"] = made
        log(f"wrote {len(made)} standalone clip(s) to {clips_dir}")

    shutil.rmtree(os.path.join(out_dir, "_montage"), ignore_errors=True)  # transient ffmpeg workdir
    log("=" * 60)
    log(f"PRESENTATION PACK READY: {out_dir}")
    for k, v in produced.items():
        if isinstance(v, list):
            log(f"  {k}: {len(v)} file(s)")
        else:
            log(f"  {k}: {os.path.basename(v)}")
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
