"""Build a slide-ready presentation pack from a person-follow stair-height sweep.

`run_follow_sweep.ps1` drives the FULL person-follow pipeline (Docker YOLO controller +
PGTT walk policy + blind-RL climb + O2 payload) across several riser heights and leaves each
episode in its own ``log/run_sim_<stamp>/``. This tool turns that scatter into one folder you
can drop straight into slides:

  <out>/
    sweep_summary.csv / sweep_summary.json     the data section (one row per riser)
    graphs/g1_climb_profile.png ... g6_dashboard.png
    stats_card.png                             leaderboard card (also montage cell 6)
    follow_sweep_montage.mp4                   2x3 grid: 5 riser videos + stats card,
                                               speed-normalized so they all finish together
    clips/clip_<riser>.mp4                     each riser standalone (labeled + speed-normalized)

It reads ONLY recorded artifacts, so it is runnable standalone on existing logs (no Isaac):

    python follow_present.py --manifest log/follow_sweep_<stamp>/manifest.json
    python follow_present.py --auto 5
    python follow_present.py --run-dirs <d1> <d2> ... --heights 0.1 0.125 0.15

Unlike the stair-waypoint sweep, follow mode judges runs on person-follow and stair-approach
success (reach_status: REACHED TOP / DID NOT REACH / FELL) rather than a deterministic
waypoint gate. The analyze_climb VERDICT is also recorded for the climb leg itself.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

# Ensure this file's directory is on the path so sweep_present imports cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_present  # noqa: E402
from sweep_present import (  # noqa: E402
    # constants
    LOG_DIR, DEFAULT_STEP_DEPTH_M, DEFAULT_STEP_COUNT, DEFAULT_TOP_EDGE_X,
    VIDEO_PREFERENCE, RISER_SHORT, CANVAS_W, CANVAS_H, COLS, ROWS, CELL_W, CELL_H, CAPTION_BAR_H,
    # small helpers
    log, have, riser_short, verdict_short, fmt_time, pick_video,
    ffprobe_duration, ffprobe_nframes, _write_caption, _ffprobe,
    # heavy helpers (shared; not overridden below)
    _assign_colors, _plot_climb_profile, _plot_steps, _plot_stability, _plot_reach,
    build_filtergraph, _compute_speeds, render_montage, render_clip,
    _git, write_summary, print_table,
)
import analyze_climb  # noqa: E402
from analyze_climb import STAIR_BASE_X, STEP_RUN, FALL_TILT_DEG, COLLAPSE_H_M  # noqa: E402

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


# ---------------------------------------------------------------------------
# episode discovery + aggregation (follow-mode variant)
# ---------------------------------------------------------------------------
def _resolve_run_dir(path):
    if os.path.isabs(path) and os.path.isdir(path):
        return path
    cand = os.path.join(LOG_DIR, path)
    return cand if os.path.isdir(cand) else path


def discover_episodes(args):
    """Return a list of follow-mode episode dicts from the chosen source."""
    eps = []
    if args.manifest:
        with open(args.manifest, encoding="utf-8-sig") as f:
            man = json.load(f)
        eps_raw = man.get("episodes", man) if isinstance(man, dict) else man
        if isinstance(eps_raw, dict):
            eps_raw = [eps_raw]
        for e in eps_raw:
            rd = _resolve_run_dir(str(e.get("run_dir", "")))
            eps.append({
                "height": e.get("height"), "label": e.get("label"),
                "run_dir": rd,
                "follow_reach": e.get("follow_reach"),  # from PS1 manifest
            })
    elif args.run_dirs:
        heights = args.heights or [None] * len(args.run_dirs)
        for rd, h in zip(args.run_dirs, heights):
            eps.append({"height": h, "label": None, "run_dir": _resolve_run_dir(rd), "follow_reach": None})
    else:
        cands = sorted(glob.glob(os.path.join(LOG_DIR, "run_sim_*")), key=os.path.getmtime, reverse=True)
        for rd in cands[: args.auto]:
            eps.append({"height": None, "label": None, "run_dir": rd, "follow_reach": None})
    return eps


def build_episode(ep):
    """Enrich a discovered follow-mode episode with parsed metrics and hero clip."""
    rd = ep["run_dir"]
    jsonl = os.path.join(rd, "debug", "isaac_env.jsonl")
    result = analyze_climb.analyze_run(rd) if os.path.isdir(rd) else {"rows": [], "stats": None}
    scene = scan_follow_events(jsonl)
    stats = result.get("stats")
    rows = result.get("rows", [])

    height = ep.get("height")
    if height is None:
        height = scene.get("step_height_m")
    video, vdur = pick_video(rd) if os.path.isdir(rd) else (None, None)

    # Prefer the PS1 manifest's follow_reach (already parsed from physics) if available;
    # fall back to the JSONL scan.
    follow_reach_raw = ep.get("follow_reach") or scene.get("reach_status") or "unknown"
    reach = reach_short(follow_reach_raw)
    verdict = stats["verdict"] if stats else "NO DATA (no fall_diag rows)"
    real_time = scene.get("approach_time_s")
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
        "follow_reach": reach,
        "approach_time_s": scene.get("approach_time_s"),
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
        "follow_reach": ep["follow_reach"],
        "verdict": ep["verdict"],
        "verdict_short": ep["verdict_short"],
        "steps_climbed": round(s["steps_climbed"], 2) if s.get("steps_climbed") is not None else None,
        "max_x_m": round(s["max_x"], 3) if s.get("max_x") is not None else None,
        "top_edge_x_m": DEFAULT_TOP_EDGE_X,
        "max_tilt_deg": round(s["max_tilt"], 1) if s.get("max_tilt") is not None else None,
        "min_h_on_m": round(s["min_h_on"], 3) if s.get("min_h_on") is not None else None,
        "approach_time_s": round(ep["approach_time_s"], 1) if isinstance(ep.get("approach_time_s"), (int, float)) else None,
        "climb_time_s": round(s["climb_time_s"], 1) if s.get("climb_time_s") is not None else None,
        "real_time_s": round(ep["real_time_s"], 1) if isinstance(ep.get("real_time_s"), (int, float)) else None,
        "step_depth_m": ep["step_depth_m"],
        "step_count": ep["step_count"],
        "top_height_m": ep.get("top_height_m"),
        "video": ep["video"] or "",
        "video_dur_s": round(ep["video_dur_s"], 2) if ep.get("video_dur_s") else None,
        "video_frames": ep.get("video_frames"),
        "run_dir": ep["run_dir"],
    }


def pick_hero(eps):
    """Index of the episode to badge as BEST for follow mode (REACHED TOP + most steps)."""
    scored = [(i, e) for i, e in enumerate(eps) if e.get("stats")]
    if not scored:
        return None
    # Prefer REACHED TOP, then most steps, then steepest
    rank = {"REACHED TOP": 3, "DID NOT REACH": 1, "FELL": 0, "INCOMPLETE": 1, "NO DATA": -1}

    def key(ie):
        e = ie[1]
        return (
            rank.get(e["follow_reach"], 0),
            e["stats"].get("steps_climbed") or 0,
            e.get("height") or 0,
        )
    return max(scored, key=key)[0]


# ---------------------------------------------------------------------------
# graphs (follow-mode variants)
# ---------------------------------------------------------------------------
def _plot_approach_time(ax, eps):
    """Bar chart: sim-time (s) for the robot to reach the stair base (x >= STAIR_BASE_X)."""
    valid = [(e, e["approach_time_s"]) for e in eps if isinstance(e.get("approach_time_s"), (int, float))]
    if not valid:
        ax.set_title("Approach time to stair base (no data)")
        return
    xs = range(len(valid))
    vals = [at for _, at in valid]
    colors = [e.get("color", (0.2, 0.4, 0.8, 1.0)) for e, _ in valid]
    ax.bar(xs, vals, color=colors)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, f"{v:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{e['height']:.3f}\n{e['label_short']}" if e.get("height") is not None else e["label_short"]
         for e, _ in valid],
        fontsize=8)
    ax.set_ylabel("sim-time to reach stair base (s)")
    ax.set_title(f"Approach time (robot reaches x ≥ {STAIR_BASE_X:.1f} m)")
    ax.axhline(30, ls=":", color="0.65", lw=1.0)
    ax.text(len(valid) - 0.5, 31, "30 s guideline", ha="right", color="0.6", fontsize=8)


def generate_graphs(eps, rows, graphs_dir):
    """Write per-metric PNGs + a 2x3 dashboard. Returns the list of written paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable ({exc}); skipping graphs")
        return []
    sweep_present._setup_dark_theme(plt)
    os.makedirs(graphs_dir, exist_ok=True)
    _assign_colors(eps)
    written = []

    def _save(fig, name):
        p = os.path.join(graphs_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight", transparent=True)
        plt.close(fig)
        written.append(p)

    singles = [
        ("g1_climb_profile.png",  lambda ax: _plot_climb_profile(ax, eps)),
        ("g2_steps_vs_riser.png", lambda ax: _plot_steps(ax, rows)),
        ("g3_stability_vs_riser.png", lambda ax: _plot_stability(ax, rows)),
        ("g4_reach_vs_riser.png", lambda ax: _plot_reach(ax, rows)),
        ("g5_approach_time.png",  lambda ax: _plot_approach_time(ax, eps)),
    ]
    for name, fn in singles:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        try:
            fn(ax)
            _save(fig, name)
        except Exception as exc:
            log(f"WARNING: chart {name} failed: {exc}")
            plt.close(fig)

    # dashboard 2x3
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 8.4))
        _plot_climb_profile(axes[0][0], eps)
        _plot_steps(axes[0][1], rows)
        _plot_approach_time(axes[0][2], eps)
        _plot_stability(axes[1][0], rows)
        _plot_reach(axes[1][1], rows)
        axes[1][2].axis("off")  # empty cell (reserved for future metric)
        fig.suptitle("Person-Follow Sweep — PGTT walk · blind-RL climb · YOLO · +O₂ payload",
                     fontsize=14, fontweight="bold", color="#e0e0e0")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, "g6_dashboard.png")
    except Exception as exc:
        log(f"WARNING: dashboard failed: {exc}")
    return written


# ---------------------------------------------------------------------------
# stats card (follow-mode: PERSON-FOLLOW SWEEP title + REACH STATUS column)
# ---------------------------------------------------------------------------
def generate_stats_card(rows, meta, out_path, best_idx=None):
    """Render the leaderboard stats card branded for person-follow sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable ({exc}); skipping stats card")
        return None

    BG      = "#0d1117"
    HEADER  = "#161b22"
    ROW_ALT = "#0d1117"
    ROW_EVN = "#101419"
    GOLD    = "#f0c040"
    TEXT    = "#e6edf3"
    MUTED   = "#8b949e"
    ACCENT  = "#58a6ff"

    n = len(rows)
    card_h = 3.4 + n * 0.58
    fig = plt.figure(figsize=(12.8, card_h), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # title strip
    title_ax = fig.add_axes([0.0, 1 - 0.52 / card_h, 1.0, 0.52 / card_h])
    title_ax.set_facecolor(HEADER)
    title_ax.axis("off")
    title_ax.text(0.022, 0.72, "PERSON-FOLLOW SWEEP", color=TEXT,
                  fontsize=17, fontweight="bold", va="center",
                  transform=title_ax.transAxes)
    title_ax.text(0.022, 0.22, "PGTT walk  ·  blind-RL climb  ·  YOLO person-follow  ·  +O₂ payload",
                  color=MUTED, fontsize=10, va="center",
                  transform=title_ax.transAxes)
    branch = meta.get("git_branch", "")
    commit = meta.get("git_commit", "")
    ref = f"{branch}@{commit}" if branch and commit else (branch or commit)
    if ref:
        title_ax.text(0.978, 0.5, ref, color=MUTED, fontsize=9, va="center",
                      ha="right", transform=title_ax.transAxes, fontfamily="monospace")

    # column headers
    COLS_DEF = [
        ("RISER",       0.060, "left"),
        ("CODE",        0.175, "left"),
        ("REACH STATUS",0.360, "left"),
        ("STEPS",       0.590, "center"),
        ("PROGRESS",    0.680, "left"),
        ("TIME",        0.930, "right"),
    ]
    hdr_top = 1 - 0.52 / card_h
    hdr_h   = 0.38 / card_h
    hdr_ax  = fig.add_axes([0.0, hdr_top - hdr_h, 1.0, hdr_h])
    hdr_ax.set_facecolor("#1c2128")
    hdr_ax.axis("off")
    for lbl, xf, ha in COLS_DEF:
        hdr_ax.text(xf, 0.5, lbl, color=ACCENT, fontsize=8.5, fontweight="bold",
                    va="center", ha=ha, transform=hdr_ax.transAxes)

    # rows
    row_top = hdr_top - hdr_h
    row_h   = (row_top - 0.28 / card_h) / max(1, n)
    step_count = (rows[0].get("step_count") or DEFAULT_STEP_COUNT) if rows else DEFAULT_STEP_COUNT

    for i, r in enumerate(rows):
        is_best = (best_idx is not None and i == best_idx)
        row_bg  = ROW_EVN if i % 2 == 0 else ROW_ALT
        y0 = row_top - (i + 1) * row_h

        row_ax = fig.add_axes([0.0, y0, 1.0, row_h])
        row_ax.set_facecolor(row_bg)
        row_ax.axis("off")

        if is_best:
            row_ax.add_patch(FancyBboxPatch((0, 0), 0.004, 1.0, boxstyle="square,pad=0",
                                            facecolor=GOLD, edgecolor="none",
                                            transform=row_ax.transAxes, clip_on=False, zorder=5))

        reach    = r.get("follow_reach", "NO DATA")
        rc_hex   = REACH_COLORS.get(reach, "#9e9e9e")
        riser    = f"{r['riser_m']:.3f} m" if r.get("riser_m") is not None else "-"
        steps    = r.get("steps_climbed") or 0
        steps_s  = f"{steps:.1f}" if r.get("steps_climbed") is not None else "-"
        t        = r.get("approach_time_s") or r.get("real_time_s") or r.get("climb_time_s")
        t_s      = fmt_time(t)

        # riser column
        mark_col = GOLD if is_best else TEXT
        row_ax.text(0.060, 0.5, ("▶ " if is_best else "   ") + riser,
                    color=mark_col, fontsize=11,
                    fontweight="bold" if is_best else "normal",
                    va="center", ha="left", transform=row_ax.transAxes)

        # code column
        row_ax.text(0.175, 0.5, r["label_short"], color=MUTED,
                    fontsize=9, va="center", ha="left", transform=row_ax.transAxes)

        # reach status pill
        pill_x, pill_y, pill_w, pill_h = 0.358, 0.18, 0.21, 0.64
        row_ax.add_patch(FancyBboxPatch((pill_x, pill_y), pill_w, pill_h,
                                        boxstyle="round,pad=0.01",
                                        facecolor=rc_hex, edgecolor="none",
                                        transform=row_ax.transAxes, clip_on=True))
        # Abbreviate long labels inside the pill
        pill_label = {
            "REACHED TOP":   "REACHED TOP",
            "DID NOT REACH": "NOT REACHED",
            "FELL":          "FELL",
            "INCOMPLETE":    "INCOMPLETE",
            "NO DATA":       "NO DATA",
        }.get(reach, reach[:11])
        row_ax.text(pill_x + pill_w / 2, 0.5, pill_label,
                    color="white", fontsize=9.0, fontweight="bold",
                    va="center", ha="center", transform=row_ax.transAxes)

        # steps number
        row_ax.text(0.590, 0.5, steps_s, color=TEXT,
                    fontsize=11, va="center", ha="center",
                    transform=row_ax.transAxes)

        # progress bar
        bar_x, bar_w_max = 0.640, 0.26
        bar_h_f = 0.28
        bar_y = (1 - bar_h_f) / 2
        row_ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w_max, bar_h_f,
                                        boxstyle="round,pad=0.005",
                                        facecolor="#30363d", edgecolor="none",
                                        transform=row_ax.transAxes, clip_on=True))
        frac = min(1.0, steps / step_count) if step_count else 0
        if frac > 0.01:
            row_ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w_max * frac, bar_h_f,
                                            boxstyle="round,pad=0.005",
                                            facecolor=rc_hex, edgecolor="none",
                                            transform=row_ax.transAxes, clip_on=True))
        row_ax.text(bar_x + bar_w_max + 0.012, 0.5, f"/{step_count}",
                    color=MUTED, fontsize=8, va="center", ha="left",
                    transform=row_ax.transAxes)

        # time column
        row_ax.text(0.965, 0.5, t_s, color=MUTED,
                    fontsize=10, va="center", ha="right",
                    transform=row_ax.transAxes)

    # footer
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
    n_reach = sum(1 for r in rows if r.get("follow_reach") == "REACHED TOP")
    ftr_ax.text(0.978, 0.5, f"{n_reach}/{n} full climbs",
                color=GOLD if n_reach > 0 else MUTED, fontsize=9.5, fontweight="bold",
                va="center", ha="right", transform=ftr_ax.transAxes)

    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# data section helpers (follow-mode print_table override)
# ---------------------------------------------------------------------------
def _print_table_follow(rows):
    log("data section (per riser):")
    print(f"  {'riser':>8}  {'code':<14}  {'reach':<14}  {'steps':>6}  {'approach':>9}  climb_verdict")
    for r in rows:
        riser = f"{r['riser_m']:.3f}" if r.get("riser_m") is not None else "-"
        steps = f"{r['steps_climbed']:.1f}" if r.get("steps_climbed") is not None else "-"
        appr  = fmt_time(r.get("approach_time_s"))
        print(f"  {riser:>8}  {r['label_short']:<14}  {r.get('follow_reach','?'):<14}  "
              f"{steps:>6}  {appr:>9}  {r.get('verdict_short','')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build a presentation pack from a person-follow stair-height sweep.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", help="follow sweep manifest.json")
    src.add_argument("--run-dirs", nargs="+", help="explicit run_sim_* dirs (with --heights)")
    src.add_argument("--auto", type=int, metavar="N", help="use the newest N run_sim_* dirs")
    ap.add_argument("--heights", nargs="+", type=float)
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--montage-seconds", type=float, default=20.0)
    ap.add_argument("--montage-mode", choices=("sync", "uniform"), default="sync")
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
    return os.path.join(LOG_DIR, "follow_presentation")


def run(args):
    discovered = discover_episodes(args)
    if not discovered:
        log("no episodes found; nothing to do")
        return 1
    eps = [build_episode(e) for e in discovered]
    eps.sort(key=lambda e: (e.get("height") is None, e.get("height") or 0))

    best_idx = pick_hero(eps)
    for i, e in enumerate(eps):
        e["_best"] = (i == best_idx)

    rows = [summary_row(e) for e in eps]

    out_dir = default_out(args)
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "montage_seconds": args.montage_seconds,
        "montage_mode":    args.montage_mode,
        "best_riser_m":    eps[best_idx]["height"] if best_idx is not None else None,
        "git_branch":      _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit":      _git(["rev-parse", "--short", "HEAD"]),
        "n_episodes":      len(eps),
    }

    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    json_path = os.path.join(out_dir, "sweep_summary.json")
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "episodes": rows}, f, indent=2)
    _print_table_follow(rows)
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
                                 os.path.join(out_dir, "follow_sweep_montage.mp4"),
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

    viewer = generate_html_viewer(eps, rows, os.path.join(out_dir, "viewer.html"))
    if viewer:
        produced["viewer"] = viewer
        log(f"wrote {viewer}")

    shutil.rmtree(os.path.join(out_dir, "_montage"), ignore_errors=True)
    log("=" * 60)
    log(f"PRESENTATION PACK READY: {out_dir}")
    for k, v in produced.items():
        if isinstance(v, list):
            log(f"  {k}: {len(v)} file(s)")
        else:
            log(f"  {k}: {os.path.basename(v)}")
    return 0


def generate_html_viewer(eps, rows, out_path):
    """Generate a self-contained HTML page showing all episode videos in one grid."""
    items = []
    for ep, row in zip(eps, rows):
        video = ep.get("video")
        if not video:
            continue
        rel = os.path.relpath(video, os.path.dirname(out_path)).replace("\\", "/")
        h = ep.get("height")
        label = f"{h:.3f} m  {ep['label_short']}" if h is not None else ep["label_short"]
        reach = ep.get("follow_reach", "NO DATA")
        verdict = ep.get("verdict_short", "")
        steps = row.get("steps_climbed")
        steps_s = f"{steps:.1f} / {row.get('step_count', 14)} steps" if steps is not None else ""
        reach_color = {
            "REACHED TOP": "#2e7d32", "DID NOT REACH": "#546e7a",
            "FELL": "#c62828", "INCOMPLETE": "#9e9e9e", "NO DATA": "#bdbdbd",
        }.get(reach, "#bdbdbd")
        items.append((rel, label, reach, reach_color, verdict, steps_s))

    cards = ""
    for rel, label, reach, reach_color, verdict, steps_s in items:
        cards += f"""
  <div class="card">
    <div class="label">{label}</div>
    <video controls preload="metadata">
      <source src="{rel}" type="video/mp4">
    </video>
    <div class="meta">
      <span class="pill" style="background:{reach_color}">{reach}</span>
      <span class="verdict">{verdict}</span>
      <span class="steps">{steps_s}</span>
    </div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Person-Follow Sweep</title>
<style>
  body {{ background:#0d1117; color:#e6edf3; font-family:sans-serif; margin:0; padding:16px; }}
  h1 {{ font-size:1.4em; color:#58a6ff; margin:0 0 4px; }}
  .subtitle {{ color:#8b949e; font-size:.9em; margin:0 0 16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(400px,1fr)); gap:16px; }}
  .card {{ background:#161b22; border-radius:8px; overflow:hidden; border:1px solid #30363d; }}
  .label {{ padding:8px 12px; font-weight:bold; font-size:.95em; border-bottom:1px solid #30363d; }}
  video {{ width:100%; display:block; background:#000; max-height:280px; }}
  .meta {{ padding:8px 12px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  .pill {{ padding:3px 10px; border-radius:12px; font-size:.8em; font-weight:bold; color:#fff; }}
  .verdict {{ color:#8b949e; font-size:.85em; }}
  .steps {{ color:#8b949e; font-size:.85em; margin-left:auto; }}
</style>
</head>
<body>
<h1>Person-Follow Sweep</h1>
<p class="subtitle">PGTT walk &middot; blind-RL climb &middot; YOLO person-follow &middot; +O&sup2; payload</p>
<div class="grid">{cards}
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
