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
import glob  # noqa: F401  (preserved on the facade; used by follow_present_episodes.discover_episodes)
import json
import os
import shutil
import sys

# Ensure this file's directory is on the path so sweep_present imports cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_present  # noqa: E402,F401
from sweep_present import (  # noqa: E402,F401
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
import analyze_climb  # noqa: E402,F401
from analyze_climb import STAIR_BASE_X  # noqa: E402,F401

# ---------------------------------------------------------------------------
# FACADE re-exports: the implementation was split into single-responsibility
# ``follow_present_*`` sibling modules (behavior-preserving refactor); every
# previously-public name is re-exported here so existing importers / script-mode
# ``follow_present.<name>`` access keep working unchanged.
#
#   follow_present_scan      follow-mode reach colors + reach_short + scan_follow_events
#   follow_present_episodes  episode discovery + aggregation (follow-mode variant)
#   follow_present_graphs    follow-mode charts + leaderboard stats card
#   follow_present_data      follow-mode console leaderboard table
# ---------------------------------------------------------------------------
from follow_present_scan import (  # noqa: E402,F401
    REACH_COLORS, reach_short, scan_follow_events,
)
from follow_present_episodes import (  # noqa: E402,F401
    _resolve_run_dir, discover_episodes, build_episode, summary_row, pick_hero,
)
from follow_present_graphs import (  # noqa: E402,F401
    _plot_approach_time, generate_graphs, generate_stats_card,
)
from follow_present_data import (  # noqa: E402,F401
    _print_table_follow,
)


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
