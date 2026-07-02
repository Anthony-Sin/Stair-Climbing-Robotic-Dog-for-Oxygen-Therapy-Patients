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

--------------------------------------------------------------------------------
This module is now a FACADE. The implementation was split into single-responsibility
sibling modules (behavior-preserving refactor); every previously-public name is re-exported
here so existing importers (notably follow_present.py) keep working unchanged:

    sweep_constants  layout/geometry constants + preset defaults (+ analyze_climb thresholds)
    sweep_helpers    logging, ffprobe wrappers, label/verdict formatting, scene-event scan
    sweep_episodes   episode discovery + aggregation (summary rows, hero selection)
    sweep_graphs     matplotlib charts + leaderboard stats card
    sweep_montage    pure filtergraph builder + ffmpeg montage/clip orchestration
    sweep_data       git ref probe + CSV/JSON summary writer + console table
"""
import argparse
import os
import shutil
import sys

# Sibling import (submodules live next to this file) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_climb  # noqa: E402,F401  (re-exported: importers rely on sweep_present.analyze_climb)

from sweep_constants import (  # noqa: E402
    STAIR_BASE_X, STEP_RUN, FALL_TILT_DEG, COLLAPSE_H_M,
    REPO_ROOT, LOG_DIR,
    DEFAULT_STEP_DEPTH_M, DEFAULT_STEP_COUNT, DEFAULT_TOP_EDGE_X,
    VIDEO_PREFERENCE, RISER_SHORT, VERDICT_COLORS,
    CANVAS_W, CANVAS_H, COLS, ROWS, CELL_W, CELL_H, CAPTION_BAR_H,
)
from sweep_helpers import (  # noqa: E402
    log, have, riser_short, verdict_short, fmt_time,
    _write_caption, _ffprobe, ffprobe_duration, ffprobe_nframes,
    pick_video, scan_scene_events,
)
from sweep_episodes import (  # noqa: E402
    _resolve_run_dir, discover_episodes, build_episode, summary_row, pick_hero,
)
from sweep_graphs import (  # noqa: E402
    _setup_dark_theme, _assign_colors, _plot_climb_profile, _verdict_bar_colors, _xt,
    _plot_steps, _plot_stability, _plot_reach, generate_graphs, generate_stats_card,
)
from sweep_montage import (  # noqa: E402
    build_filtergraph, _compute_speeds, render_montage, render_clip,
)
from sweep_data import (  # noqa: E402
    _git, write_summary, print_table,
)

__all__ = [
    # analyze_climb-derived thresholds (shared source of truth)
    "STAIR_BASE_X", "STEP_RUN", "FALL_TILT_DEG", "COLLAPSE_H_M",
    # constants
    "REPO_ROOT", "LOG_DIR",
    "DEFAULT_STEP_DEPTH_M", "DEFAULT_STEP_COUNT", "DEFAULT_TOP_EDGE_X",
    "VIDEO_PREFERENCE", "RISER_SHORT", "VERDICT_COLORS",
    "CANVAS_W", "CANVAS_H", "COLS", "ROWS", "CELL_W", "CELL_H", "CAPTION_BAR_H",
    # small helpers
    "log", "have", "riser_short", "verdict_short", "fmt_time",
    "_write_caption", "_ffprobe", "ffprobe_duration", "ffprobe_nframes",
    "pick_video", "scan_scene_events",
    # episode discovery + aggregation
    "_resolve_run_dir", "discover_episodes", "build_episode", "summary_row", "pick_hero",
    # graphs
    "_setup_dark_theme", "_assign_colors", "_plot_climb_profile", "_verdict_bar_colors", "_xt",
    "_plot_steps", "_plot_stability", "_plot_reach", "generate_graphs", "generate_stats_card",
    # montage
    "build_filtergraph", "_compute_speeds", "render_montage", "render_clip",
    # data section
    "_git", "write_summary", "print_table",
    # main / orchestration
    "parse_args", "default_out", "run", "main",
]


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
