"""Individual performance-chart generators + the generate_all orchestrator.

Split out of charts.py (Phase 2 structural refactor). Each chart_* function
renders one PNG from the loaded run rows using the shared theme/util helpers in
chart_theme.py; matplotlib is imported lazily inside each function so importing
this module never requires matplotlib.
"""

from pathlib import Path
from typing import List, Dict

from chart_theme import (
    _load_rows,
    _outcome_color,
    _safe_float,
    _branch_colors,
    _setup_dark_theme,
)


# ---------------------------------------------------------------------------
# Chart 1: Progress over time (improved — git SHA annotation, branch line color)
# ---------------------------------------------------------------------------

def chart_progress_over_time(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _setup_dark_theme(plt)

    if not rows:
        return

    branch_colors = _branch_colors(rows)
    xs = list(range(len(rows)))
    ys = [_safe_float(r.get("max_x_m")) or 0.0 for r in rows]
    dot_colors = [_outcome_color(str(r.get("outcome", ""))) for r in rows]
    labels = [str(r.get("run_id", ""))[-9:] for r in rows]

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.9 + 2), 4.5))

    # Draw line segments colored by branch
    for i in range(len(rows) - 1):
        b = str(rows[i].get("git_branch", ""))
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color=branch_colors.get(b, "#bdc3c7"), linewidth=1.5, zorder=1)

    ax.scatter(xs, ys, c=dot_colors, s=90, zorder=2, edgecolors="#888", linewidths=0.4)

    for i, r in enumerate(rows):
        sha = str(r.get("git_commit_sha") or "")[:7]
        if sha:
            ax.annotate(sha, (xs[i], ys[i]), textcoords="offset points",
                        xytext=(0, -13), fontsize=6, ha="center", color="#aaaaaa",
                        fontfamily="monospace")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("max_x_m (m)")
    ax.set_title("Furthest forward reach per run  (dot = outcome, line = branch, label = commit SHA)")
    ax.grid(axis="y", alpha=0.3)

    outcome_legend = [
        mpatches.Patch(color=_outcome_color(o), label=o)
        for o in ["completed", "robot_fell", "timeout", "unknown"]
    ]
    branch_legend = [
        mpatches.Patch(color=c, label=b)
        for b, c in branch_colors.items()
    ]
    ax.legend(handles=outcome_legend + branch_legend, fontsize=7, loc="upper left",
              ncol=2, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Outcomes bar (improved — adds percentage label)
# ---------------------------------------------------------------------------

def chart_outcomes_bar(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from collections import Counter
    _setup_dark_theme(plt)

    if not rows:
        return

    counts = Counter(str(r.get("outcome", "unknown")) for r in rows)
    total = sum(counts.values())
    outcomes = sorted(counts.keys())
    vals = [counts[o] for o in outcomes]
    colors = [_outcome_color(o) for o in outcomes]

    fig, ax = plt.subplots(figsize=(5, max(2, len(outcomes) * 0.6 + 1)))
    bars = ax.barh(outcomes, vals, color=colors)
    for bar, val in zip(bars, vals):
        pct = val / total * 100
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlabel("Run count")
    ax.set_title(f"Outcome distribution  (n={total})")
    ax.set_xlim(0, max(vals) * 1.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Stability scatter (unchanged)
# ---------------------------------------------------------------------------

def chart_stability_scatter(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _setup_dark_theme(plt)

    if not rows:
        return

    xs, ys, sizes, colors = [], [], [], []
    for r in rows:
        x = _safe_float(r.get("max_x_m"))
        y = _safe_float(r.get("max_abs_pitch_deg"))
        s = _safe_float(r.get("max_action_norm")) or 5.0
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        sizes.append(max(20, min(300, s * 8)))
        colors.append(_outcome_color(str(r.get("outcome", ""))))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors="#888", linewidths=0.5)
    ax.set_xlabel("max_x_m (m) — how far the robot got")
    ax.set_ylabel("max pitch excursion (deg) — instability")
    ax.set_title("Stability vs Progress\n(dot size = max action norm; lower-right = better)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Config diff heatmap — what changed between runs?
# ---------------------------------------------------------------------------

def chart_config_diff_heatmap(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _setup_dark_theme(plt)

    if len(rows) < 2:
        return

    PARAMS = [
        ("trans_x_max",              "vx_max"),
        ("kp",                       "kp"),
        ("kd",                       "kd"),
        ("target_distance",          "tgt_dist"),
        ("sim_latency_ms",           "latency_ms"),
        ("trans_x_tolerance",        "vx_tol"),
        ("parkour_heading_mode",     "heading"),
        ("parkour_person_mask_enabled", "mask"),
        ("obstacle_stop_enabled",    "obs_stop"),
        ("stair_preset",             "stair_preset"),
    ]
    keys = [p[0] for p in PARAMS]
    col_labels = [p[1] for p in PARAMS]
    n_rows = len(rows)
    n_cols = len(PARAMS)

    cell_vals = []
    cell_changed = []
    for i, r in enumerate(rows):
        row_vals = []
        row_changed = []
        prev = rows[i - 1] if i > 0 else None
        for key in keys:
            v = r.get(key)
            row_vals.append("" if v is None else str(v))
            changed = (prev is not None) and (str(v) != str(prev.get(key)))
            row_changed.append(changed)
        cell_vals.append(row_vals)
        cell_changed.append(row_changed)

    fig_h = max(3, n_rows * 0.45 + 1.2)
    fig_w = max(6, n_cols * 1.1 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    for i in range(n_rows):
        for j in range(n_cols):
            bg = "#c47a00" if cell_changed[i][j] else "#1e2230"
            txt_color = "#fff" if cell_changed[i][j] else "#cccccc"
            rect = plt.Rectangle((j, n_rows - i - 1), 1, 1,
                                  facecolor=bg, edgecolor="#444444", linewidth=0.5)
            ax.add_patch(rect)
            val = cell_vals[i][j]
            if len(val) > 10:
                val = val[:9] + "…"
            ax.text(j + 0.5, n_rows - i - 0.5, val,
                    ha="center", va="center", fontsize=7,
                    color=txt_color, fontfamily="monospace")

    # Row labels (run ID short form)
    for i, r in enumerate(rows):
        run_id = str(r.get("run_id", ""))[-13:]
        outcome = str(r.get("outcome", ""))
        dot = {"completed": "●", "robot_fell": "✕", "timeout": "◐"}.get(outcome, "○")
        ax.text(-0.05, n_rows - i - 0.5,
                f"{dot} {run_id}", ha="right", va="center", fontsize=6.5,
                color=_outcome_color(outcome))

    # Column labels
    for j, label in enumerate(col_labels):
        ax.text(j + 0.5, n_rows + 0.1, label,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#e0e0e0")

    ax.set_title("Config per run  (amber = changed from prior run)", pad=14, fontsize=9)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#c47a00", label="changed"),
        mpatches.Patch(color="#1e2230", label="same / null"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), framealpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Stability timeseries — 4 stacked subplots over runs
# ---------------------------------------------------------------------------

def chart_stability_timeseries(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _setup_dark_theme(plt)

    if not rows:
        return

    xs = list(range(len(rows)))
    dot_colors = [_outcome_color(str(r.get("outcome", ""))) for r in rows]

    metrics = [
        ("max_abs_pitch_deg",  "Max pitch (deg)",   35.0),
        ("max_abs_roll_deg",   "Max roll (deg)",    None),
        ("mean_action_norm",   "Mean action norm",  None),
        ("mean_vx_cmd_mps",    "Mean vx cmd (m/s)", None),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(7, len(rows) * 0.9 + 2), 8),
                             sharex=True)

    run_labels = [str(r.get("run_id", ""))[-9:] for r in rows]

    for ax, (key, ylabel, threshold) in zip(axes, metrics):
        ys = [_safe_float(r.get(key)) for r in rows]
        valid = [(x, y, c) for x, y, c in zip(xs, ys, dot_colors) if y is not None]
        if valid:
            vx, vy, vc = zip(*valid)
            ax.plot(list(vx), list(vy), color="#888888", linewidth=1, zorder=1)
            ax.scatter(list(vx), list(vy), c=list(vc), s=60, zorder=2,
                       edgecolors="#888", linewidths=0.4)
        if threshold is not None:
            ax.axhline(threshold, color="#e74c3c", linewidth=0.8, linestyle="--",
                       alpha=0.6, label=f"threshold ({threshold}°)")
            ax.legend(fontsize=6, loc="upper right")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(run_labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Stability metrics over runs  (dot color = outcome)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: Branch performance — max_x_m distribution per branch
# ---------------------------------------------------------------------------

def chart_branch_performance(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_dark_theme(plt)

    if not rows:
        return

    from collections import defaultdict
    branch_data: Dict[str, List] = defaultdict(list)
    for r in rows:
        x = _safe_float(r.get("max_x_m"))
        if x is not None:
            branch_data[str(r.get("git_branch", "unknown"))].append(
                (x, _outcome_color(str(r.get("outcome", ""))))
            )

    if not branch_data:
        return

    branches = sorted(branch_data.keys())
    fig, ax = plt.subplots(figsize=(max(5, len(branches) * 1.8 + 1.5), 4))

    for i, branch in enumerate(branches):
        pts = branch_data[branch]
        ys = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        # Jitter x for readability
        jitter = (np.random.default_rng(seed=i).random(len(ys)) - 0.5) * 0.3
        ax.scatter(np.full(len(ys), i) + jitter, ys, c=cs, s=70,
                   edgecolors="#888", linewidths=0.4, zorder=2)
        # Box summary
        if len(ys) >= 3:
            ax.boxplot(ys, positions=[i], widths=0.4, patch_artist=False,
                       medianprops={"color": "#e0e0e0", "linewidth": 1.5},
                       whiskerprops={"linewidth": 0.8, "color": "#aaaaaa"},
                       capprops={"linewidth": 0.8, "color": "#aaaaaa"},
                       boxprops={"linewidth": 0.8, "color": "#aaaaaa"},
                       flierprops={"marker": ""}, zorder=1)
        ax.text(i, -0.15, f"n={len(ys)}", ha="center", va="top",
                fontsize=7, color="#aaaaaa", transform=ax.get_xaxis_transform())

    short_branches = [b[-22:] if len(b) > 22 else b for b in branches]
    ax.set_xticks(range(len(branches)))
    ax.set_xticklabels(short_branches, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("max_x_m (m)")
    ax.set_title("Performance by branch  (dot = outcome; box shown for n≥3)")
    ax.grid(axis="y", alpha=0.3)

    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color=_outcome_color(o), label=o)
              for o in ["completed", "robot_fell", "timeout", "unknown"]]
    ax.legend(handles=legend, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 7: Stair phase breakdown — approach + stair time per run
# ---------------------------------------------------------------------------

def chart_stair_phase_breakdown(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _setup_dark_theme(plt)

    valid = [r for r in rows
             if _safe_float(r.get("motion_elapsed_sec")) is not None
             and _safe_float(r.get("stair_phase_sec")) is not None]
    if not valid:
        return

    run_labels = [str(r.get("run_id", ""))[-9:] for r in valid]
    approach = [max(0.0, (_safe_float(r["motion_elapsed_sec"]) or 0)
                    - (_safe_float(r["stair_phase_sec"]) or 0)) for r in valid]
    stair = [_safe_float(r["stair_phase_sec"]) or 0.0 for r in valid]
    fell = [str(r.get("outcome", "")) == "robot_fell" for r in valid]

    ys = range(len(valid))
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(valid) * 0.45 + 1.2)))

    bars_a = ax.barh(list(ys), approach, color="#3498db", label="Approach phase", height=0.6)
    ax.barh(list(ys), stair, left=approach, color="#e67e22", label="Stair phase", height=0.6)

    for i, (bar, f) in enumerate(zip(bars_a, fell)):
        if f:
            total = approach[i] + stair[i]
            ax.barh(i, total, height=0.62, fill=False,
                    edgecolor="#e74c3c", linewidth=2.0)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(run_labels, fontsize=7)
    ax.set_xlabel("Sim seconds")
    ax.set_title("Time breakdown per run  (red border = robot fell)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 8: Controller perf — YOLO detection, latency, person-lost count
# ---------------------------------------------------------------------------

def chart_controller_perf(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _setup_dark_theme(plt)

    traced = [r for r in rows if r.get("yolo_total_frames") is not None]
    if len(traced) < 2:
        return

    xs = list(range(len(traced)))
    run_labels = [str(r.get("run_id", ""))[-9:] for r in traced]
    dot_colors = [_outcome_color(str(r.get("outcome", ""))) for r in traced]

    metrics = [
        ("yolo_detect_pct",       "YOLO detect rate (%)", "#2ecc71"),
        ("mean_frame_latency_ms", "Mean frame latency (ms)", "#3498db"),
        ("person_lost_count",     "Person lost count", "#e74c3c"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(6, len(traced) * 0.9 + 2), 6),
                             sharex=True)

    for ax, (key, ylabel, line_color) in zip(axes, metrics):
        ys = [_safe_float(r.get(key)) for r in traced]
        valid = [(x, y, c) for x, y, c in zip(xs, ys, dot_colors) if y is not None]
        if valid:
            vx, vy, vc = zip(*valid)
            ax.plot(list(vx), list(vy), color=line_color, linewidth=1, alpha=0.7, zorder=1)
            ax.scatter(list(vx), list(vy), c=list(vc), s=60, zorder=2,
                       edgecolors="#888", linewidths=0.4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(run_labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Controller-side performance  (from vision_main_trace.jsonl)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(jsonl_path: Path, charts_dir: Path) -> None:
    """Regenerate all charts from the full run history. Called after each run."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(jsonl_path)
    if not rows:
        print("[charts] No data yet; skipping chart generation.", flush=True)
        return

    chart_fns = [
        ("progress_over_time.png",    chart_progress_over_time),
        ("outcomes_bar.png",          chart_outcomes_bar),
        ("stability_scatter.png",     chart_stability_scatter),
        ("config_diff_heatmap.png",   chart_config_diff_heatmap),
        ("stability_timeseries.png",  chart_stability_timeseries),
        ("branch_performance.png",    chart_branch_performance),
        ("stair_phase_breakdown.png", chart_stair_phase_breakdown),
        ("controller_perf.png",       chart_controller_perf),
    ]

    generated = 0
    for filename, fn in chart_fns:
        try:
            fn(rows, charts_dir / filename)
            print(f"[charts] Generated {filename}", flush=True)
            generated += 1
        except Exception as exc:
            print(f"[charts] Skipped {filename}: {exc}", flush=True)

    print(f"[charts] {generated}/{len(chart_fns)} charts generated from {len(rows)} runs -> {charts_dir}",
          flush=True)
