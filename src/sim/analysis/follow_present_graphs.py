"""Matplotlib (Agg) chart + stats-card rendering for the person-follow presenter (follow-mode).

Split out of ``follow_present.py`` (single-responsibility): the follow-mode approach-time bar
chart, the follow-mode 2x3 dashboard (reusing the shared per-metric plotters from
``sweep_present``), and the leaderboard stats card branded for the person-follow sweep (with a
REACH STATUS column). Matplotlib is imported lazily inside the entrypoints so the rest of the
package works without it.
"""
import os
import sys

# Sibling import (sweep_present.py lives next to this file) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_present  # noqa: E402

from sweep_present import (  # noqa: E402
    STAIR_BASE_X, DEFAULT_STEP_COUNT,
    log, fmt_time,
    _assign_colors, _plot_climb_profile, _plot_steps, _plot_stability, _plot_reach,
)
from follow_present_scan import REACH_COLORS  # noqa: E402


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
