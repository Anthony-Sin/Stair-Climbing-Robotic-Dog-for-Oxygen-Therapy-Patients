"""Matplotlib (Agg) chart + stats-card rendering for the stair-sweep presenter.

Split out of ``sweep_present.py`` (single-responsibility): the dark-theme setup, per-metric
plot helpers, the multi-panel dashboard, and the leaderboard stats card (also embedded as
montage cell 6). Matplotlib is imported lazily inside the entrypoints so the rest of the
package works without it.
"""
import os

from sweep_constants import (
    STAIR_BASE_X, FALL_TILT_DEG, COLLAPSE_H_M, UPRIGHT_TILT_DEG,
    VERDICT_COLORS, DEFAULT_STEP_COUNT, DEFAULT_TOP_EDGE_X,
)
from sweep_helpers import log, fmt_time


# ---------------------------------------------------------------------------
# graphs (matplotlib, Agg)
# ---------------------------------------------------------------------------
def _setup_dark_theme(plt) -> None:
    """Dark rcParams for charts that paste cleanly onto black PowerPoint slides (transparent bg)."""
    plt.rcParams.update({
        "figure.facecolor":  "none",
        "axes.facecolor":    "none",
        "savefig.facecolor": "none",
        "text.color":        "#e0e0e0",
        "axes.labelcolor":   "#e0e0e0",
        "axes.edgecolor":    "#666666",
        "xtick.color":       "#cccccc",
        "ytick.color":       "#cccccc",
        "grid.color":        "#444444",
        "axes.titlecolor":   "#e0e0e0",
        "legend.facecolor":  "#0d0d0d",
        "legend.edgecolor":  "#666666",
        "legend.framealpha": 0.6,
    })


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
    ax.axvline(STAIR_BASE_X, ls="--", color="0.65", lw=1.2)
    ax.text(STAIR_BASE_X + 0.05, 0.03, "stair base", color="0.6", fontsize=8)
    ax.axhspan(0.22, 0.6, color="#2e7d32", alpha=0.12)
    ax.axhline(COLLAPSE_H_M, ls=":", color="#e05050", lw=1.0)
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
    ax.axhline(full, ls="--", color="0.65", lw=1.0)
    ax.text(len(rows) - 0.5, full + 0.2, f"full staircase ({full})", ha="right", color="0.6", fontsize=8)
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
    ax.axhspan(0, UPRIGHT_TILT_DEG, color="#2e7d32", alpha=0.14)
    ax.axhline(FALL_TILT_DEG, ls="--", color="#e05050", lw=1.2)
    ax.text(len(rows) - 0.5, FALL_TILT_DEG + 1, f"fall line ({FALL_TILT_DEG:.0f}°)", ha="right",
            color="#e05050", fontsize=8)
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
    ax.axhline(STAIR_BASE_X, ls=":", color="0.65", lw=1.0)
    ax.text(len(rows) - 0.5, STAIR_BASE_X + 0.05, "stair base", ha="right", color="0.6", fontsize=8)
    top = rows[0].get("top_edge_x_m") or DEFAULT_TOP_EDGE_X
    ax.axhline(top, ls="--", color="0.65", lw=1.0)
    ax.text(len(rows) - 0.5, top + 0.05, "top edge", ha="right", color="0.6", fontsize=8)
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
    _setup_dark_theme(plt)
    os.makedirs(graphs_dir, exist_ok=True)
    _assign_colors(eps)
    written = []

    def _save(fig, name):
        p = os.path.join(graphs_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight", transparent=True)
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
        fig.suptitle("Stair-Climb Sweep — blind-RL climb policy + O2 payload", fontsize=15, fontweight="bold", color="#e0e0e0")
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
