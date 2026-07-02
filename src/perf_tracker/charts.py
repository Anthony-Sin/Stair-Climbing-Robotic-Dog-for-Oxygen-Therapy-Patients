"""
Auto-generate performance charts from performance_table.jsonl.
Called by update_table.py after every run.
Silently skips if matplotlib is not installed.

Charts generated:
  progress_over_time.png    -- max_x_m per run, annotated with git SHA, line coloured by branch
  outcomes_bar.png          -- outcome distribution with count + percentage
  stability_scatter.png     -- max_pitch vs max_x_m, sized by action norm
  config_diff_heatmap.png   -- key params per run, amber cells = changed from prior run
  stability_timeseries.png  -- pitch / roll / action_norm / vx_cmd over time (4 subplots)
  branch_performance.png    -- max_x_m distribution grouped by git branch
  stair_phase_breakdown.png -- approach vs stair phase time per run (stacked bar)
  controller_perf.png       -- YOLO detection %, frame latency, person-lost count over runs

Structure note (Phase 2 refactor): this module is a facade. The shared theme +
data helpers now live in chart_theme.py and the chart generators (+ generate_all)
in chart_generators.py; they are re-exported below so every previously-importable
name still resolves from perf_tracker.charts.
"""

from chart_theme import (  # noqa: F401
    _load_rows,
    _outcome_color,
    _safe_float,
    _branch_colors,
    _setup_dark_theme,
)

from chart_generators import (  # noqa: F401
    chart_progress_over_time,
    chart_outcomes_bar,
    chart_stability_scatter,
    chart_config_diff_heatmap,
    chart_stability_timeseries,
    chart_branch_performance,
    chart_stair_phase_breakdown,
    chart_controller_perf,
    generate_all,
)
