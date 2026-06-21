"""Offline analysis tools for sim runs.

Standalone CLI scripts that read a run's ``debug/isaac_env.jsonl`` ``fall_diag``
stream from the repo-root ``log/`` dir. Not imported by the simulation; run
directly, e.g. ``python sim/analysis/analyze_climb.py``.

Members:
  - analyze_climb         3-way climb verdict (FELL / COLLIDED / CLEAN CLIMB)
  - analyze_follow_climb  verbose follow-up-stairs timeline + controller-trace correlation
  - trace_climber         closed-loop stair-climber engagement/gait trace
"""
