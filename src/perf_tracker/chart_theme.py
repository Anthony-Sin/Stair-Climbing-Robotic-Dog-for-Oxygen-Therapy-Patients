"""Shared theme + data helpers for the performance charts.

Split out of charts.py (Phase 2 structural refactor). Row loading, the outcome
-> color map, numeric coercion, per-branch color assignment, and the dark
rcParams used by every chart generator live here so the individual chart
functions in chart_generators.py can share them.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json


def _load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not jsonl_path.exists():
        return rows
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows.sort(key=lambda r: str(r.get("timestamp") or ""))
    return rows


def _outcome_color(outcome: str) -> str:
    return {
        "completed": "#2ecc71",
        "robot_fell": "#e74c3c",
        "timeout":    "#f39c12",
        "docker_failed": "#95a5a6",
    }.get(str(outcome), "#3498db")


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _branch_colors(rows: List[Dict]) -> Dict[str, str]:
    """Assign a stable color to each unique branch name."""
    palette = ["#2980b9", "#8e44ad", "#16a085", "#d35400", "#c0392b", "#27ae60"]
    branches = list(dict.fromkeys(str(r.get("git_branch", "")) for r in rows))
    return {b: palette[i % len(palette)] for i, b in enumerate(branches)}


def _setup_dark_theme(plt) -> None:
    """Dark rcParams so charts paste cleanly onto black PowerPoint slides (transparent bg)."""
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
