"""Run-directory resolution, JSONL loading, event accessors, and report parsing.

Moved verbatim from analyze_run.py as part of a pure structural split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Resolve run directory
# ---------------------------------------------------------------------------
def resolve_run_dir(arg: Optional[str] = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    log_root = repo_root / "log"

    if arg:
        p = Path(arg)
        # Try as-is first, then relative to repo root, then relative to log dir
        candidates = [p]
        if not p.is_absolute():
            candidates += [repo_root / p, log_root / p]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"Run dir not found: {arg!r} (tried {[str(c) for c in candidates]})")

    # Auto-detect from latest_run.txt
    latest_file = log_root / "latest_run.txt"
    if latest_file.exists():
        run_name = latest_file.read_text().strip()
        if run_name:
            candidate = log_root / run_name
            if candidate.is_dir():
                return candidate

    # Fall back: most recent run_sim_* directory
    candidates = sorted(log_root.glob("run_sim_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No run directory found. Pass one explicitly or ensure {log_root} has run_sim_* folders."
    )


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    events: List[Dict] = []
    if not path.exists():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _action(event: Dict) -> str:
    return event.get("event", {}).get("action", "")


def _sim(event: Dict) -> Dict:
    return event.get("sim", {})


def _ts(event: Dict) -> str:
    return event.get("@timestamp", "")


def _level(event: Dict) -> str:
    return event.get("level", "")


# ---------------------------------------------------------------------------
# Parse text/JSON reports
# ---------------------------------------------------------------------------
def parse_evaluation_summary(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def parse_stair_demo_report(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
