"""Data-section output for the stair-sweep presenter: git ref probe, CSV/JSON summary
writer, and the console leaderboard table.

Split out of ``sweep_present.py`` (single-responsibility).
"""
import csv
import json
import os
import subprocess

from sweep_constants import REPO_ROOT
from sweep_helpers import log, fmt_time


# ---------------------------------------------------------------------------
# data section
# ---------------------------------------------------------------------------
def _git(args):
    try:
        return subprocess.run(["git", "-C", REPO_ROOT] + args, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def write_summary(rows, meta, out_dir):
    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    json_path = os.path.join(out_dir, "sweep_summary.json")
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "episodes": rows}, f, indent=2)
    return csv_path, json_path


def print_table(rows):
    log("data section (per riser):")
    print(f"  {'riser':>8}  {'code':<14}  {'verdict':<11}  {'steps':>6}  {'time':>7}  waypoint")
    for r in rows:
        riser = f"{r['riser_m']:.3f}" if r.get("riser_m") is not None else "-"
        steps = f"{r['steps_climbed']:.1f}" if r.get("steps_climbed") is not None else "-"
        t = r.get("real_time_s") or r.get("climb_time_s")
        print(f"  {riser:>8}  {r['label_short']:<14}  {r['verdict_short']:<11}  {steps:>6}  "
              f"{fmt_time(t):>7}  {r.get('waypoint', '')}")
