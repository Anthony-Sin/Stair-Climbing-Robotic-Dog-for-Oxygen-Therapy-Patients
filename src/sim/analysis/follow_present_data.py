"""Data-section console output for the person-follow presenter (follow-mode print_table).

Split out of ``follow_present.py`` (single-responsibility): the follow-mode leaderboard table
(REACH STATUS + approach time + climb verdict), the follow-mode override of
``sweep_present.print_table``.
"""
from sweep_present import log, fmt_time


# ---------------------------------------------------------------------------
# data section helpers (follow-mode print_table override)
# ---------------------------------------------------------------------------
def _print_table_follow(rows):
    log("data section (per riser):")
    print(f"  {'riser':>8}  {'code':<14}  {'reach':<14}  {'steps':>6}  {'approach':>9}  climb_verdict")
    for r in rows:
        riser = f"{r['riser_m']:.3f}" if r.get("riser_m") is not None else "-"
        steps = f"{r['steps_climbed']:.1f}" if r.get("steps_climbed") is not None else "-"
        appr  = fmt_time(r.get("approach_time_s"))
        print(f"  {riser:>8}  {r['label_short']:<14}  {r.get('follow_reach','?'):<14}  "
              f"{steps:>6}  {appr:>9}  {r.get('verdict_short','')}")
