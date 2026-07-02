"""Interactive start-screen loop, non-interactive preview, and CLI entry.

Extracted verbatim from ``launcher.py``: the ``interactive`` loop, the
``preview`` one-shot render, and ``main`` (arg parsing + dispatch). The facade
``launcher.py`` keeps ``if __name__ == "__main__": sys.exit(main())``.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from typing import List, Optional

from core.telemetry import term_ui as tu

from launcher_lib.config import (
    Config,
    _real_config,
    _sim_config,
    build_command,
)
from launcher_lib.keyreader import KeyReader
from launcher_lib.render import (
    _term_size,
    _two_col,
    render_dashboard,
    render_start,
)
from launcher_lib.runner import (
    run_passthrough,
    run_with_dashboard,
)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def interactive(cfg_sim: Config, cfg_real: Config, theme: tu.Theme, start_target: str,
                demo: bool, dashboard: bool) -> int:
    cfg = cfg_sim if start_target == "sim" else cfg_real
    sel = 0
    screen = tu.Screen(theme=theme)
    with screen, KeyReader() as keys:
        while True:
            visible = cfg.visible()
            sel = max(0, min(sel, len(visible) - 1))
            screen.render(render_start(cfg, sel, theme, demo))
            k = keys.read()
            if k in ("q", "quit", "esc"):
                return 0
            if k == "tab":
                cfg = cfg_real if cfg.target == "sim" else cfg_sim
                sel = 0
                continue
            if k == "up":
                sel = (sel - 1) % len(visible)
            elif k == "down":
                sel = (sel + 1) % len(visible)
            elif k in ("left", "right", "space"):
                if visible:
                    visible[sel].cycle(-1 if k == "left" else 1)
            elif k == "enter":
                break
    # leaving the alt-screen, then launch
    display, argv, cwd, env, supported = build_command(cfg)
    if demo or not supported:
        print(theme.paint("$ ", fg="success", bold=True) + theme.paint(display, fg="secondary"))
        if not supported and not demo:
            tgt = "Linux + ROS 2" if cfg.target == "real" else "Windows"
            print(theme.paint(f"(launch this on {tgt}; command printed above)", fg="warning"))
        return 0
    if dashboard:
        return run_with_dashboard(cfg, argv, cwd, env, theme)
    return run_passthrough(argv, cwd, env, theme, display)


def preview(theme: tu.Theme) -> int:
    """Non-interactive: print the start screen and a sample dashboard once."""
    cfg = _sim_config()
    cfg.get("headless").value = True
    cfg.get("waypoint_test").value = True
    cols, _ = _term_size()
    layout = "two-column grid" if _two_col(cols) else "stacked (narrow)"
    print(theme.paint(f"  [{cols} cols → {layout}]", fg="muted"))
    print("\n".join(render_start(cfg, 2, theme)))
    print()
    stages = {
        "setup": {"ts": "12:00:01", "state": "ready", "msg": "selected options"},
        "network": {"ts": "12:00:01", "state": "ready", "msg": "frame 127.0.0.1:52002"},
        "build": {"ts": "12:00:02", "state": "skipped", "msg": "using existing image"},
        "isaac": {"ts": "12:00:03", "state": "start", "msg": "Isaac Sim window launch"},
        "isaac_wait": {"ts": "12:00:48", "state": "complete", "msg": "world_ready observed"},
        "docker": {"ts": "12:01:05", "state": "running", "msg": "controller up; following"},
    }
    tail = collections.deque(
        [f"[isaac] sim step {i}" for i in range(40)]
        + ["world_ready: scene loaded (Biped + stairs + O2)",
           "[handoff] PGTT walk -> blind_rl climb armed",
           "climb: step 3/14  base_height 0.31m  upright"], maxlen=200)
    telemetry = {"x": 2.34, "h": 0.31, "pitch": 8.2, "roll": -3.1, "tilt_deg": 8.6,
                 "body_vx": 0.38, "action_norm": 4.7, "policy_cmd": [0.42, 0.0, 0.05],
                 "person_detected": True, "gap_m": 0.52, "stairs_detected": True,
                 "stairs_action_active": True,
                 "handoff": {"handoff_state": "climb", "stair_count": 3,
                             "handoff_climbs_done": 2, "stalled": False}}
    x_hist = [(-4.5 + 0.18 * i) for i in range(38)]
    hist = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8]
    print(theme.paint("  ── telemetry view: ROBOT (press t to toggle) ──", fg="muted"))
    print("\n".join(render_dashboard(cfg, stages, tail, time.time() - 67.0, theme, None, hist,
                                     telemetry=telemetry, x_hist=x_hist,
                                     run_dir="log/run_sim_20260630", tele_view="robot")))
    print()
    print(theme.paint("  ── telemetry view: MISSION + console scrolled up 6 ──", fg="muted"))
    print("\n".join(render_dashboard(cfg, stages, tail, time.time() - 67.0, theme, None, hist,
                                     telemetry=telemetry, x_hist=x_hist,
                                     run_dir="log/run_sim_20260630", tele_view="mission", scroll=6)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="btop-style launcher for the go2 sim/real stack")
    ap.add_argument("--sim", action="store_true", help="preselect the Isaac sim target")
    ap.add_argument("--real", action="store_true", help="preselect the real Go2 EDU target")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI color (also honors NO_COLOR)")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="passthrough mode: stream the underlying launcher instead of the live dashboard")
    ap.add_argument("--demo", action="store_true",
                    help="interactive, but Enter prints the command instead of launching")
    ap.add_argument("--preview", action="store_true",
                    help="non-interactive one-shot render of the UI (for screenshots/no TTY)")
    args = ap.parse_args(argv)

    force_color = False if args.no_color else (True if args.preview else None)
    theme = tu.Theme.detect(force_color=force_color)

    if args.preview:
        return preview(theme)

    if not sys.stdin.isatty():
        # No interactive terminal: show the UI once and explain how to drive it.
        preview(theme)
        print("\n(no interactive TTY detected — run `python launcher.py` in a real terminal to drive it)")
        return 0

    start_target = "real" if args.real else "sim"
    return interactive(_sim_config(), _real_config(), theme, start_target,
                       demo=args.demo, dashboard=not args.no_dashboard)
