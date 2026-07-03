"""The go2 preset launcher: a menu of ready-to-run profiles, editable before launch.

Arrow through a list of predefined launch profiles (``Follow + climb demo``,
``Headless``, ``Stair waypoint self-test`` …) and press ⏎ to run the highlighted
one — no configuration needed for the common case. Press ``e`` to open the
arrow-key flag editor for the selected preset (↑↓ pick a flag, ←→ change it) and
⏎ to run your tweaked version. It assembles and runs the exact same underlying
command (``sim\\run_sim.bat …`` / ``./real/run_real.sh …``); the entry points are
untouched.

The public names ``interactive`` / ``preview`` / ``main`` are kept (the
``launcher.py`` facade re-exports them and ``main`` stays the entry point).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional

from core.telemetry import term_ui as tu

from launcher_lib.config import (
    Config,
    _real_config,
    _sim_config,
    build_command,
    preset_config,
    presets_for,
)
from launcher_lib.keyreader import KeyReader
from launcher_lib.render import (
    _term_size,
    render_menu,
)
from launcher_lib.runner import (
    run_passthrough,
    run_with_dashboard,
)


# ---------------------------------------------------------------------------
# Menu state + helpers
# ---------------------------------------------------------------------------


@dataclass
class MenuState:
    target: str
    sel: int = 0
    editing: bool = False
    help_mode: bool = False
    edit_cfg: Optional[Config] = None
    edit_sel: int = 0


def _base_cfg(target: str) -> Config:
    return _sim_config() if target == "sim" else _real_config()


def _await_return(theme: tu.Theme) -> bool:
    """After a run, pause on the normal screen so output stays readable."""
    try:
        ans = input("\n" + theme.paint("  ⏎ back to the menu · q quit  ", fg="muted"))
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() not in ("q", "quit", "exit")


# ---------------------------------------------------------------------------
# Menu screen (one alt-screen session that returns a launch/quit intent)
# ---------------------------------------------------------------------------


def _menu_screen(state: MenuState, theme: tu.Theme, demo: bool, dashboard: bool) -> dict:
    view = "dashboard" if dashboard else "stream"
    with tu.Screen(theme=theme) as screen, KeyReader() as keys:
        while True:
            presets = presets_for(state.target)
            if presets:
                state.sel = max(0, min(state.sel, len(presets) - 1))

            if state.editing:
                vis = state.edit_cfg.visible()
                state.edit_sel = max(0, min(state.edit_sel, len(vis) - 1))
                cfg_live = state.edit_cfg
            else:
                cfg_live = preset_config(state.target, presets[state.sel]) if presets else _base_cfg(state.target)

            screen.render(render_menu(state.target, cfg_live, presets, state.sel, state.editing,
                                      state.edit_cfg, state.edit_sel, theme, demo, state.help_mode))
            key = keys.read()

            if state.help_mode:                       # any key dismisses the flag overlay
                state.help_mode = False
                continue

            if state.editing:
                vis = state.edit_cfg.visible()
                if key == "esc":
                    state.editing = False
                elif key in ("up", "k"):
                    state.edit_sel = (state.edit_sel - 1) % len(vis)
                elif key in ("down", "j"):
                    state.edit_sel = (state.edit_sel + 1) % len(vis)
                elif key in ("left", "right", "space"):
                    vis[state.edit_sel].cycle(-1 if key == "left" else 1)
                elif key == "enter":
                    state.editing = False
                    return {"kind": "launch", "cfg": state.edit_cfg, "view": view}
                elif key in ("q", "quit"):
                    return {"kind": "quit"}
                continue

            # --- menu navigation ---
            if key in ("q", "quit", "esc"):
                return {"kind": "quit"}
            if not presets:
                continue
            if key in ("up", "k"):
                state.sel = (state.sel - 1) % len(presets)
            elif key in ("down", "j"):
                state.sel = (state.sel + 1) % len(presets)
            elif key == "tab":
                state.target = "real" if state.target == "sim" else "sim"
                state.sel = 0
            elif key in ("e", "right"):
                state.editing = True
                state.edit_cfg = preset_config(state.target, presets[state.sel])
                state.edit_sel = 0
            elif key in ("?", "h"):
                state.help_mode = True
            elif key == "enter":
                return {"kind": "launch", "cfg": preset_config(state.target, presets[state.sel]),
                        "view": view}


def interactive(cfg_sim: Config, cfg_real: Config, theme: tu.Theme, start_target: str,
                demo: bool, dashboard: bool) -> int:
    """The launcher loop: pick/edit a preset, run it, review, repeat."""
    state = MenuState(target=start_target)
    while True:
        intent = _menu_screen(state, theme, demo, dashboard)
        if intent["kind"] == "quit":
            return 0

        cfg, view = intent["cfg"], intent["view"]
        display, argv, cwd, env, supported = build_command(cfg)

        if demo or not supported:
            print(theme.paint("$ ", fg="success", bold=True) + theme.paint(display, fg="secondary"))
            if not supported and not demo:
                where = "Linux + ROS 2" if cfg.target == "real" else "Windows"
                print(theme.paint(f"(launch this on {where}; command printed above)", fg="warning"))
            if not _await_return(theme):
                return 0
            continue

        rc = (run_with_dashboard(cfg, argv, cwd, env, theme) if view == "dashboard"
              else run_passthrough(argv, cwd, env, theme, display))
        if not _await_return(theme):
            return rc


# ---------------------------------------------------------------------------
# Non-interactive preview + CLI entry
# ---------------------------------------------------------------------------


def preview(theme: tu.Theme) -> int:
    """One-shot render of the menu, the flag editor, and a restyled log stream."""
    cols, _ = _term_size()
    presets = presets_for("sim")
    print(theme.paint("  ── the preset menu (↑↓ move, ⏎ run, e edit) ──", fg="muted"))
    print("\n".join(render_menu("sim", preset_config("sim", presets[0]), presets, 0,
                                 False, None, 0, theme)))
    print()
    print(theme.paint("  ── editing a preset: ↑↓ pick a flag, ←→ change it (no typing) ──", fg="muted"))
    ecfg = preset_config("sim", presets[1])       # the Headless preset
    print("\n".join(render_menu("sim", ecfg, presets, 1, True, ecfg, 3, theme)))
    print()
    print(theme.paint("  ── run_sim output, restyled (what streams after ⏎) ──", fg="muted"))
    sample = [
        ("12:00:01", "setup", "ready", "selected options · PersonApproachTurns 2"),
        ("12:00:02", "build", "skipped", "using existing image"),
        ("12:00:03", "isaac", "start", "Isaac Sim window launch"),
    ]
    for ts, stage, state, msg in sample:
        print(tu.status_line(ts, stage, state, msg, theme))
    print("          " + tu.thinking(3.0, theme, verb="Percolating",
                                      suffix="waiting for world_ready   00:41"))
    for ts, stage, state, msg in [
        ("12:00:48", "isaac_wait", "complete", "world_ready observed · scene loaded"),
        ("12:03:20", "summary", "complete", "run finished cleanly · 02:19"),
    ]:
        print(tu.status_line(ts, stage, state, msg, theme))
    print()
    done = tu.panel("done", [
        tu.kv("logs", "log\\run_sim_20260702_142645", theme, 7, "primary", bold_value=False),
        tu.kv("open", "00_READ_ME_FIRST.txt", theme, 7, "muted", bold_value=False),
    ], min(cols, 72), accent="success", theme=theme)
    print("\n".join(done))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preset launcher for the go2 sim/real stack")
    ap.add_argument("--sim", action="store_true", help="preselect the Isaac sim target")
    ap.add_argument("--real", action="store_true", help="preselect the real Go2 EDU target")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI color (also honors NO_COLOR)")
    ap.add_argument("--dashboard", action="store_true",
                    help="launch runs into the live telemetry dashboard instead of streaming logs")
    ap.add_argument("--no-dashboard", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: streaming is now the default
    ap.add_argument("--demo", action="store_true",
                    help="interactive, but ⏎ prints the command instead of launching")
    ap.add_argument("--preview", action="store_true",
                    help="non-interactive one-shot render of the UI (for screenshots/no TTY)")
    args = ap.parse_args(argv)

    # The menu draws box-drawing + glyphs; make sure stdout can encode them
    # (legacy Windows code pages are cp1252 and would raise UnicodeEncodeError).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    force_color = False if args.no_color else (True if args.preview else None)
    theme = tu.Theme.detect(force_color=force_color)

    if args.preview:
        return preview(theme)

    if not sys.stdin.isatty():
        preview(theme)
        print("\n(no interactive TTY detected — run `python launcher.py` in a real terminal to drive it)")
        return 0

    start_target = "real" if args.real else "sim"
    return interactive(_sim_config(), _real_config(), theme, start_target,
                       demo=args.demo, dashboard=args.dashboard)
