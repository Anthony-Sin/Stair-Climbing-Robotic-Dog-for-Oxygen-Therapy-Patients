"""Start-screen and live-dashboard rendering + telemetry-read helpers.

Extracted verbatim from ``launcher.py``: the terminal-size helpers, the
``render_start`` start screen, the ``render_dashboard`` live view, the
telemetry row builders, and the small on-disk readers (``_find_run_dir`` /
``_read_fall_diag``) they use.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import List, Optional

from core.telemetry import term_ui as tu

from launcher_lib.config import (
    Config,
    _REAL_PIPELINE,
    _SIM_PIPELINE,
    build_command,
    catalog,
    preset_config,
    presets_for,
)
from launcher_lib.paths import REPO_ROOT


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_LOGO = "go2  follow + climb"


def _term_size():
    sz = shutil.get_terminal_size((100, 30))
    return max(54, sz.columns - 1), max(20, sz.lines)


def _width() -> int:
    return _term_size()[0]


def _two_col(width: int) -> bool:
    """Wide terminals get the btop side-by-side grid; narrow ones stack."""
    return width >= 92


def _target_chips(cfg: Config, theme: tu.Theme) -> str:
    def chip(name, label):
        on = cfg.target == name
        if on:
            return theme.paint(f" {label} ", fg="bg", bg="secondary", bold=True)
        return theme.paint(f" {label} ", fg="muted")
    return (chip("sim", "SIM · Isaac") + " " + chip("real", "REAL · Go2 EDU")
            + theme.paint("   Tab ⇄", fg="muted"))


def _flag_rows(cfg: Config, sel: int, theme: tu.Theme, label_w: int) -> List[str]:
    """The arrow-key flag editor rows: label + value, ‹ › on the selected one."""
    rows: List[str] = []
    marker = "❯" if theme.unicode else ">"
    lo, ro = ("‹", "›") if theme.unicode else ("<", ">")
    for i, opt in enumerate(cfg.visible()):
        selected = (i == sel)
        cursor = theme.paint(marker, fg="primary", bold=True) if selected else " "
        label = theme.paint(tu.pad(opt.label, label_w),
                            fg="primary" if selected else "fg", bold=selected)
        if selected and opt.kind != "bool":
            val = theme.paint(lo + " ", fg="accent") + opt.display_value(theme) + theme.paint(" " + ro, fg="accent")
        elif selected:  # bool: show the toggle hint
            val = opt.display_value(theme) + theme.paint("   ← → toggle", fg="muted")
        else:
            val = opt.display_value(theme)
        rows.append(f"{cursor} {label}{val}")
    return rows


def _target_bar(target: str, theme: tu.Theme) -> str:
    """A 'you are here' row, left-grouped: a filled pill for the active target,
    a plain-language subtitle, and an explicit ``Tab →`` to the other one."""
    sim = target == "sim"
    active = "SIM · Isaac" if sim else "REAL · Go2 EDU"
    subtitle = "the Isaac simulation" if sim else "the real Go2 EDU robot"
    other = "REAL · Go2 EDU" if sim else "SIM · Isaac"
    arrow = "⇄" if theme.unicode else "<>"
    return (" " + theme.paint("target", fg="muted") + "   "
            + theme.paint(f" {active} ", fg="bg", bg="primary", bold=True)
            + "   " + theme.paint(subtitle, fg="muted")
            + "      " + theme.paint(f"Tab {arrow} ", fg="muted")
            + theme.paint(other, fg="fg"))


def _divider(theme: tu.Theme, width: int) -> str:
    ch = "─" if theme.unicode else "-"
    return " " + theme.paint(ch * max(1, width - 2), fg="#4a4a4a")


def _section(label: str, right: str, theme: tu.Theme, width: int) -> str:
    """Left label + right hint, spread only within the capped content width."""
    left = theme.paint(label, fg="muted", bold=True)
    space = max(3, width - tu.visible_len(left) - tu.visible_len(right) - 1)
    return " " + left + " " * space + right


def render_help(cfg: Config, theme: tu.Theme) -> List[str]:
    """The full flag reference, shown as an overlay from the menu (``?``)."""
    W = _width()
    out = [
        theme.paint("flags", fg="primary", bold=True)
        + theme.paint("  — while editing a preset, type the name; add a value where a "
                      "range/choice is shown", fg="muted"),
        "",
    ]
    for name, hint, help_txt in catalog(cfg):
        label = theme.paint(tu.pad(name, 13), fg="primary")
        hintcol = theme.paint(tu.pad(hint, 20), fg="secondary")
        desc = theme.paint(tu.truncate(help_txt, max(10, W - 40)), fg="muted")
        out.append("  " + label + " " + hintcol + " " + desc)
    return out


def _menu_keys(theme: tu.Theme) -> str:
    def k(key, label):
        return theme.paint(key, fg="primary", bold=True) + theme.paint(f" {label}   ", fg="muted")
    return (k("↑↓", "move") + k("⏎", "run") + k("e", "edit")
            + k("?", "flags") + k("q", "quit")).rstrip()


def _preset_flags_plain(target: str, preset) -> str:
    """The plain flag tail of a preset's command (or ``default``)."""
    display, *_ = build_command(preset_config(target, preset))
    parts = display.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else "default"


def _preset_row(target: str, preset, selected: bool, theme: tu.Theme, namew: int) -> str:
    """One menu row: a terracotta ``❯`` + bold name marks the selection."""
    flags = _preset_flags_plain(target, preset)
    marker = "❯" if theme.unicode else ">"
    if selected:
        return (" " + theme.paint(marker, fg="primary", bold=True) + " "
                + theme.paint(tu.pad(preset.name, namew), fg="primary", bold=True)
                + theme.paint(flags, fg="secondary"))
    return ("   " + theme.paint(tu.pad(preset.name, namew), fg="fg")
            + theme.paint(flags, fg="muted"))


#: Cap the content width so a maximized terminal doesn't spread rows edge-to-edge.
_CONTENT_W = 84


def _term_rows() -> int:
    """The real terminal height (NOT the ``_term_size`` floor of 20 — we must
    know when the window is genuinely short so the header can never scroll off)."""
    try:
        return max(6, shutil.get_terminal_size((100, 30)).lines)
    except Exception:
        return 30


def _fit(rows: List[str], sel: int, avail: int, theme: tu.Theme) -> List[str]:
    """Window *rows* to *avail* lines, always keeping index *sel* visible, with
    muted ``↑/↓ N more`` markers overlaid on the clipped edges."""
    avail = max(1, avail)
    n = len(rows)
    if n <= avail:
        return list(rows)
    sel = max(0, min(sel, n - 1))
    start = max(0, min(sel - avail // 2, n - avail))
    win = list(rows[start:start + avail])
    # Overlay a marker on an edge row only when it is NOT the selected row, so
    # the highlighted preset is never hidden behind a "more" label.
    if start > 0 and start != sel:
        win[0] = " " + theme.paint(f"↑ {start} more", fg="muted")
    last = start + avail - 1
    if last < n - 1 and last != sel:
        win[-1] = " " + theme.paint(f"↓ {n - 1 - last} more", fg="muted")
    return win


def render_menu(target: str, cfg_live: Config, presets: List, sel: int, editing: bool,
                edit_cfg: Optional[Config], edit_sel: int, theme: tu.Theme,
                demo: bool = False, help_mode: bool = False) -> List[str]:
    """The preset launcher: arrow through the list and ⏎ to run, or ``e`` to
    open the arrow-key flag editor. The header is pinned; the list windows to
    the terminal height so it never scrolls the header off a short window."""
    CW = min(_width(), _CONTENT_W)
    rows_h = _term_rows()
    header = [" " + theme.paint("go2 · follow + climb", fg="primary", bold=True)]
    if rows_h >= 20:                       # drop the decorative tagline in short windows
        header.append(" " + theme.paint("stair-climbing robotic dog · oxygen-therapy patients", fg="muted"))
    header += ["", _target_bar(target, theme), _divider(theme, CW)]
    if help_mode:
        body = render_help(cfg_live, theme)
        avail = rows_h - len(header) - 3
        return header + [""] + _fit(body, 0, avail, theme) + [
            "", " " + theme.paint("press any key to go back", fg="muted")]

    display, _a, _c, _e, supported = build_command(cfg_live)
    echo = (" " + theme.paint("$", fg="success", bold=True) + " "
            + theme.paint(tu.truncate(display, CW - 3), fg="secondary"))
    warn = []
    if not supported:
        need = "Linux + ROS 2" if target == "real" else "Windows"
        warn = [" " + theme.paint(f"⚠ needs {need}; ⏎ prints the command instead", fg="warning")]

    if editing:
        name = presets[sel].name if presets else ""
        title = ["", " " + theme.paint("editing · ", fg="muted") + theme.paint(name, fg="primary", bold=True), ""]
        rows = [" " + r for r in _flag_rows(edit_cfg, edit_sel, theme, 22)]
        keys = (theme.paint("↑↓", fg="primary", bold=True) + theme.paint(" move   ", fg="muted")
                + theme.paint("←→", fg="primary", bold=True) + theme.paint(" change   ", fg="muted")
                + theme.paint("⏎", fg="primary", bold=True) + theme.paint(" run   ", fg="muted")
                + theme.paint("esc", fg="primary", bold=True) + theme.paint(" back", fg="muted"))
        footer = ["", _divider(theme, CW), echo, " " + keys] + warn
        avail = rows_h - len(header) - len(title) - len(footer) - 1
        return header + title + _fit(rows, edit_sel, avail, theme) + footer

    # --- menu mode -------------------------------------------------------
    section = ["", _section("presets", _menu_keys(theme), theme, CW), ""]
    namew = max((tu.visible_len(p.name) for p in presets), default=10) + 2
    rows = [_preset_row(target, preset, i == sel, theme, namew) for i, preset in enumerate(presets)]
    footer = ["", _divider(theme, CW), echo]
    if presets:
        footer.append(" " + theme.paint(tu.truncate(presets[sel].desc, CW - 2), fg="muted"))
    footer += warn
    avail = rows_h - len(header) - len(section) - len(footer) - 1
    return header + section + _fit(rows, sel, avail, theme) + footer


def render_start(cfg: Config, sel: int, theme: tu.Theme, demo: bool = False) -> List[str]:
    """Back-compat: the preset menu for *cfg*'s target with *sel* highlighted."""
    presets = presets_for(cfg.target)
    s = max(0, min(sel, len(presets) - 1)) if presets else 0
    live = preset_config(cfg.target, presets[s]) if presets else cfg
    return render_menu(cfg.target, live, presets, s, False, None, 0, theme, demo)


def _spinner(t: float, theme: tu.Theme) -> str:
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if theme.unicode else "|/-\\"
    return theme.paint(frames[int(t * 10) % len(frames)], fg="secondary", bold=True)


def _find_run_dir() -> Optional[str]:
    """The active sim run folder (run_sim.ps1 writes log/latest_run.txt at startup)."""
    try:
        with open(os.path.join(REPO_ROOT, "log", "latest_run.txt"), encoding="utf-8") as fh:
            d = fh.read().strip()
        return d if d and os.path.isdir(d) else None
    except Exception:
        return None


def _read_fall_diag(run_dir: Optional[str]) -> Optional[dict]:
    """Last physics sample from the run's fall_diag stream (see fall_diag_schema)."""
    if not run_dir:
        return None
    path = os.path.join(run_dir, "debug", "isaac_env.jsonl")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            chunk = fh.read().decode("utf-8", "replace")
    except Exception:
        return None
    for line in reversed(chunk.splitlines()):
        if '"fall_diag"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("event", {}).get("action") == "fall_diag":
            return d.get("sim") or {}
    return None


#: Telemetry views the dashboard can toggle through with `t`.
TELE_VIEWS = ("robot", "mission")
TELE_TITLE = {"robot": "robot pose", "mission": "mission"}


def _ang_color(v) -> str:
    a = abs(v or 0.0)
    return "success" if a < 10 else "warning" if a < 25 else "error"


def _telemetry_rows(sim: Optional[dict], x_hist, theme: tu.Theme, width: int,
                    view: str = "robot") -> List[str]:
    if not sim:
        return [theme.paint("(no robot telemetry yet)", fg="muted")]

    if view == "mission":
        return _mission_rows(sim, theme, width)

    # --- robot pose view -------------------------------------------------
    x, h = sim.get("x"), sim.get("h")
    pitch, roll = sim.get("pitch"), sim.get("roll")
    cmd = sim.get("policy_cmd") or []
    rows = [tu.kv("pos x", f"{x:+.2f} m" if x is not None else "—", theme, 8, "secondary")]
    if h is not None:
        hc = "success" if h > 0.28 else "warning" if h > 0.18 else "error"
        rows.append(tu.kv("height", f"{h:.2f} m", theme, 8, hc))
    if pitch is not None:
        rows.append(tu.kv("pitch", f"{pitch:+5.1f}°", theme, 8, _ang_color(pitch)))
    if roll is not None:
        rows.append(tu.kv("roll", f"{roll:+5.1f}°", theme, 8, _ang_color(roll)))
    if len(cmd) >= 3:
        rows.append(tu.kv("cmd", f"vx{cmd[0]:+.2f} vy{cmd[1]:+.2f} wz{cmd[2]:+.2f}", theme, 8, "blue"))
    if x_hist:
        rows.append("x-trace " + tu.sparkline(x_hist, theme, width=max(8, width - 12)))
    return rows


def _mission_rows(sim: dict, theme: tu.Theme, width: int) -> List[str]:
    """Second telemetry view: handoff phase, climb step, person-gap (from fall_diag)."""
    h = sim.get("handoff") or {}
    state = str(h.get("handoff_state", "?")).upper()
    state_color = "warning" if state == "CLIMB" else "secondary"
    rows = [tu.kv("phase", state, theme, 8, state_color, bold_value=True)]

    step = h.get("stair_count", 0)
    climbs = h.get("handoff_climbs_done", 0)
    rows.append(tu.kv("stair", f"step {step}   climbed {climbs}", theme, 8, "primary"))

    gap, seen = sim.get("gap_m"), sim.get("person_detected")
    if gap is not None:
        gc = "success" if gap <= 0.6 else "warning" if gap <= 1.5 else "error"
        rows.append(tu.kv("person", f"{gap:.2f} m", theme, 8, gc))
    else:
        rows.append(tu.kv("person", "seen, no range" if seen else "lost", theme, 8,
                          "warning" if seen else "muted"))

    det = sim.get("stairs_detected")
    gate = sim.get("stairs_action_active")
    stairs = ("detected" if det else "none") + (" · climb-gate" if gate else "")
    rows.append(tu.kv("stairs", stairs, theme, 8, "success" if det else "muted"))

    bvx = sim.get("body_vx")
    if bvx is not None:
        rows.append(tu.kv("speed", f"{bvx:+.2f} m/s", theme, 8, "blue"))
    tilt = sim.get("tilt_deg")
    if tilt is not None:
        rows.append(tu.kv("tilt", f"{tilt:.1f}°", theme, 8, _ang_color(tilt)))
    if h.get("stalled"):
        rows.append(theme.paint("⚠ stall detected", fg="warning", bold=True))
    return rows


def render_dashboard(cfg: Config, stages: dict, log_tail, started: float,
                     theme: tu.Theme, finished: Optional[int], progress_hist,
                     telemetry: Optional[dict] = None, x_hist=None,
                     run_dir: Optional[str] = None, tele_view: str = "robot",
                     scroll: int = 0) -> List[str]:
    cols, term_rows = _term_size()
    W = cols
    pipeline = _SIM_PIPELINE if cfg.target == "sim" else _REAL_PIPELINE
    done = sum(1 for s in pipeline if stages.get(s, {}).get("state") in
               ("ready", "complete", "ok", "pruned", "cleanup", "skipped"))
    failed = any(v.get("state") in ("failed", "error") for v in stages.values())
    frac = 1.0 if finished is not None else ((done / len(pipeline)) if pipeline else 0.0)
    elapsed = time.time() - started
    mm, ss = divmod(int(elapsed), 60)

    if finished == 0:
        word, hue, glyph = "done", "success", "✔" if theme.unicode else "OK"
    elif finished not in (None, 0) or failed:
        word, hue, glyph = "failed", "error", "✖" if theme.unicode else "X"
    else:
        word, hue, glyph = "running", "net", _spinner(elapsed, theme)

    # --- header bar -------------------------------------------------------
    hdr_left = f" {glyph}  go2 · {cfg.target.upper()} · {word} · {mm:02d}:{ss:02d}"
    hdr_right = f"{done}/{len(pipeline)} stages · {time.strftime('%H:%M:%S')} "
    out = [tu.bar(hdr_left, hdr_right, W, theme, bg=hue, fg="bg")]

    # --- progress (full width) -------------------------------------------
    meter_w = max(16, W - 22)
    out += tu.panel("progress", [
        tu.meter(frac, meter_w, theme) + f"  {int(frac*100):3d}%",
        "trend " + tu.sparkline(progress_hist or [0], theme, width=meter_w),
    ], W, accent="cpu", theme=theme)

    # --- body grid (fills remaining height) ------------------------------
    def pipeline_rows():
        rows = []
        for s in pipeline:
            st = stages.get(s)
            if st:
                rows.append(tu.status_line(st["ts"], s, st["state"], st["msg"], theme))
            else:
                rows.append(theme.paint("--:--:-- ", fg="muted") + tu.state_glyph("skipped", theme)
                            + " " + theme.paint(tu.pad(s, 11), fg="muted")
                            + theme.paint("pending", fg="muted"))
        return rows

    def console_panel(view_h, width):
        """Console with scrollback: a window of `view_h` rows + a live/scrolled footer."""
        lines = list(log_tail)
        total = len(lines)
        view_h = max(1, view_h)
        max_off = max(0, total - view_h)
        off = min(max(0, scroll), max_off)
        end = total - off
        start = max(0, end - view_h)
        window = lines[start:end]
        body = [theme.paint(tu.truncate(ln, width - 4), fg="fg") for ln in window]
        if not body:
            body = [theme.paint("(waiting for output…)", fg="muted")]
        body += [""] * max(0, view_h - len(body))  # fill so the box bottom is flush
        if off == 0:
            foot = f"live · {total} lines · ↑↓ scroll"
        else:
            foot = f"⏸ {start + 1}-{end}/{total} · End=live · {off}↑"
        return tu.panel("console", body, width, accent="proc", theme=theme, footer=foot)

    tele_title = TELE_TITLE.get(tele_view, "telemetry")
    body_h = max(8, term_rows - len(out) - 1)  # minus footer
    if _two_col(W):
        gap = 2
        left_w = max(38, min(48, int(W * 0.40)))
        right_w = W - left_w - gap
        left = tu.panel("pipeline", pipeline_rows(), left_w, accent="net", theme=theme)
        left += tu.panel(tele_title, _telemetry_rows(telemetry, x_hist, theme, left_w, tele_view),
                         left_w, accent="mem", theme=theme, footer="t: toggle view")
        left = tu.fit_height(left, body_h)
        right = console_panel(body_h - 2, right_w)
        out += tu.hjoin([left, right], gap=gap)
    else:
        out += tu.panel("pipeline", pipeline_rows(), W, accent="net", theme=theme)
        out += tu.panel(tele_title, _telemetry_rows(telemetry, x_hist, theme, W, tele_view), W,
                        accent="mem", theme=theme, footer="t: toggle view")
        rem = max(3, term_rows - len(out) - 1)
        out += console_panel(rem - 2, W)

    # --- footer bar -------------------------------------------------------
    rd = os.path.basename(run_dir) if run_dir else ""
    out.append(tu.bar("  q stop · ↑↓ scroll · t view", (rd + "  ") if rd else "", W, theme,
                      bg="surface", fg="muted", bold=False))
    return out
