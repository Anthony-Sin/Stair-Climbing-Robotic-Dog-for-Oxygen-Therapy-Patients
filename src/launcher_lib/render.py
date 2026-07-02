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
    rows: List[str] = []
    for i, opt in enumerate(cfg.visible()):
        selected = (i == sel)
        cursor = theme.paint("›", fg="accent", bold=True) if selected else " "
        label = theme.paint(tu.pad(opt.label, label_w),
                            fg="primary" if selected else "fg", bold=selected)
        if selected and opt.kind != "bool":
            val = theme.paint("‹", fg="accent") + " " + opt.display_value(theme) + " " + theme.paint("›", fg="accent")
        else:
            val = opt.display_value(theme)
        rows.append(f"{cursor} {label}{val}")
    return rows


def render_start(cfg: Config, sel: int, theme: tu.Theme, demo: bool = False) -> List[str]:
    cols, _rows = _term_size()
    W = cols
    visible = cfg.visible()
    sel = max(0, min(sel, len(visible) - 1))
    display, _argv, _cwd, _env, supported = build_command(cfg)

    # --- banner (full width): logo + target chips + tagline ---------------
    banner = tu.panel(_LOGO, [
        _target_chips(cfg, theme),
        theme.paint("stair-climbing robotic dog · oxygen-therapy patients", fg="muted"),
    ], W, accent="cpu", theme=theme)

    # --- command preview + keys (right column / bottom) -------------------
    def info_block(width: int) -> List[str]:
        cmd_lines = [theme.paint("$ ", fg="success", bold=True)
                     + theme.paint(tu.truncate(display, width - 6), fg="secondary")]
        if not supported:
            warn = "needs Linux/ROS2" if cfg.target == "real" else "needs Windows"
            cmd_lines.append(theme.paint(f"⚠ {warn}; Enter prints cmd", fg="warning"))
        out = tu.panel("command", cmd_lines, width, accent="mem", theme=theme)
        help_txt = visible[sel].help if visible else ""
        launch_word = "print-cmd" if demo else "launch"
        keys = [
            theme.paint(tu.truncate(help_txt, width - 4), fg="fg"),
            theme.paint("↑↓", fg="secondary", bold=True) + theme.paint(" move   ", fg="muted")
            + theme.paint("←→", fg="secondary", bold=True) + theme.paint(" change   ", fg="muted")
            + theme.paint("⏎", fg="secondary", bold=True) + theme.paint(f" {launch_word}   ", fg="muted")
            + theme.paint("q", fg="secondary", bold=True) + theme.paint(" quit", fg="muted"),
        ]
        out += tu.panel("keys", keys, width, accent="net", theme=theme)
        return out

    if _two_col(W):
        gap = 2
        flags_w = max(48, min(72, int(W * 0.52)))
        info_w = W - flags_w - gap
        flags = tu.panel("flags", _flag_rows(cfg, sel, theme, 23), flags_w, accent="proc", theme=theme)
        info = info_block(info_w)
        return banner + tu.hjoin([flags, info], gap=gap)

    # narrow: stacked but compact
    out = banner
    out += tu.panel("flags", _flag_rows(cfg, sel, theme, 23), W, accent="proc", theme=theme)
    out += info_block(W)
    return out


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
