#!/usr/bin/env python3
"""go2 launcher — a btop-style interface to start the robot stack.

A "cool interface to start the thing" for both the **sim** and the **real**
robot, styled per the repo's ``DESIGN.md`` (rounded notched boxes, gradient
meters, dense panels, jewel-tone accents). You pick the flags interactively and
it builds + runs the equivalent normal command, then shows a live dashboard of
the launch.

This is a thin, additive convenience layer. The underlying entry points are
untouched and you can STILL run them directly with normal commands:

    sim:   sim\\run_sim.bat [--headless] [--locomotion-policy pgtt] ...
    real:  ./real/run_real.sh [--lidar] [--record]

Usage
-----
    python launcher.py                # interactive start screen (sim/real)
    python launcher.py --real         # preselect the real-robot target
    python launcher.py --preview      # static, non-interactive UI preview (no GPU/TTY)
    python launcher.py --demo         # interactive, but Enter prints the command instead of launching
    python launcher.py --no-color     # plain ASCII (also respects NO_COLOR)

Keys: ↑/↓ move · ←/→ or Space change · Tab switch sim/real · Enter launch · q quit
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.telemetry import term_ui as tu  # noqa: E402

_STAGE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(\S+)\s+(\S+)\s+(.*)$")

# Ordered pipeline stages used to draw the launch-progress meter. Both targets
# map their events onto this spine; unknown stages still scroll in the log.
_SIM_PIPELINE = ["setup", "network", "build", "isaac", "isaac_wait",
                 "operator", "models", "docker", "summary"]
_REAL_PIPELINE = ["preflight", "record", "control", "vision", "summary"]


# ---------------------------------------------------------------------------
# Configurable options (data-driven; each maps to a real launcher flag)
# ---------------------------------------------------------------------------


@dataclass
class Option:
    key: str
    label: str
    kind: str                      # 'bool' | 'choice' | 'int' | 'float'
    value: Any
    flag: str = ""
    choices: Optional[List[Any]] = None
    default: Any = None
    step: float = 1.0
    minimum: float = 0.0
    maximum: float = 1e9
    help: str = ""
    bool_true_flag: bool = True    # True: emit flag when value True; False: emit when value False (--no-x)
    visible_if: Optional[Callable[["Config"], bool]] = None

    def cycle(self, direction: int) -> None:
        if self.kind == "bool":
            self.value = not self.value
        elif self.kind == "choice":
            i = self.choices.index(self.value)
            self.value = self.choices[(i + direction) % len(self.choices)]
        else:
            v = float(self.value) + direction * self.step
            v = max(self.minimum, min(self.maximum, v))
            self.value = int(round(v)) if self.kind == "int" else round(v, 3)

    def to_flags(self) -> List[str]:
        if self.kind == "bool":
            if self.bool_true_flag and self.value:
                return [self.flag]
            if (not self.bool_true_flag) and (not self.value):
                return [self.flag]
            return []
        if self.default is not None and self.value == self.default:
            return []
        return [self.flag, str(self.value)]

    def display_value(self, theme: tu.Theme) -> str:
        if self.kind == "bool":
            return (theme.paint("on", fg="success", bold=True) if self.value
                    else theme.paint("off", fg="muted"))
        return theme.paint(str(self.value), fg="primary", bold=True)


@dataclass
class Config:
    target: str                    # 'sim' | 'real'
    options: List[Option] = field(default_factory=list)

    def get(self, key: str) -> Option:
        return next(o for o in self.options if o.key == key)

    def visible(self) -> List[Option]:
        return [o for o in self.options if o.visible_if is None or o.visible_if(self)]


def _sim_config() -> Config:
    cfg = Config(target="sim")
    cfg.options = [
        Option("policy", "locomotion policy", "choice", "pgtt",
               flag="--locomotion-policy", choices=["pgtt", "parkour"], default="pgtt",
               help="pgtt = phase-guided heightmap stair policy (default); parkour = legacy depth/vision."),
        Option("pgtt_level", "pgtt level", "choice", "level17",
               flag="--pgtt-level", choices=["level10", "level15", "level17", "level20"],
               default="level17", help="Curriculum checkpoint; higher = trained on taller stairs.",
               visible_if=lambda c: c.get("policy").value == "pgtt"),
        Option("climb", "climb backend", "choice", "blind_rl",
               flag="--handoff-climb-backend", choices=["blind_rl", "parkour", "ik"],
               default="blind_rl", help="Policy that takes over to climb after PGTT walks to the stairs."),
        Option("headless", "headless render", "bool", False,
               flag="--headless", help="No GUI window (faster; recordings still written to disk)."),
        Option("fast_render", "fast render", "bool", False,
               flag="--fast-render", help="Lower-fidelity RTX settings for quicker startup."),
        Option("vision_preview", "vision preview", "bool", False,
               flag="--vision-preview", help="Show the live OpenCV YOLO/LiDAR preview window."),
        Option("waypoint_test", "stair waypoint test", "bool", False,
               flag="--stair-waypoint-test",
               help="Docker-free climb self-test: drive straight up the stairs to a waypoint."),
        Option("o2", "O2 payload", "bool", False,
               flag="--with-o2-payload", help="Attach the oxygen-tank payload + weight/fall monitor."),
        Option("step_height", "stair step height (m)", "float", 0.0,
               flag="--stair-step-height", default=0.0, step=0.025, minimum=0.0, maximum=0.4,
               help="Override the commercial preset riser (0 = keep preset 0.150 m)."),
        Option("max_run", "max run time (s)", "int", 900,
               flag="--max-run-time-sec", default=900, step=30, minimum=30, maximum=3600,
               help="Hard cap on the run before the launcher stops the container."),
        Option("keep_logs", "keep run logs", "int", 1,
               flag="--keep-run-logs", default=1, step=1, minimum=1, maximum=50,
               help="How many past run_sim_* folders to retain."),
        Option("skip_build", "skip docker build", "bool", False,
               flag="--skip-build", help="Reuse the existing image; skip the build check entirely."),
        Option("force_build", "force docker build", "bool", False,
               flag="--force-build", help="Rebuild the controller image even if it exists."),
        Option("no_docker", "no docker controller", "bool", False,
               flag="--no-docker-run", help="Run Isaac only; do not start the vision/control container."),
    ]
    return cfg


def _real_config() -> Config:
    cfg = Config(target="real")
    cfg.options = [
        Option("heightscan", "heightscan", "choice", "flat (blind)",
               flag="--lidar", choices=["flat (blind)", "lidar"], default="flat (blind)",
               help="flat = blind proprioceptive walk; lidar = Hesai XT16 heightscan."),
        Option("record", "record rosbag", "bool", False,
               flag="--record", help="rosbag-record the control topics for offline review."),
    ]
    return cfg


def build_command(cfg: Config):
    """Return (display_str, argv, cwd, env, supported) for the chosen config."""
    env = dict(os.environ)
    if cfg.target == "sim":
        flags: List[str] = []
        for opt in cfg.visible():
            if opt.key == "vision_preview":
                # vision preview implies showing recordings windows
                if opt.value:
                    env["SHOW_RECORDINGS"] = "1"
                flags += opt.to_flags()
            else:
                flags += opt.to_flags()
        bat = os.path.join(REPO_ROOT, "sim", "run_sim.bat")
        display = "sim\\run_sim.bat " + " ".join(flags) if flags else "sim\\run_sim.bat"
        env["NO_PAUSE"] = "1"
        argv = ["cmd", "/c", bat] + flags
        return display, argv, os.path.join(REPO_ROOT, "sim"), env, (os.name == "nt")
    # real
    flags = []
    hs = cfg.get("heightscan")
    if hs.value == "lidar":
        flags.append("--lidar")
    if cfg.get("record").value:
        flags.append("--record")
    script = os.path.join(REPO_ROOT, "real", "run_real.sh")
    display = "./real/run_real.sh " + " ".join(flags) if flags else "./real/run_real.sh"
    argv = ["bash", script] + flags
    return display, argv, REPO_ROOT, env, (os.name != "nt")


# ---------------------------------------------------------------------------
# Cross-platform single-key reader
# ---------------------------------------------------------------------------


class KeyReader:
    """Read logical keys ('up','down','left','right','enter','space','tab',char)."""

    def __init__(self):
        self.is_windows = os.name == "nt"
        self._fd = None
        self._old = None

    def __enter__(self):
        if not self.is_windows:
            try:
                if sys.stdin.isatty():
                    import termios
                    import tty
                    self._fd = sys.stdin.fileno()
                    self._old = termios.tcgetattr(self._fd)
                    tty.setcbreak(self._fd)  # leaves ISIG on, so Ctrl-C still works
            except Exception:
                self._old = None
        return self

    def __exit__(self, *exc):
        if not self.is_windows and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        """Non-blocking: return a key if one is buffered, else None."""
        try:
            if self.is_windows:
                import msvcrt
                if msvcrt.kbhit():
                    return self.read()
                return None
            import select
            if not sys.stdin.isatty():
                return None
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return self.read()
        except Exception:
            return None
        return None

    def read(self) -> str:
        if self.is_windows:
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                return {"H": "up", "P": "down", "K": "left", "M": "right",
                        "I": "pgup", "Q": "pgdn", "G": "home", "O": "end"}.get(code, "")
            return self._classify(ch)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(1)
            if nxt == "[":
                code = sys.stdin.read(1)
                if code.isdigit():  # ESC [ <n> ~  (PgUp/PgDn/Home/End)
                    seq = code
                    while True:
                        c = sys.stdin.read(1)
                        if c == "~" or not c:
                            break
                        seq += c
                    return {"5": "pgup", "6": "pgdn", "1": "home", "7": "home",
                            "4": "end", "8": "end"}.get(seq, "")
                return {"A": "up", "B": "down", "C": "right", "D": "left",
                        "H": "home", "F": "end"}.get(code, "")
            return "esc"
        return self._classify(ch)

    @staticmethod
    def _classify(ch: str) -> str:
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\t":
            return "tab"
        if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            return "quit"
        return ch


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


# ---------------------------------------------------------------------------
# Live run + dashboard
# ---------------------------------------------------------------------------


def _reader_thread(proc, log_tail, stages, lock):
    for raw in iter(proc.stdout.readline, ""):
        line = tu.strip_ansi(raw.rstrip("\n"))
        if not line:
            continue
        with lock:
            log_tail.append(line)
            m = _STAGE_RE.match(line)
            if m:
                ts, stage, state, msg = m.groups()
                stages[stage] = {"ts": ts, "state": state, "msg": msg}
    try:
        proc.stdout.close()
    except Exception:
        pass


def run_with_dashboard(cfg: Config, argv, cwd, env, theme: tu.Theme) -> int:
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        print(theme.paint(f"Could not launch: {exc}", fg="error"))
        return 1

    log_tail = collections.deque(maxlen=200)
    stages: dict = {}
    lock = threading.Lock()
    th = threading.Thread(target=_reader_thread, args=(proc, log_tail, stages, lock), daemon=True)
    th.start()

    started = time.time()
    progress_hist: List[float] = []
    x_hist: List[float] = []
    screen = tu.Screen(theme=theme)
    pipeline = _SIM_PIPELINE if cfg.target == "sim" else _REAL_PIPELINE
    interrupted = False
    tele_every = 3  # refresh telemetry from disk every Nth tick (~0.3 s)
    tick = 0
    telemetry: Optional[dict] = None
    run_dir: Optional[str] = None
    tele_view = "robot"
    scroll = 0  # console scrollback offset (lines from the bottom; 0 = follow live)
    PAGE = 10
    try:
        with screen, KeyReader() as keys:
            while True:
                code = proc.poll()
                # interactive full-TUI key handling
                k = keys.poll()
                if k in ("q", "quit", "esc"):
                    interrupted = True
                    _terminate(proc)
                    code = proc.poll()
                elif k == "t":
                    tele_view = TELE_VIEWS[(TELE_VIEWS.index(tele_view) + 1) % len(TELE_VIEWS)]
                elif k in ("up", "k"):
                    scroll += 1
                elif k in ("down", "j"):
                    scroll -= 1
                elif k == "pgup":
                    scroll += PAGE
                elif k == "pgdn":
                    scroll -= PAGE
                elif k in ("home", "g"):
                    scroll = 10 ** 9      # clamped to top below
                elif k in ("end", "G"):
                    scroll = 0            # resume live follow
                with lock:
                    done = sum(1 for s in pipeline if stages.get(s, {}).get("state") in
                               ("ready", "complete", "ok", "pruned", "cleanup", "skipped"))
                    snapshot_stages = dict(stages)
                    tail_copy = list(log_tail)
                scroll = max(0, min(scroll, max(0, len(tail_copy) - 1)))  # keep in range
                progress_hist.append(done)
                if len(progress_hist) > 240:
                    progress_hist = progress_hist[-240:]
                if cfg.target == "sim" and tick % tele_every == 0:
                    run_dir = _find_run_dir()
                    telemetry = _read_fall_diag(run_dir)
                    if telemetry and telemetry.get("x") is not None:
                        x_hist.append(float(telemetry["x"]))
                        if len(x_hist) > 240:
                            x_hist = x_hist[-240:]
                tick += 1
                screen.render(render_dashboard(cfg, snapshot_stages, tail_copy, started,
                                               theme, code, progress_hist,
                                               telemetry=telemetry, x_hist=x_hist, run_dir=run_dir,
                                               tele_view=tele_view, scroll=scroll))
                if code is not None:
                    break
                time.sleep(0.1)
    except KeyboardInterrupt:
        interrupted = True
        _terminate(proc)

    th.join(timeout=1.0)
    rc = proc.poll()
    rc = 130 if interrupted else (rc if rc is not None else 0)
    _print_summary(cfg, rc, theme)
    return rc


def _terminate(proc) -> None:
    try:
        proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        proc.kill()
    except Exception:
        pass


def run_passthrough(argv, cwd, env, theme: tu.Theme, display: str) -> int:
    print(theme.paint("$ ", fg="success", bold=True) + theme.paint(display, fg="secondary"))
    print()
    try:
        return subprocess.call(argv, cwd=cwd, env=env)
    except FileNotFoundError as exc:
        print(theme.paint(f"Could not launch: {exc}", fg="error"))
        return 1


def _print_summary(cfg: Config, rc: int, theme: tu.Theme) -> None:
    W = _width()
    state = "complete" if rc == 0 else "failed"
    msg = "run finished cleanly" if rc == 0 else f"run exited with code {rc}"
    rows = [tu.status_line(time.strftime("%H:%M:%S"), "summary", state, msg, theme)]
    latest = os.path.join(REPO_ROOT, "log", "latest_run.txt")
    if cfg.target == "sim" and os.path.isfile(latest):
        try:
            with open(latest, encoding="utf-8") as fh:
                run_dir = fh.read().strip()
            rows.append(tu.kv("logs", run_dir, theme, 7, "primary", bold_value=False))
            rows.append(tu.kv("read me", os.path.join(run_dir, "00_READ_ME_FIRST.txt"),
                              theme, 7, "muted", bold_value=False))
        except Exception:
            pass
    for ln in tu.panel("done", rows, W, accent="cpu", theme=theme):
        print(ln)


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


if __name__ == "__main__":
    sys.exit(main())
