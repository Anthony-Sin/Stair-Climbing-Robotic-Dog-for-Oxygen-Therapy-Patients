"""Command spawn + live-dashboard run loop, passthrough, and summary.

Extracted verbatim from ``launcher.py``: the stdout reader thread, the
``run_with_dashboard`` interactive loop, process termination,
``run_passthrough``, and the end-of-run ``_print_summary``.
"""

from __future__ import annotations

import collections
import os
import subprocess
import threading
import time
from typing import List, Optional

from core.telemetry import term_ui as tu

from launcher_lib.config import (
    Config,
    _REAL_PIPELINE,
    _SIM_PIPELINE,
    _STAGE_RE,
)
from launcher_lib.keyreader import KeyReader
from launcher_lib.paths import REPO_ROOT
from launcher_lib.render import (
    TELE_VIEWS,
    _find_run_dir,
    _read_fall_diag,
    _width,
    render_dashboard,
)


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
    for ln in tu.panel("done", rows, W, accent="success" if rc == 0 else "proc", theme=theme):
        print(ln)
