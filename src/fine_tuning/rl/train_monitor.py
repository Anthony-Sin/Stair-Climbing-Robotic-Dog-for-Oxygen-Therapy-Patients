r"""Live terminal monitor for the blind-RL stair fine-tune.

Watches the rsl_rl TensorBoard event files as training runs and draws a colored dashboard
so you can SEE at a glance whether the climb is progressing or STALLING -- the hero graph is
``Curriculum/terrain_levels`` (the metric that plateaued at 3.74 last time; the real 0.15 m
step is ~level 6). No TensorBoard/torch needed -- it parses the event files directly.

    py -3.11 src/fine_tuning/rl/train_monitor.py --logdir <logs/rsl_rl/unitree_go2_rough>
    # or, alongside a local run:  src\fine_tuning\rl\watch_training.bat

Styling follows docs/DESIGN.md (Claude Code TUI palette): terracotta headers, hot-pink panel
borders, green/amber/red status, muted captions -- COLORS ONLY, none of the chatbot chrome.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------ DESIGN.md palette (truecolor)
def _fg(hexc: str) -> str:
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"\x1b[38;2;{r};{g};{b}m"

RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
FG        = _fg("#ffffff")   # foreground white
TERRA     = _fg("#d77757")   # primary terracotta (headers, the terrain line)
SHIMMER   = _fg("#eb9f7f")   # lighter terracotta
PINK      = _fg("#fd5db1")   # hot pink (panel borders)
LAV       = _fg("#b1b9f9")   # lavender (accent values)
GREEN     = _fg("#4eba65")   # success / climbing
AMBER     = _fg("#ffc107")   # warning / plateau
REDPINK   = _fg("#ff6b80")   # error / demoting
MUTED     = _fg("#888888")   # captions, axes
SUBTLE    = _fg("#505050")   # separators, empty plot cells

def c(s: str, color: str, bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{color}{s}{RESET}"


def _enable_vt() -> None:
    """ANSI escapes + UTF-8 output on the classic Windows console (no-op / best-effort elsewhere)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # box/block glyphs need UTF-8, not cp1252
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)  # PROCESSED | WRAP | VIRTUAL_TERMINAL
        except Exception:
            pass


# ------------------------------------------------------------------ tfevents reader (stdlib only)
def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = res = 0
    while True:
        b = buf[i]; i += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, i
        shift += 7

def _fields(buf: bytes):
    i, n = 0, len(buf)
    while i < n:
        try:
            key, i = _read_varint(buf, i)
        except IndexError:
            return
        fn, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _read_varint(buf, i); yield fn, wt, val
        elif wt == 1:
            yield fn, wt, buf[i:i+8]; i += 8
        elif wt == 2:
            ln, i = _read_varint(buf, i); yield fn, wt, buf[i:i+ln]; i += ln
        elif wt == 5:
            yield fn, wt, buf[i:i+4]; i += 4
        else:
            return

def _parse_event(buf: bytes) -> Tuple[Optional[int], List[Tuple[str, float]]]:
    step, summary = None, None
    for fn, wt, val in _fields(buf):
        if fn == 2 and wt == 0:
            step = val
        elif fn == 5 and wt == 2:
            summary = val
    out: List[Tuple[str, float]] = []
    if summary is not None:
        for fn, wt, val in _fields(summary):
            if fn == 1 and wt == 2:  # repeated Summary.Value
                tag = sv = None
                for vfn, vwt, vval in _fields(val):
                    if vfn == 1 and vwt == 2:
                        tag = vval.decode("utf-8", "ignore")
                    elif vfn == 2 and vwt == 5:
                        sv = struct.unpack("<f", vval)[0]
                if tag is not None and sv is not None:
                    out.append((tag, sv))
    return step, out

def _iter_records(data: bytes):
    i, n = 0, len(data)
    while i + 12 <= n:
        length = struct.unpack("<Q", data[i:i+8])[0]; i += 8
        i += 4  # length CRC
        if i + length + 4 > n:
            break
        yield data[i:i+length]; i += length + 4

def load_scalars(paths: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    tags: Dict[str, List[Tuple[int, float]]] = {}
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        for rec in _iter_records(data):
            try:
                step, vals = _parse_event(rec)
            except Exception:
                continue
            if step is None:
                continue
            for tag, v in vals:
                tags.setdefault(tag, []).append((step, v))
    for t in tags:
        tags[t].sort(key=lambda x: x[0])
    return tags


def find_event_files(logdir: str) -> List[str]:
    """Newest run dir's event files under ``logdir`` (recursive)."""
    files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return []
    newest = max(files, key=os.path.getmtime)
    d = os.path.dirname(newest)
    return sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))


# ------------------------------------------------------------------ analysis
def slope_per_iter(pts: List[Tuple[int, float]], window: int = 400) -> Optional[float]:
    """Least-squares slope (value change per iteration) over the last ``window`` iters."""
    if len(pts) < 5:
        return None
    x_last = pts[-1][0]
    recent = [(s, v) for s, v in pts if s >= x_last - window]
    if len(recent) < 5:
        recent = pts[-min(len(pts), 30):]
    xs = [p[0] for p in recent]; ys = [p[1] for p in recent]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

TARGET_LEVEL = 6.0   # ~0.15 m real step (difficulty (0.15-0.05)/0.15 ~ 0.67 -> level ~6)

def stall_status(terrain: List[Tuple[int, float]], ep_len: List[Tuple[int, float]]):
    """Return (icon, label, color, detail) describing whether the climb is progressing."""
    if not terrain or len(terrain) < 5:
        return ("*", "WARMING UP", MUTED, "waiting for enough iterations to judge the trend")
    lvl = terrain[-1][1]
    sl = slope_per_iter(terrain) or 0.0
    per500 = sl * 500.0
    timeouts = bool(ep_len) and ep_len[-1][1] >= 990
    to_note = "  ·  100% timeouts (cautious, not falling)" if timeouts else ""
    if sl > 2e-3:
        return ("v", "CLIMBING", GREEN,
                f"terrain level {lvl:.2f} / {TARGET_LEVEL:.0f} target  ·  +{per500:.2f} lvl/500it{to_note}")
    if sl < -2e-3:
        return ("x", "STALLING / DEMOTING", REDPINK,
                f"terrain level {lvl:.2f} FALLING ({per500:.2f} lvl/500it) -- the classic stall{to_note}")
    return ("!", "PLATEAU", AMBER,
            f"terrain level {lvl:.2f} / {TARGET_LEVEL:.0f} target  ·  flat ({per500:+.2f} lvl/500it){to_note}")


# ------------------------------------------------------------------ rendering
BLOCKS = "▁▂▃▄▅▆▇█"  # ▁▂▃▄▅▆▇█

def sparkline(pts: List[Tuple[int, float]], n: int = 44) -> str:
    if not pts:
        return c("(no data)", SUBTLE)
    vals = [v for _, v in pts[-n:]]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return c(BLOCKS[0] * len(vals), MUTED)
    return "".join(BLOCKS[int((v - lo) / (hi - lo) * (len(BLOCKS) - 1))] for v in vals)

def render_chart(pts: List[Tuple[int, float]], width: int, height: int,
                 target: Optional[float] = None) -> List[str]:
    if not pts:
        return [c("   (waiting for training to emit data...)", MUTED)]
    steps = [p[0] for p in pts]
    x0, x1 = steps[0], steps[-1]
    span = max(1, x1 - x0)
    vmax = max([v for _, v in pts] + ([target] if target else [0]))
    ymax = max(1.0, vmax * 1.15)
    ymin = 0.0

    # one value per column (mean of samples in the column's step bucket; else nearest)
    cols: List[float] = []
    for k in range(width):
        s_lo = x0 + span * k / width
        s_hi = x0 + span * (k + 1) / width
        bucket = [v for s, v in pts if s_lo <= s < s_hi]
        cols.append(sum(bucket) / len(bucket) if bucket
                    else min(pts, key=lambda p: abs(p[0] - (s_lo + s_hi) / 2))[1])

    def row_of(v: float) -> int:
        frac = (v - ymin) / (ymax - ymin)
        return max(0, min(height - 1, int(round((height - 1) * (1 - frac)))))

    grid = [[" "] * width for _ in range(height)]
    colr = [[None] * width for _ in range(height)]  # type: ignore
    if target is not None:
        tr = row_of(target)
        for k in range(width):
            grid[tr][k] = "┈"; colr[tr][k] = GREEN     # ┈ dashed target line
    prev = None
    for k, v in enumerate(cols):
        r = row_of(v)
        grid[r][k] = "●"; colr[r][k] = TERRA           # ● data point
        if prev is not None and prev != r:
            d = 1 if r > prev else -1
            for rr in range(prev + d, r, d):
                if grid[rr][k] == " " or grid[rr][k] == "┈":
                    grid[rr][k] = "│"; colr[rr][k] = SHIMMER   # │ connector
        prev = r

    lines: List[str] = []
    for r in range(height):
        yval = ymax - (ymax - ymin) * r / (height - 1)
        row = "".join(c(grid[r][k], colr[r][k]) if colr[r][k] else (grid[r][k])
                      for k in range(width))
        lines.append(c(f"{yval:4.1f} ", MUTED) + c("│", PINK) + row)
    lines.append(c("      └" + "─" * width, MUTED))
    xr = f"iter {x0}".ljust(width - len(str(x1)) - 2) + f"{x1}"
    lines.append(c("      " + xr, MUTED))
    return lines

def hpanel(title: str, body_lines: List[str], width: int) -> List[str]:
    """A hot-pink bordered panel, exactly ``width`` cols (DESIGN.md tool-call block style).

    Interior content area is ``width - 4`` (one border + one space on each side). Content
    longer than that is truncated (visible chars) so a border never gets pushed out of line.
    """
    inner = width - 4
    head_txt = f"┌─ {title} "
    head = c("┌─ ", PINK) + c(title, TERRA, bold=True) + c(" ", PINK) \
        + c("─" * max(0, width - _vis_len(head_txt) - 1) + "┐", PINK)
    out = [head]
    for ln in body_lines:
        vis = _vis_len(ln)
        if vis > inner:
            ln = _truncate_vis(ln, inner); vis = inner
        out.append(c("│", PINK) + " " + ln + " " * (inner - vis) + " " + c("│", PINK))
    out.append(c("└" + "─" * (width - 2) + "┘", PINK))
    return out

def _truncate_vis(s: str, limit: int) -> str:
    """Truncate a string to ``limit`` VISIBLE chars, preserving ANSI codes already in it."""
    import re
    out, vis = [], 0
    for tok in re.split(r"(\x1b\[[0-9;]*m)", s):
        if tok.startswith("\x1b"):
            out.append(tok); continue
        for ch in tok:
            if vis >= limit:
                out.append(RESET); return "".join(out)
            out.append(ch); vis += 1
    return "".join(out)

def _strip(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

def _vis_len(s: str) -> int:
    return len(_strip(s))


def render_frame(tags: Dict[str, List[Tuple[int, float]]], run_name: str,
                 width: int, chart_w: int, chart_h: int, started: float) -> str:
    terrain = tags.get("Curriculum/terrain_levels", [])
    reward  = tags.get("Train/mean_reward", [])
    ep_len  = tags.get("Train/mean_episode_length", [])
    ascent  = tags.get("Episode_Reward/ascent_rate", [])
    track   = tags.get("Episode_Reward/track_lin_vel_xy_exp", [])
    noise   = tags.get("Policy/mean_noise_std", [])
    it = terrain[-1][0] if terrain else (reward[-1][0] if reward else 0)

    icon, label, scolor, detail = stall_status(terrain, ep_len)

    lines: List[str] = []
    lines.append("")
    lines.append("  " + c("O2 STAIR FINE-TUNE", TERRA, bold=True) + c("  ·  ", SUBTLE)
                 + c("LIVE MONITOR", FG, bold=True) + c("   ·  the graph to watch for a stall", MUTED))
    lines.append("")

    # --- STATUS banner (green climbing / amber plateau / red stalling) ---
    banner = "  " + c(f"[{icon}] {label}", scolor, bold=True) + c("   " + detail, scolor)
    lines.append(banner)
    lines.append("")

    # --- hero chart: terrain_levels vs target ---
    chart = render_chart(terrain, chart_w, chart_h, target=TARGET_LEVEL)
    lines += hpanel("Curriculum/terrain_levels   ┈ green = 0.15 m step (level ~6)", chart, width)
    lines.append("")

    # --- sparkline strip for the other levers ---
    spark_n = max(12, width - 40)
    def strip(name: str, series: List[Tuple[int, float]], good_up: bool = True) -> str:
        if not series:
            return c(name.ljust(20), MUTED) + c("(no data yet)", SUBTLE)
        last = series[-1][1]
        sl = slope_per_iter(series) or 0.0
        arrow = "↑" if sl > 0 else ("↓" if sl < 0 else "→")
        acol = GREEN if (sl > 0) == good_up and abs(sl) > 1e-6 else (AMBER if abs(sl) <= 1e-6 else REDPINK)
        return (c(name.ljust(20), FG) + c(sparkline(series, spark_n), TERRA)
                + "  " + c(f"{last:8.3f}", LAV) + " " + c(arrow, acol))

    body = [
        strip("mean_reward", reward),
        strip("ascent_rate (climb pay)", ascent),
        strip("track_lin_vel (fwd pay)", track, good_up=False),
        strip("episode_length", ep_len, good_up=False),
        strip("noise_std (exploration)", noise, good_up=False),
    ]
    lines += hpanel("reward / exploration levers   ↑ rising  ↓ falling", body, width)

    # --- status bar (DESIGN.md muted bottom line) ---
    elapsed = int(time.time() - started)
    bar = (f"  run {run_name}  ·  iter {it}  ·  watching {elapsed//60}m{elapsed%60:02d}s"
           f"  ·  refresh {time.strftime('%H:%M:%S')}  ·  Ctrl-C to quit")
    lines.append("")
    lines.append(c(bar, MUTED))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live colored monitor for the RL stair fine-tune.")
    ap.add_argument("--logdir", default=os.environ.get("FT_RL_LOGDIR", ""),
                    help="rsl_rl log dir, e.g. ~/robot_lab/logs/rsl_rl/unitree_go2_rough "
                         "(searched recursively for the newest run).")
    ap.add_argument("--interval", type=float, default=3.0, help="refresh seconds (default 3).")
    ap.add_argument("--width", type=int, default=72, help="panel width (default 72).")
    ap.add_argument("--once", action="store_true", help="render one frame and exit (for testing).")
    args = ap.parse_args(argv)

    _enable_vt()
    logdir = os.path.expanduser(args.logdir) if args.logdir else ""
    if not logdir:
        print(c("  No --logdir given. Pass the rsl_rl log dir "
                "(e.g. ~/robot_lab/logs/rsl_rl/unitree_go2_rough).", AMBER))
        return 2

    chart_w = max(30, args.width - 10)
    chart_h = 13
    started = time.time()
    HIDE, SHOW = "\x1b[?25l", "\x1b[?25h"
    sys.stdout.write(HIDE)
    try:
        while True:
            files = find_event_files(logdir)
            if not files:
                sys.stdout.write("\x1b[H\x1b[2J")
                print(c(f"  Waiting for training to start... (no event files under {logdir})", MUTED))
                print(c("  This is normal during the one-time install, or before iter 1.", SUBTLE))
                if args.once:
                    return 0
                time.sleep(args.interval)
                continue
            run_name = os.path.basename(os.path.dirname(files[-1]))
            tags = load_scalars(files)
            frame = render_frame(tags, run_name, args.width, chart_w, chart_h, started)
            sys.stdout.write("\x1b[H\x1b[2J")   # home + clear
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        sys.stdout.write(SHOW + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
