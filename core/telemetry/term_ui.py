"""btop-style terminal UI toolkit (pure stdlib, cross-platform).

A small rendering library that draws the btop aesthetic described in the repo's
``DESIGN.md``: pure-black dashboard, rounded boxes with notched/colored titles
(``╭─┐ title ┌────╮``), TrueColor gradient meters (green → yellow → red), and
block sparklines. It has **no third-party dependencies** so the exact same code
styles the Windows/sim launcher, the Linux/real launcher and the shared
``core/main.py`` controller banner.

Design goals
------------
* **Degrade, never break.** When the stream is not a TTY, ``NO_COLOR`` is set,
  ``TERM=dumb``, or the user passes ``--no-color``, every helper falls back to
  plain ASCII with zero escape codes. The box characters fall back to ``+-|``.
* **TrueColor → 256 → 16.** Colors are emitted at the richest level the
  terminal advertises and converted down automatically, so the palette looks
  right in Windows Terminal, modern conhost, and a plain Linux console alike.
* **Measure visible width.** All padding/truncation ignores ANSI escapes and
  the zero-width box notch, so panels line up even when colored.

The public surface is intentionally tiny:

    theme = Theme.detect()              # or Theme.detect(force_color=False)
    print(theme.paint("hi", "green", bold=True))
    for line in panel("config", rows, width=48, accent="net", theme=theme):
        print(line)
    print(meter(0.62, 20, theme))
    print(sparkline([1, 3, 5, 8, 5, 2], theme))
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Palette (verbatim from DESIGN.md §2)
# ---------------------------------------------------------------------------

#: Semantic role -> hex. These are the btop "Semantic Roles" table.
PALETTE: Dict[str, str] = {
    "bg": "#000000",
    "fg": "#cccccc",
    "primary": "#eeeeee",
    "secondary": "#77ca9b",
    "accent": "#dc4c4c",
    "success": "#77ca9b",
    "warning": "#cbc06c",
    "error": "#dc4c4c",
    "muted": "#555555",
    "surface": "#111111",
    # extra named hues used by gradients / temperatures
    "green": "#77ca9b",
    "yellow": "#cbc06c",
    "red": "#dc4c4c",
    "blue": "#4897d4",
    "pink": "#ff40b6",
}

#: Box-specific accents (DESIGN.md §2 "Box-Specific Accents"). The four btop
#: hues are reused as the accent vocabulary for this project's panels.
BOX_ACCENTS: Dict[str, str] = {
    "cpu": "#556d59",   # muted green
    "mem": "#6c6c4b",   # olive
    "net": "#5c588d",   # muted purple
    "proc": "#805252",  # muted red
}

#: The green → yellow → red severity ramp (DESIGN.md §2 "Gradient Ramps").
_GRADIENT_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (0x77, 0xCA, 0x9B)),
    (0.5, (0xCB, 0xC0, 0x6C)),
    (1.0, (0xDC, 0x4C, 0x4C)),
]

_ESC = "\x1b"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# ---------------------------------------------------------------------------
# Box-drawing character sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BoxChars:
    tl: str
    tr: str
    bl: str
    br: str
    h: str
    v: str
    notch_l: str  # before the title  (btop "inverted" corner ┐)
    notch_r: str  # after the title   (btop "inverted" corner ┌)
    tee_l: str
    tee_r: str


_UNICODE_BOX = _BoxChars(
    tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│",
    notch_l="┐", notch_r="┌", tee_l="┤", tee_r="├",
)
_ASCII_BOX = _BoxChars(
    tl="+", tr="+", bl="+", br="+", h="-", v="|",
    notch_l="]", notch_r="[", tee_l="+", tee_r="+",
)

#: Block sparkline ramp (low → high). Width-1 glyphs, render everywhere.
_BLOCK_RAMP = " ▁▂▃▄▅▆▇█"
_TTY_RAMP = " ░▒▓█"

# 16-color xterm RGB table (basic + bright) for nearest-match downconversion.
_ANSI16 = [
    (0x00, 0x00, 0x00), (0x80, 0x00, 0x00), (0x00, 0x80, 0x00), (0x80, 0x80, 0x00),
    (0x00, 0x00, 0x80), (0x80, 0x00, 0x80), (0x00, 0x80, 0x80), (0xC0, 0xC0, 0xC0),
    (0x80, 0x80, 0x80), (0xFF, 0x00, 0x00), (0x00, 0xFF, 0x00), (0xFF, 0xFF, 0x00),
    (0x00, 0x00, 0xFF), (0xFF, 0x00, 0xFF), (0x00, 0xFF, 0xFF), (0xFF, 0xFF, 0xFF),
]


# ---------------------------------------------------------------------------
# Color level detection
# ---------------------------------------------------------------------------

NONE, ANSI16, ANSI256, TRUECOLOR = 0, 1, 2, 3


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def detect_color_level(stream=None, force: Optional[bool] = None) -> int:
    """Return the supported color level for *stream*.

    ``force=True`` assumes TrueColor regardless of TTY (useful when piping into
    a file that will be viewed in a color-aware pager, or for ``--demo``).
    ``force=False`` disables color outright.
    """
    if force is True:
        return TRUECOLOR
    if force is False:
        return NONE
    if os.environ.get("NO_COLOR") is not None:
        return NONE
    if os.environ.get("CLICOLOR_FORCE") not in (None, "0"):
        return TRUECOLOR
    stream = stream if stream is not None else sys.stdout
    try:
        if not stream.isatty():
            return NONE
    except Exception:
        return NONE
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return NONE
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return TRUECOLOR
    # Windows Terminal / modern conhost (Win10 1703+) and most modern emulators
    # speak 24-bit even without COLORTERM set.
    if os.name == "nt":
        return TRUECOLOR
    if os.environ.get("WT_SESSION") or os.environ.get("KONSOLE_VERSION"):
        return TRUECOLOR
    if "256" in term:
        return ANSI256
    if term:
        return ANSI16
    return NONE


def detect_unicode(stream=None, force: Optional[bool] = None) -> bool:
    """Whether box-drawing/block glyphs are safe to emit."""
    if force is not None:
        return force
    enc = ""
    stream = stream if stream is not None else sys.stdout
    try:
        enc = (stream.encoding or "").lower()
    except Exception:
        enc = ""
    if "utf" in enc:
        return True
    # Windows code page 65001 == UTF-8; otherwise assume the box chars render
    # (modern Windows Terminal does) but fall back when we clearly can't encode.
    if os.name == "nt":
        return True
    return "utf" in (os.environ.get("LANG", "") + os.environ.get("LC_ALL", "")).lower()


def _enable_windows_vt() -> None:
    """Best-effort: turn on ANSI escape processing for legacy conhost."""
    if os.name != "nt":
        return
    try:  # pragma: no cover - platform specific
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # STDOUT, STDERR
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Color downconversion
# ---------------------------------------------------------------------------


def _rgb_to_256(r: int, g: int, b: int) -> int:
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + round((r - 8) / 247 * 24)
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def _rgb_to_16(r: int, g: int, b: int) -> int:
    best, best_d = 7, None
    for idx, (cr, cg, cb) in enumerate(_ANSI16):
        d = (cr - r) ** 2 + (cg - g) ** 2 + (cb - b) ** 2
        if best_d is None or d < best_d:
            best, best_d = idx, d
    return best


def _fg_code(rgb: Tuple[int, int, int], level: int) -> str:
    r, g, b = rgb
    if level >= TRUECOLOR:
        return f"{_ESC}[38;2;{r};{g};{b}m"
    if level == ANSI256:
        return f"{_ESC}[38;5;{_rgb_to_256(r, g, b)}m"
    idx = _rgb_to_16(r, g, b)
    return f"{_ESC}[{30 + idx}m" if idx < 8 else f"{_ESC}[{90 + idx - 8}m"


def _bg_code(rgb: Tuple[int, int, int], level: int) -> str:
    r, g, b = rgb
    if level >= TRUECOLOR:
        return f"{_ESC}[48;2;{r};{g};{b}m"
    if level == ANSI256:
        return f"{_ESC}[48;5;{_rgb_to_256(r, g, b)}m"
    idx = _rgb_to_16(r, g, b)
    return f"{_ESC}[{40 + idx}m" if idx < 8 else f"{_ESC}[{100 + idx - 8}m"


def gradient_rgb(frac: float) -> Tuple[int, int, int]:
    """Sample the green → yellow → red severity ramp at *frac* in [0, 1]."""
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    for i in range(len(_GRADIENT_STOPS) - 1):
        f0, c0 = _GRADIENT_STOPS[i]
        f1, c1 = _GRADIENT_STOPS[i + 1]
        if frac <= f1:
            t = 0.0 if f1 == f0 else (frac - f0) / (f1 - f0)
            return tuple(round(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))  # type: ignore[return-value]
    return _GRADIENT_STOPS[-1][1]


# ---------------------------------------------------------------------------
# Width helpers (ANSI-aware)
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def truncate(text: str, width: int) -> str:
    """Truncate to *width* visible columns, preserving a trailing reset."""
    if visible_len(text) <= width:
        return text
    out, count = [], 0
    i = 0
    has_ansi = False
    while i < len(text) and count < width:
        m = _ANSI_RE.match(text, i)
        if m:
            out.append(m.group())
            has_ansi = True
            i = m.end()
            continue
        out.append(text[i])
        count += 1
        i += 1
    if width >= 1:
        out[-1] = "…"
    if has_ansi:
        out.append(f"{_ESC}[0m")
    return "".join(out)


def pad(text: str, width: int, align: str = "left") -> str:
    """Pad (or truncate) *text* to *width* visible columns."""
    vis = visible_len(text)
    if vis > width:
        return truncate(text, width)
    space = width - vis
    if align == "right":
        return " " * space + text
    if align == "center":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return text + " " * space


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass
class Theme:
    """Resolved color/glyph capabilities + paint helpers."""

    level: int = TRUECOLOR
    unicode: bool = True

    @classmethod
    def detect(cls, stream=None, force_color: Optional[bool] = None,
               force_unicode: Optional[bool] = None) -> "Theme":
        level = detect_color_level(stream, force_color)
        if level > NONE:
            _enable_windows_vt()
        return cls(level=level, unicode=detect_unicode(stream, force_unicode))

    # -- low level -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.level > NONE

    def box(self) -> _BoxChars:
        return _UNICODE_BOX if self.unicode else _ASCII_BOX

    def _resolve(self, color) -> Optional[Tuple[int, int, int]]:
        if color is None:
            return None
        if isinstance(color, tuple):
            return color
        if isinstance(color, str):
            if color.startswith("#"):
                return _hex_to_rgb(color)
            if color in PALETTE:
                return _hex_to_rgb(PALETTE[color])
            if color in BOX_ACCENTS:
                return _hex_to_rgb(BOX_ACCENTS[color])
        return None

    def paint(self, text: str, fg=None, bg=None, bold: bool = False,
              dim: bool = False) -> str:
        """Wrap *text* in ANSI styling (no-op when color is disabled)."""
        if not self.enabled:
            return text
        codes = []
        if bold:
            codes.append(f"{_ESC}[1m")
        if dim:
            codes.append(f"{_ESC}[2m")
        fg_rgb = self._resolve(fg)
        if fg_rgb is not None:
            codes.append(_fg_code(fg_rgb, self.level))
        bg_rgb = self._resolve(bg)
        if bg_rgb is not None:
            codes.append(_bg_code(bg_rgb, self.level))
        if not codes:
            return text
        return "".join(codes) + text + f"{_ESC}[0m"

    def gradient(self, text: str, frac: float, bold: bool = False) -> str:
        return self.paint(text, fg=gradient_rgb(frac), bold=bold)


#: A module-level theme created lazily on first use.
_DEFAULT: Optional[Theme] = None


def default_theme() -> Theme:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Theme.detect()
    return _DEFAULT


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def panel(title: str, lines: Sequence[str], width: int = 60,
          accent: str = "cpu", theme: Optional[Theme] = None,
          footer: Optional[str] = None) -> List[str]:
    """Render a btop box with a notched, colored title.

    ``╭─┐ title ┌──────────────╮``
    ``│  ...content...          │``
    ``╰──────────────────────────╯``

    *lines* are already-rendered content rows (may contain ANSI); each is
    padded/truncated to the interior width. *width* is the total outer width.
    """
    theme = theme or default_theme()
    bc = theme.box()
    accent_rgb = theme._resolve(accent) or theme._resolve("muted")
    width = max(width, visible_len(title) + 8)
    inner = width - 2

    def border(s: str) -> str:
        return theme.paint(s, fg=accent_rgb)

    # Top: ╭─┐ title ┌───╮
    title_txt = theme.paint(f" {title} ", fg="primary", bold=True)
    left_vis = 3 + visible_len(title_txt)  # tl + h + notch_l + title + notch_r
    fill = max(0, width - left_vis - 2)
    top = border(bc.tl + bc.h + bc.notch_l) + title_txt + border(
        bc.notch_r + bc.h * fill + bc.tr)

    out = [top]
    for ln in lines:
        body = pad(" " + ln, inner)
        out.append(border(bc.v) + body + border(bc.v))

    if footer:
        foot_txt = theme.paint(f" {footer} ", fg="muted")
        ffill = max(0, width - 3 - visible_len(foot_txt) - 2)
        out.append(border(bc.bl + bc.h * ffill + bc.notch_l) + foot_txt
                   + border(bc.notch_r + bc.h + bc.br))
    else:
        out.append(border(bc.bl + bc.h * inner + bc.br))
    return out


def block_width(lines: Sequence[str]) -> int:
    """Max visible width of a block of (possibly ANSI-styled) lines."""
    return max((visible_len(ln) for ln in lines), default=0)


def hjoin(blocks: Sequence[Sequence[str]], gap: int = 2) -> List[str]:
    """Place rendered blocks (e.g. panels) side by side into one block.

    This is what turns the stacked-boxes look into a btop dashboard grid: each
    block is padded to its own width and to the tallest block's height, then the
    rows are concatenated with ``gap`` spaces between columns.
    """
    cols = [list(b) for b in blocks if b]
    if not cols:
        return []
    widths = [block_width(b) for b in cols]
    height = max(len(b) for b in cols)
    sep = " " * gap
    out: List[str] = []
    for r in range(height):
        cells = []
        for b, w in zip(cols, widths):
            cells.append(pad(b[r] if r < len(b) else "", w))
        out.append(sep.join(cells))
    return out


def fit_height(lines: List[str], rows: int) -> List[str]:
    """Pad or truncate a block to exactly *rows* lines (dynamic full-height fill)."""
    lines = list(lines)
    if len(lines) >= rows:
        return lines[:rows]
    return lines + [""] * (rows - len(lines))


def kv(label: str, value: str, theme: Optional[Theme] = None,
       label_w: int = 12, value_color="primary", bold_value: bool = True) -> str:
    """A muted ``label`` followed by a colored ``value`` (btop metric row)."""
    theme = theme or default_theme()
    lbl = theme.paint(pad(label, label_w), fg="muted")
    val = theme.paint(value, fg=value_color, bold=bold_value)
    return f"{lbl}{val}"


def meter(frac: float, width: int = 20, theme: Optional[Theme] = None,
          mode: str = "meter") -> str:
    """A gradient-filled meter: ``[■■■■■■░░░░]``-style, colored per cell.

    ``mode``: ``meter`` (■/░), ``bar`` (█/░), or ``tty`` (#/.).
    """
    theme = theme or default_theme()
    width = max(1, width)
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    filled = int(round(frac * width))
    if theme.unicode and mode != "tty":
        full, empty = ("■", "░") if mode == "meter" else ("█", "░")
    else:
        full, empty = "#", "."
    cells = []
    for i in range(width):
        if i < filled:
            pos = i / max(1, width - 1)
            cells.append(theme.paint(full, fg=gradient_rgb(pos)))
        else:
            cells.append(theme.paint(empty, fg="muted"))
    return "".join(cells)


def sparkline(values: Sequence[float], theme: Optional[Theme] = None,
              width: Optional[int] = None, mode: str = "block") -> str:
    """A gradient block sparkline (``▁▂▃▅▇█``) sampled from *values*."""
    theme = theme or default_theme()
    vals = list(values)
    if not vals:
        return ""
    if width and len(vals) > width:
        # keep the most recent `width` samples (btop scrolls left)
        vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    ramp = _BLOCK_RAMP if (theme.unicode and mode == "block") else _TTY_RAMP
    out = []
    for v in vals:
        norm = 0.0 if rng <= 0 else (v - lo) / rng
        idx = min(len(ramp) - 1, int(round(norm * (len(ramp) - 1))))
        out.append(theme.paint(ramp[idx], fg=gradient_rgb(norm)))
    return "".join(out)


# ---------------------------------------------------------------------------
# Status glyphs
# ---------------------------------------------------------------------------

#: launcher stage states -> (glyph, color). ASCII fallbacks chosen by Theme.
_STATE_STYLE = {
    "start": ("▶", "blue"),
    "running": ("▶", "blue"),
    "notice": ("•", "yellow"),
    "warning": ("▲", "warning"),
    "ready": ("✔", "success"),
    "complete": ("✔", "success"),
    "ok": ("✔", "success"),
    "cleanup": ("◦", "muted"),
    "pruned": ("◦", "muted"),
    "skipped": ("–", "muted"),
    "dry-run": ("◦", "muted"),
    "failed": ("✖", "error"),
    "error": ("✖", "error"),
}
_STATE_ASCII = {
    "▶": ">", "•": "*", "▲": "!", "✔": "OK", "◦": "-", "–": "-", "✖": "X",
}


def state_glyph(state: str, theme: Optional[Theme] = None) -> str:
    theme = theme or default_theme()
    glyph, color = _STATE_STYLE.get(state, ("•", "fg"))
    if not theme.unicode:
        glyph = _STATE_ASCII.get(glyph, glyph)
    return theme.paint(glyph, fg=color, bold=True)


def bar(left: str, right: str = "", width: int = 80, theme: Optional[Theme] = None,
        bg="net", fg="bg", bold: bool = True) -> str:
    """A full-width status/title bar: ``left`` text, ``right`` right-justified, one bg.

    Classic TUI chrome (header/footer). Pass plain text; the whole line is painted
    one background color so it reads as a solid bar.
    """
    theme = theme or default_theme()
    avail = max(0, width - visible_len(right) - 1)
    if visible_len(left) > avail:
        left = truncate(left, avail)
    space = max(0, width - visible_len(left) - visible_len(right))
    return theme.paint(left + " " * space + right, fg=fg, bg=bg, bold=bold)


def status_line(timestamp: str, stage: str, state: str, message: str,
                theme: Optional[Theme] = None) -> str:
    """A single colorized launcher status row (used by the live dashboard)."""
    theme = theme or default_theme()
    ts = theme.paint(timestamp, fg="muted")
    gl = state_glyph(state, theme)
    stg = theme.paint(pad(stage, 11), fg="secondary", bold=True)
    _, color = _STATE_STYLE.get(state, ("•", "fg"))
    msg = theme.paint(message, fg=color if state in ("failed", "error") else "fg")
    return f"{ts} {gl} {stg} {msg}"


# ---------------------------------------------------------------------------
# Live screen helper (alt-screen, flicker-free repaint)
# ---------------------------------------------------------------------------


class Screen:
    """Minimal alt-screen controller for in-place dashboard repaints."""

    def __init__(self, stream=None, theme: Optional[Theme] = None):
        self.stream = stream if stream is not None else sys.stdout
        self.theme = theme or default_theme()
        self._active = False

    def __enter__(self) -> "Screen":
        if self.theme.enabled:
            self.stream.write(f"{_ESC}[?1049h{_ESC}[?25l{_ESC}[2J")
            self.stream.flush()
            self._active = True
        return self

    def __exit__(self, *exc) -> None:
        if self._active:
            self.stream.write(f"{_ESC}[?25h{_ESC}[?1049l")
            self.stream.flush()
            self._active = False

    def render(self, lines: Iterable[str]) -> None:
        """Repaint from the home position, clearing each line to EOL."""
        buf = [f"{_ESC}[H"] if self._active else []
        for ln in lines:
            buf.append(ln + (f"{_ESC}[K" if self._active else ""))
            buf.append("\n")
        if self._active:
            buf.append(f"{_ESC}[J")  # clear below
        self.stream.write("".join(buf))
        self.stream.flush()


def hide_cursor(stream=None) -> str:
    return f"{_ESC}[?25l"


def show_cursor(stream=None) -> str:
    return f"{_ESC}[?25h"
