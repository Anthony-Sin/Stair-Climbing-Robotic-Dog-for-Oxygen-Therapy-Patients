"""ANSI/color primitives for the btop-style terminal UI toolkit.

Palette, box-drawing character sets, color-level detection, and TrueColor →
256 → 16 downconversion. This is the dependency-free foundation the rest of the
toolkit builds on; see ``term_ui`` for the public facade and design notes.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Palette (verbatim from DESIGN.md §2)
# ---------------------------------------------------------------------------

#: Semantic role -> hex, from DESIGN.md §2 "Semantic Roles" (Claude Code theme).
#: Historical btop role KEYS are all preserved so every call site keeps working;
#: only the hues moved to the warm terracotta/hot-pink/lavender palette.
#:   primary   = terracotta brand accent (titles, prompts, stage names, chips)
#:   secondary = hot pink (the `$` command echo + tool-call borders)
#:   accent    = lavender (running/in-progress state, selection markers)
PALETTE: Dict[str, str] = {
    "bg": "#1a1a1a",
    "fg": "#e9e9e9",
    "primary": "#d77757",     # terracotta — Anthropic brand accent
    "secondary": "#fd5db1",   # hot pink — command echo / tool borders
    "accent": "#b1b9f9",      # lavender — permission / running state
    "success": "#4eba65",
    "warning": "#ffc107",
    "error": "#ff6b80",
    "muted": "#888888",
    "surface": "#373737",
    # extra named hues used by gradients / temperatures / older call sites
    "green": "#4eba65",
    "yellow": "#ffc107",
    "red": "#ff6b80",
    "blue": "#b1b9f9",        # remapped to lavender (kept for legacy callers)
    "pink": "#fd5db1",
    "shimmer": "#eb9f7f",     # lighter terracotta (spinner shimmer)
    "auto": "#af87ff",        # purple — auto-accept / YOLO mode
}

#: Panel-border accents. DESIGN.md keeps content mostly unframed, so these are
#: muted, warm tints of the palette rather than the four saturated btop hues.
BOX_ACCENTS: Dict[str, str] = {
    "cpu": "#d77757",   # terracotta (banner / progress)
    "mem": "#9a7d55",   # muted tan/gold
    "net": "#7f86c4",   # muted lavender
    "proc": "#c44d8c",  # muted hot pink
}

#: The green → amber → red-pink severity ramp (meters + sparklines), tracking
#: the DESIGN.md success/warning/error hues.
_GRADIENT_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (0x4E, 0xBA, 0x65)),
    (0.5, (0xFF, 0xC1, 0x07)),
    (1.0, (0xFF, 0x6B, 0x80)),
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
