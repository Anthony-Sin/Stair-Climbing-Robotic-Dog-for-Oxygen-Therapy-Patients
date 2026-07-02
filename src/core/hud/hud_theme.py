"""ARCV/MUTEK-scanner HUD design tokens + colour helpers.

The shared palette (single signal-red accent-core on near-pure black), the
back-compat colour aliases older call-sites/tests reference, and the tiny
colour utilities (:func:`bgr`, :func:`dim`, :func:`hex4`) that the rest of the
raster ``core/hud`` primitives build on.
"""
from typing import Tuple


# ───────────────────────────── Design tokens (RGB) ────────────────────────────
# Reference look ("ARCV / MUTEK" scanner): white-grey type on near-pure black with
# a SINGLE saturated red accent-core.  Red now carries all focus / active / warning
# meaning (connection lines, headers, the locked target); everything else is neutral
# off-white / grey / hairline.  Straight lines, open brackets, radius 0.
BG_BASE   = (7, 7, 8)        # #070708  near-pure black base
BG_PANEL  = (16, 15, 16)     # #100F10  raised panel fill
HAIRLINE  = (58, 56, 58)     # #3A383A  dividers / inactive borders (neutral grey)
TEXT      = (232, 232, 230)  # #E8E8E6  primary readout text (off-white)
DIM       = (146, 148, 150)  # #929496  secondary / labels / meta (neutral grey)
ACCENT    = (228, 54, 42)    # #E4362A  signal-red accent-core (active / focus / lock)
ALERT     = (255, 74, 58)    # #FF4A3A  hotter red — critical danger (blinks/fills)
TEXTURE   = (26, 18, 18)     # #1A1212  faint red-black dot-matrix fill
ACCENT_DK = (138, 42, 34)    # #8A2A22  dimmed red (connection lines, ticks, decode tail)
INK       = (0, 0, 0)        # text outline / shadow halo

# ── Back-compat aliases ────────────────────────────────────────────────────────
# Older call-sites + tests referenced the previous "Tac-Amber" names; map them onto
# the new two-colour system so nothing imports a missing symbol and the whole HUD
# reads in one accent.  AMBER/SUCCESS → cyan accent; SLATE/MUTED → neutral dim;
# WARNING/ERROR → alert red (the doc has no separate caution colour).
FG = TEXT
AMBER = ACCENT
SLATE = DIM
SUCCESS = ACCENT
BRIGHT_SLATE = TEXT
WARNING = ALERT
ERROR = ALERT
MUTED = DIM
SURFACE = HAIRLINE
DEEP_AMBER = ACCENT_DK
BG_FILL = BG_BASE
HUD_TEXT = HUD_FG = TEXT
HUD_AMBER = ACCENT
HUD_SLATE = DIM
HUD_MUTED = DIM
HUD_SUCCESS = ACCENT
HUD_WARNING = HUD_ALERT = HUD_ERROR = ALERT
HUD_SURFACE = HAIRLINE
HUD_INK = INK


def bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """RGB → OpenCV BGR."""
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def dim(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    """Scale a colour toward black by factor ``f`` (0..1)."""
    return (int(rgb[0] * f), int(rgb[1] * f), int(rgb[2] * f))


def hex4(value) -> str:
    """A 4-hex-digit code block cell (real value rendered as device-style hex)."""
    try:
        return f"{int(round(float(value))) & 0xFFFF:04X}"
    except Exception:
        return "----"
