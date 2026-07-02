"""WARNING HUD — palette, colour helpers, and the shared animation-timing ease.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor): the
acid-yellow hazard palette (ref1 art), the RGB→BGR / colour-mix helpers, and the
optional arcv ``out_cubic`` ease used by the card stack + HUD chrome.  These are
the lowest-level primitives every other warning_* module depends on.
"""
from __future__ import annotations

from typing import Tuple

# arcv is used for its animation-timing model (staggered assemble / eases).  The
# import is optional so the kit still renders (static) if arcv is unavailable.
try:
    from arcv.overlay.anim import out_cubic as _out_cubic
except Exception:  # pragma: no cover
    def _out_cubic(t: float) -> float:
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        f = t - 1.0
        return f * f * f + 1.0


RGB = Tuple[int, int, int]

# ───────────────────────────────── palette ────────────────────────────────────
# Acid-yellow hazard scheme lifted from the ref1 art: bright yellow panels, black
# ink, a single hot-red reserved for danger (target lost / fault).
YELLOW      = (247, 221, 8)      # primary accent / panel fill  (#F7DD08)
YELLOW_HOT  = (255, 238, 60)     # brighter yellow for pulses / lock
YELLOW_DK   = (150, 134, 0)      # dim yellow (inactive strokes, ticks)
BLACK       = (10, 11, 6)        # ink on yellow / black bars
PANEL_BLK   = (18, 19, 12)       # black-card fill
GREY        = (128, 126, 96)     # muted labels on black cards
WHITE       = (238, 238, 224)    # bright readout on black cards
ALERT       = (255, 66, 40)      # danger red — used sparingly (lost / fault)
ALERT_DK    = (120, 26, 16)
FEED_DIM    = 0.34               # how far the non-feed frame is darkened


def bgr(c: RGB) -> Tuple[int, int, int]:
    """RGB → OpenCV BGR."""
    return (int(c[2]), int(c[1]), int(c[0]))


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore
