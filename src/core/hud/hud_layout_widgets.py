"""Reusable arcv draw primitives for the tactical overlay.

The chamfered bracket panel, status LEDs, instrument rows, and the segmented /
bipolar fill bars — the small building blocks the section builders in
:mod:`core.hud.hud_layout_sections` compose.  All operate on an arcv ``Overlay``
(``ov``) and the RGBA-float palette; they never read :class:`HudState` directly.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from core.hud.hud_layout_state import (
    Color, FRAME, GREY, GREY_DIM, RED, RED_HOT, _CUT, _fa,
)


# -------------------------------------------------------------- small utils
def _fmt_m(v, p=2) -> str:
    try:
        return "N/A" if v is None else f"{float(v):.{p}f}M"
    except Exception:
        return "N/A"


def _fill(ov, x0, y0, x1, y1, color: Color) -> None:
    ov.vector.rounded_rect_fill(x0, y0, x1, y1, 0.0, color)


# --------------------------------------------------------- tactical widgets
def _panel(ov, x0, y0, x1, y1, *, s, reveal=1.0, accent: Color = RED,
           title: Optional[str] = None, t: float = 0.0, badge: Optional[Tuple[str, Color]] = None):
    """Chamfered bracket panel — the signature frame element.

    Frame is a closed polyline with the top-right and bottom-left corners cut;
    the two square corners (TL, BR) get accent L-ticks.  Optional header: an
    accent notch + title + hairline underline (+ an optional right-aligned
    badge).  Returns the content-top y."""
    c = _CUT * s
    pts = [(x0, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1),
           (x0 + c, y1), (x0, y1 - c), (x0, y0)]
    ov.vector.polyline(pts, _fa(FRAME, 0.95), 1.3 * s, reveal=reveal)
    tick = 13 * s * min(1.0, reveal * 1.4)
    if tick > 1.0:
        ov.vector.line((x0, y0), (x0 + tick, y0), accent, 2.0 * s)
        ov.vector.line((x0, y0), (x0, y0 + tick), accent, 2.0 * s)
        ov.vector.line((x1, y1), (x1 - tick, y1), accent, 2.0 * s)
        ov.vector.line((x1, y1), (x1, y1 - tick), accent, 2.0 * s)
    cy = y0 + 9 * s
    if title and reveal > 0.5:
        hh = 10.5 * s
        _fill(ov, x0 + 6 * s, y0 + 7 * s, x0 + 9.5 * s, y0 + 7 * s + hh, accent)
        ov.text.text(title, x0 + 14 * s, y0 + 6 * s, hh, accent, align="left")
        if badge is not None:
            ov.text.text(badge[0], x1 - 8 * s, y0 + 7 * s, 9.0 * s, badge[1], align="right")
        uy = y0 + 7 * s + hh + 6 * s
        ov.vector.line((x0 + 6 * s, uy), (x1 - 6 * s, uy), _fa(FRAME, 0.8), 1.0 * s, reveal=reveal)
        cy = uy + 9 * s
    return cy


def _led(ov, x, y, r, state: str, s, t) -> None:
    """Status pip: filled accent=active, blinking hot=fault, grey ring=offline."""
    if state == "ok":
        ov.vector.disc(x, y, r, RED)
        ov.vector.ring(x, y, r + 2.0 * s, _fa(RED, 0.5), 1.0 * s)
    elif state == "fault":
        blink = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 8.0))
        ov.vector.disc(x, y, r, _fa(RED_HOT, blink))
    else:
        ov.vector.ring(x, y, r, _fa(GREY_DIM, 0.9), 1.2 * s)


def _row(ov, x0, x1, y, label, value, vcol: Color, s, *, led: Optional[str] = None,
         t: float = 0.0, vsize=11.5, reveal=1.0) -> None:
    """One instrument row: label left (grey), value right (vcol).  Optional LED."""
    if reveal <= 0.0:
        return
    lx = x0
    if led is not None:
        _led(ov, x0 + 3.5 * s, y + 5.0 * s, 3.0 * s, led, s, t)
        lx = x0 + 11 * s
    ov.text.text(label, lx, y + 1.0 * s, 9.0 * s, GREY_DIM, align="left")
    ov.text.text(value, x1, y - 1.0 * s, vsize * s, vcol, align="right",
                 mode="typeon", t=t, progress=reveal)


def _bar(ov, x0, y, w, frac: float, color: Color, s, *, segs=14, h=4.0) -> None:
    """Segmented horizontal fill bar (0..1)."""
    frac = max(0.0, min(1.0, frac))
    gap = 1.6 * s
    seg_w = (w - gap * (segs - 1)) / segs
    lit = int(round(frac * segs))
    for i in range(segs):
        sx = x0 + i * (seg_w + gap)
        col = color if i < lit else _fa(FRAME, 0.55)
        _fill(ov, sx, y, sx + seg_w, y + h * s, col)


def _bipolar(ov, cx, y, half_w, value: float, scale: float, color: Color, s, *, h=4.0) -> None:
    """Centre-anchored bipolar bar (signed command)."""
    ov.vector.line((cx, y - 2 * s), (cx, y + h * s + 2 * s), _fa(GREY, 0.7), 1.0 * s)
    bw = max(-half_w, min(half_w, value * scale))
    if bw >= 0:
        _fill(ov, cx, y, cx + bw, y + h * s, color)
    else:
        _fill(ov, cx + bw, y, cx, y + h * s, color)
