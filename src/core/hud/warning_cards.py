"""WARNING HUD — the spawn/collapse card system.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor).  The
:class:`CardSpec` description, the :class:`CardStack` column that rolls cards
open/shut and reflows them, and the :func:`_render_card` / :func:`_seg_bar`
renderers that draw one ref1 hazard-panel card to its own opaque tile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.hud.warning_render import Layer, blit, cut_poly, hazard_band
from core.hud.warning_text import text_width
from core.hud.warning_theme import (
    ALERT,
    BLACK,
    GREY,
    PANEL_BLK,
    RGB,
    YELLOW,
    YELLOW_DK,
    YELLOW_HOT,
    _mix,
    _out_cubic,
)


# ──────────────────────────────── card system ─────────────────────────────────
@dataclass
class CardSpec:
    key: str
    title: str
    style: str = "panel"                 # panel | solid | alert
    big: Optional[Tuple[str, RGB]] = None
    rows: List[Tuple[str, str, RGB]] = field(default_factory=list)  # (label, value, value_col)
    bar: Optional[Tuple[float, RGB]] = None
    note: Optional[str] = None
    height: int = 84


class _CardAnim:
    __slots__ = ("spec", "anim")

    def __init__(self, spec: CardSpec) -> None:
        self.spec = spec
        self.anim = 0.0


class CardStack:
    """A column of cards that spawn (roll open) / collapse (roll shut) as the
    active set changes, reflowing so there is no dead space."""

    APPEAR = 0.26   # seconds to open
    VANISH = 0.20   # seconds to close

    def __init__(self, side: str) -> None:
        self.side = side                       # "L" | "R"
        self._cards: Dict[str, _CardAnim] = {}
        self._order: List[str] = []

    def update(self, specs: List[CardSpec], dt: float) -> None:
        want = {s.key: s for s in specs}
        for key, spec in want.items():
            c = self._cards.get(key)
            if c is None:
                c = _CardAnim(spec)
                self._cards[key] = c
                self._order.append(key)
            else:
                c.spec = spec
        for key, c in self._cards.items():
            if key in want:
                c.anim = min(1.0, c.anim + dt / self.APPEAR)
            else:
                c.anim = max(0.0, c.anim - dt / self.VANISH)
        # drop fully-collapsed cards that are no longer wanted
        dead = [k for k, c in self._cards.items() if k not in want and c.anim <= 0.001]
        for k in dead:
            self._cards.pop(k, None)
            self._order.remove(k)
        # keep order: wanted specs define ordering; closing cards keep last position
        wanted_order = [s.key for s in specs]
        self._order = wanted_order + [k for k in self._order if k not in want]

    def draw(self, dst: np.ndarray, x: int, y_top: int, w: int, gap: int,
             t: float) -> None:
        y = y_top
        for key in self._order:
            c = self._cards.get(key)
            if c is None:
                continue
            e = _out_cubic(c.anim)
            layer = _render_card(c.spec, w, t)
            vis = e
            blit(dst, layer, x, y, alpha=min(1.0, c.anim * 1.4), reveal=vis,
                 edge=YELLOW_HOT if c.spec.style != "alert" else ALERT)
            y += int(round(layer.h * e)) + gap


def _render_card(spec: CardSpec, w: int, t: float) -> Layer:
    """Render one card to its own opaque tile (ref1 hazard-panel look).

    Height is derived from the content (header + optional big line + rows +
    optional bar) so nothing ever clips and the column can pack tightly."""
    s = w / 275.0                        # scale relative to a 275px reference card
    pad = int(12 * s)
    header_h = int(30 * s)
    title_sz = max(11, int(15 * s))
    big_sz = max(20, int(33 * s))
    big_lh = int(43 * s)
    label_sz = max(10, int(13 * s))
    row_h = int(24 * s)
    bar_zone = int(24 * s)

    style = spec.style                       # solid | panel | alert | bracket | underline
    solid = style == "solid"
    is_alert = style == "alert"
    open_style = style in ("bracket", "underline")

    # ---- measure -> height
    h = header_h + int(9 * s)
    if spec.big is not None:
        h += big_lh
    h += row_h * len(spec.rows)
    if spec.bar is not None:
        h += bar_zone
    h += int(10 * s)
    L = Layer(w, h)
    cut = max(6, int(w * 0.05))

    fill = YELLOW if solid else PANEL_BLK
    accent = ALERT if is_alert else YELLOW
    label_col = _mix(YELLOW, BLACK, 0.6) if solid else GREY
    val_default = BLACK if solid else YELLOW
    body = cut_poly(1, 1, w - 1, h - 1, cut, corners="tr,bl")
    L.fill_poly(body, fill)
    ind = pad + (int(9 * s) if style == "bracket" else 0)   # content indent (past the spine)

    # ---- frame + title, distinct per style ------------------------------------
    if style == "bracket":
        # dark card framed by accent corner L-brackets, a left accent spine, and
        # a cut-corner title TAB (reads as a targeting readout).
        L.poly(body, _mix(YELLOW, PANEL_BLK, 0.4), 1, closed=True)
        leg = int(18 * s)
        for bx, by, sx, sy in ((3, 3, 1, 1), (w - 3, 3, -1, 1), (3, h - 3, 1, -1), (w - 3, h - 3, -1, -1)):
            L.poly([(bx + sx * leg, by), (bx, by), (bx, by + sy * leg)], accent, 2, closed=False)
        L.fill_rect(3, header_h * 0.5, int(4 * s) + 3, h - 4, accent, 255)      # left spine
        tabw = int(text_width(spec.title, title_sz, 1.6)) + int(18 * s)
        L.fill_poly(cut_poly(3, 3, 3 + tabw, header_h, int(8 * s), corners="br"), accent)
        L.text(spec.title, pad, header_h * 0.5, title_sz, BLACK, anchor="lm", tracking=1.6)
    elif style == "underline":
        # dark card, title with an accent underline (no header bar) + corner ticks.
        L.poly(body, _mix(YELLOW, PANEL_BLK, 0.4), 1, closed=True)
        L.text(spec.title, pad, header_h * 0.5, title_sz, accent, anchor="lm", tracking=1.8)
        L.line((pad, header_h - int(3 * s)), (w - pad, header_h - int(3 * s)),
               _mix(accent, PANEL_BLK, 0.15), 2)
        L.line((w - 3, 3), (w - 3 - int(12 * s), 3), accent, 2)
        L.line((w - 3, 3), (w - 3, 3 + int(12 * s)), accent, 2)
    elif is_alert:
        L.poly(body, ALERT, max(2, int(2 * s)), closed=True)
        band = hazard_band(w - 3, header_h - 2, stripe=int(10 * s), phase=t * 30.0)
        L.paste_bgr(band, 2, 2)
        blink = 0.55 + 0.45 * math.sin(t * 12.0)
        tw = int(text_width(spec.title, title_sz, 2.0)) + int(16 * s)
        L.fill_rect(w * 0.5 - tw * 0.5, 3, w * 0.5 + tw * 0.5, header_h - 1, BLACK, 235)
        L.text(spec.title, w * 0.5, header_h * 0.5, title_sz, _mix(ALERT, YELLOW, blink),
               anchor="cm", tracking=2.0)
    else:                                    # solid | panel — filled header bar
        L.poly(body, (BLACK if solid else YELLOW), max(2, int(2 * s)), closed=True)
        head_bg = BLACK if solid else YELLOW
        head_fg = YELLOW if solid else BLACK
        L.fill_rect(2, 2, w - 2, header_h, head_bg, 255)
        L.text(spec.title, pad, header_h * 0.5, title_sz, head_fg, anchor="lm", tracking=1.6)
        for i in range(3):                   # index ticks on the header right
            tx = w - pad - i * int(7 * s)
            L.line((tx, 6), (tx, header_h - 5), _mix(head_fg, head_bg, 0.45), 2)

    # ---- content (big value / rows / bar) — shared -----------------------------
    cy = header_h + int(9 * s)
    if spec.big is not None:
        txt, col = spec.big
        L.text(txt, ind, cy, big_sz, col, anchor="lt", tracking=1.0)
        cy += big_lh
    for (label, value, vcol) in spec.rows:
        L.text(label, ind, cy + row_h * 0.5, label_sz, label_col, anchor="lm", tracking=1.0)
        L.text(value, w - pad, cy + row_h * 0.5, label_sz + int(2 * s),
               vcol if vcol is not None else val_default, anchor="rm", tracking=0.5)
        cy += row_h
    if spec.bar is not None:
        frac, col = spec.bar
        _seg_bar(L, ind, cy + int(3 * s), w - pad - ind, int(9 * s), frac, col,
                 track=_mix(YELLOW, BLACK, 0.55) if solid else _mix(PANEL_BLK, YELLOW_DK, 0.3))
    if spec.note:
        note_col = accent if open_style else (YELLOW if solid else BLACK)
        L.text(spec.note, w - pad, header_h * 0.5, label_sz - 1, note_col, anchor="rm", tracking=0.5)
    return L


def _seg_bar(L: Layer, x, y, w, h, frac: float, col: RGB, track: RGB, segs: int = 16) -> None:
    frac = max(0.0, min(1.0, frac))
    gap = max(1, int(w * 0.008))
    sw = (w - gap * (segs - 1)) / segs
    lit = int(round(frac * segs))
    for i in range(segs):
        sx = x + i * (sw + gap)
        L.fill_rect(sx, y, sx + sw, y + h, col if i < lit else track, 255)
