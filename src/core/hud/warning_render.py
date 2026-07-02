"""WARNING HUD — low-level opaque drawing primitives.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor).  The
:class:`Layer` surface (opaque BGR + companion alpha + deferred Pillow text),
:func:`blit` (alpha-composite with a roll-open reveal), and the shared geometry
helpers :func:`cut_poly` (chamfered ref1 panel silhouette) + :func:`hazard_band`
(diagonal yellow/black stripe tile).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.hud.warning_text import _PIL_OK, Image, ImageDraw, _advance, load_font, text_width
from core.hud.warning_theme import BLACK, RGB, YELLOW, bgr


# ─────────────────────────── layer: bgr + alpha + text ─────────────────────────
class Layer:
    """A drawing surface: an opaque BGR image, a companion alpha mask (0..255),
    and a Pillow text pass baked on ``flush()``.  cv2 shapes write BGR *and*
    stamp the alpha; text is deferred so labels always sit on top."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = int(w), int(h)
        self.bgr = np.zeros((self.h, self.w, 3), np.uint8)
        self.a = np.zeros((self.h, self.w), np.uint8)
        self._txt = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)) if _PIL_OK else None
        self._draw = ImageDraw.Draw(self._txt) if _PIL_OK else None

    # -- vector (opaque) --------------------------------------------------------
    def fill_poly(self, pts, color: RGB, a: int = 255) -> None:
        p = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(self.bgr, [p], bgr(color), cv2.LINE_AA)
        cv2.fillPoly(self.a, [p], int(a), cv2.LINE_AA)

    def poly(self, pts, color: RGB, width: float = 2.0, closed: bool = True, a: int = 255) -> None:
        p = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.bgr, [p], closed, bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.polylines(self.a, [p], closed, int(a), int(round(width)), cv2.LINE_AA)

    def line(self, p0, p1, color: RGB, width: float = 2.0, a: int = 255) -> None:
        p0 = (int(round(p0[0])), int(round(p0[1])))
        p1 = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(self.bgr, p0, p1, bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.line(self.a, p0, p1, int(a), int(round(width)), cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, color: RGB, width: float = 2.0) -> None:
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, width, closed=True)

    def fill_rect(self, x0, y0, x1, y1, color: RGB, a: int = 255) -> None:
        self.fill_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, a)

    def disc(self, cx, cy, r, color: RGB, a: int = 255) -> None:
        cv2.circle(self.bgr, (int(cx), int(cy)), int(r), bgr(color), -1, cv2.LINE_AA)
        cv2.circle(self.a, (int(cx), int(cy)), int(r), int(a), -1, cv2.LINE_AA)

    def ring(self, cx, cy, r, color: RGB, width: float = 2.0, a: int = 255,
             a0: float = 0.0, a1: float = 360.0) -> None:
        cv2.ellipse(self.bgr, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1,
                    bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.ellipse(self.a, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1,
                    int(a), int(round(width)), cv2.LINE_AA)

    def paste_bgr(self, sub: np.ndarray, x: int, y: int, a: int = 255) -> None:
        h, w = sub.shape[:2]
        x, y = int(x), int(y)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        self.bgr[y0:y1, x0:x1] = sub[y0 - y:y1 - y, x0 - x:x1 - x]
        self.a[y0:y1, x0:x1] = np.maximum(self.a[y0:y1, x0:x1], a)

    # -- text (deferred) --------------------------------------------------------
    def text(self, s: str, x: float, y: float, size: int, color: RGB,
             anchor: str = "lt", tracking: float = 0.0, alpha: int = 255) -> None:
        if not s or self._draw is None:
            return
        h, v = anchor[0], anchor[1]
        w = text_width(s, size, tracking)
        if h == "c":
            x -= w * 0.5
        elif h == "r":
            x -= w
        vmap = {"t": "t", "m": "m", "b": "s"}
        f = load_font(size)
        adv = _advance(size) + tracking
        col = (int(color[0]), int(color[1]), int(color[2]), int(alpha))
        cx = x
        for ch in s:
            if ch != " ":
                self._draw.text((cx, y), ch, font=f, fill=col, anchor="l" + vmap.get(v, "t"))
            cx += adv

    def flush(self) -> None:
        """Bake the text layer down onto ``bgr``/``a`` (text on top, inside mask)."""
        if self._txt is None:
            return
        rgba = np.asarray(self._txt)  # (H,W,4) RGBA
        ta = rgba[:, :, 3].astype(np.float32) / 255.0
        if ta.max() <= 0:
            return
        tbgr = rgba[:, :, 2::-1]  # RGB→BGR
        m = ta[:, :, None]
        self.bgr[:] = (tbgr.astype(np.float32) * m + self.bgr.astype(np.float32) * (1.0 - m)).astype(np.uint8)
        self.a[:] = np.maximum(self.a, (ta * 255).astype(np.uint8))


def blit(dst: np.ndarray, layer: Layer, x: int, y: int, alpha: float = 1.0,
         reveal: float = 1.0, edge: Optional[RGB] = None) -> None:
    """Alpha-composite ``layer`` onto ``dst`` (BGR) at (x,y).

    ``reveal`` (0..1) roll-opens the layer from the top (a bright ``edge`` line is
    drawn at the wipe front).  ``alpha`` fades the whole layer."""
    layer.flush()
    vis_h = int(round(layer.h * max(0.0, min(1.0, reveal))))
    if vis_h <= 0 or alpha <= 0.0:
        return
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + layer.w), min(dst.shape[0], y + vis_h)
    if x1 <= x0 or y1 <= y0:
        return
    src_bgr = layer.bgr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    src_a = (layer.a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0) * float(alpha)
    m = src_a[:, :, None]
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (src_bgr * m + roi * (1.0 - m)).astype(np.uint8)
    if edge is not None and 0.02 < reveal < 0.995:
        ey = y1 - 1
        if 0 <= ey < dst.shape[0]:
            cv2.line(dst, (x0, ey), (x1 - 1, ey), bgr(edge), 2, cv2.LINE_AA)


# ───────────────────────────── shared geometry ────────────────────────────────
def cut_poly(x0, y0, x1, y1, cut: float, corners: str = "tr,bl") -> List[Tuple[float, float]]:
    """Rectangle with chamfered (cut) corners — the ref1 panel silhouette.

    ``corners`` picks which corners are cut: any of tl,tr,br,bl."""
    c = cut
    cs = set(s.strip() for s in corners.split(",")) if corners else set()
    pts: List[Tuple[float, float]] = []
    # TL
    pts += [(x0 + c, y0)] if "tl" in cs else [(x0, y0)]
    # TR
    pts += [(x1 - c, y0), (x1, y0 + c)] if "tr" in cs else [(x1, y0)]
    # BR
    pts += [(x1, y1 - c), (x1 - c, y1)] if "br" in cs else [(x1, y1)]
    # BL
    pts += [(x0 + c, y1), (x0, y1 - c)] if "bl" in cs else [(x0, y1)]
    # close TL
    if "tl" in cs:
        pts += [(x0, y0 + c)]
    return pts


def hazard_band(w: int, h: int, stripe: int = 13, fg: RGB = BLACK, bg: RGB = YELLOW,
                phase: float = 0.0) -> np.ndarray:
    """A yellow/black diagonal hazard-stripe tile (BGR)."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bgr(bg)
    step = stripe * 2
    off = int(phase * step) % step
    for x in range(-h - step, w + step, step):
        p0 = (x + off, 0)
        p1 = (x + off + h, h)
        cv2.line(img, p0, p1, bgr(fg), stripe, cv2.LINE_AA)
    return img
