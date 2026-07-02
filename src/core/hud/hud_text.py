"""Font resolution + the deferred Pillow text layer for the raster HUD.

Fonts are cached and resolved with sensible cross-platform fallbacks; text draws
are collected and flushed in a single Pillow pass per frame by :class:`TextLayer`
(true monospace + AA + a 1px dark legibility outline), with an OpenCV fallback
when Pillow is unavailable.
"""
import os

import cv2
import numpy as np
from typing import List, Optional

from core.hud.hud_theme import TEXT, INK, bgr

try:                                            # Pillow is the text backend
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:                               # pragma: no cover - degraded fallback
    _PIL_OK = False


# ───────────────────────────────── Fonts ──────────────────────────────────────
_FONT_CACHE: dict = {}
_FONT_FILE_CACHE: dict = {}


def _matplotlib_ttf(name: str) -> Optional[str]:
    try:
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name)
    except Exception:
        return None


def _font_file(bold: bool, mono: bool) -> Optional[str]:
    """Resolve a font file.  ``mono`` → the data/monospace face (hex, decode,
    numbers); else the *display* face — a condensed techy sans (Bahnschrift →
    Agency FB → Segoe Semibold), falling back to DejaVu so the Jetson container
    still renders."""
    key = (bool(bold), bool(mono))
    if key in _FONT_FILE_CACHE:
        return _FONT_FILE_CACHE[key]
    cands: List[str] = []
    if mono:
        cands += [r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
                  r"C:\Windows\Fonts\CascadiaMono.ttf",
                  _matplotlib_ttf("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf" % ("-Bold" if bold else "")]
    else:
        cands += [r"C:\Windows\Fonts\bahnschrift.ttf",
                  r"C:\Windows\Fonts\AGENCYB.TTF" if bold else r"C:\Windows\Fonts\AGENCYR.TTF",
                  r"C:\Windows\Fonts\seguisb.ttf", r"C:\Windows\Fonts\tahoma.ttf",
                  _matplotlib_ttf("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")]
    found = next((c for c in cands if c and os.path.exists(c)), None)
    _FONT_FILE_CACHE[key] = found
    return found


def _font(size: int, bold: bool = False, mono: bool = False):
    key = (int(size), bool(bold), bool(mono))
    f = _FONT_CACHE.get(key)
    if f is not None:
        return f
    if _PIL_OK:
        path = _font_file(bold, mono)
        try:
            f = ImageFont.truetype(path, int(size)) if path else ImageFont.load_default()
        except Exception:
            f = ImageFont.load_default()
        if bold and not mono and path and path.lower().endswith("bahnschrift.ttf"):
            # Bahnschrift is a variable font — pick a heavier named instance.
            for name in ("Bold", "SemiBold"):
                try:
                    f.set_variation_by_name(name)
                    break
                except Exception:
                    pass
    else:
        f = None
    _FONT_CACHE[key] = f
    return f


def _text_width(text: str, size: int, bold: bool = True, mono: bool = False) -> int:
    """Pixel advance of ``text`` in the chosen face (exact via Pillow metrics,
    else an em-ratio estimate)."""
    if _PIL_OK:
        try:
            return int(round(_font(size, bold, mono).getlength(text)))
        except Exception:
            pass
    return int(round(len(text) * size * (0.6 if mono else 0.52)))


# ───────────────────────────── Deferred text layer ────────────────────────────
class TextLayer:
    """Collects text draws and renders them all in one Pillow pass per frame.

    Each ``add`` records a string + style; ``flush`` converts the BGR frame to a
    Pillow image once, draws every string with a true monospace font + a 1px dark
    outline (the only legibility aid the design permits — never a box), then writes
    the result back into the same ndarray in place.
    """

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def add(self, x: int, y: int, text: str, color=TEXT, *, size: int = 13,
            bold: bool = False, anchor: str = "lt", outline: bool = True,
            tracking: float = 0.0, mono: bool = False) -> None:
        if text is None or text == "":
            return
        self.items.append((int(x), int(y), str(text), color, int(size), bool(bold),
                           anchor, bool(outline), float(tracking), bool(mono)))

    def flush(self, frame_bgr: np.ndarray) -> None:
        if not self.items:
            return
        if not _PIL_OK:                          # OpenCV fallback (no true monospace)
            for x, y, text, color, size, bold, anchor, outline, _trk, _mono in self.items:
                self._cv_fallback(frame_bgr, x, y, text, color, size, anchor, outline)
            self.items = []
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for x, y, text, color, size, bold, anchor, outline, trk, mono in self.items:
            font = _font(size, bold, mono)
            kw = {"font": font, "fill": tuple(color), "anchor": anchor}
            if outline:
                kw["stroke_width"] = 1
                kw["stroke_fill"] = INK
            try:
                if trk:                           # mechanical letter-spacing (§2.2)
                    self._tracked(draw, x, y, text, font, tuple(color), anchor, outline, trk)
                else:
                    draw.text((x, y), text, **kw)
            except TypeError:                     # very old Pillow w/o anchor/stroke
                draw.text((x, y), text, font=font, fill=tuple(color))
        frame_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.items = []

    @staticmethod
    def _tracked(draw, x, y, text, font, color, anchor, outline, trk) -> None:
        # Per-glyph layout with extra advance; honours left/middle/right horizontal
        # anchor.  Vertical anchor falls back to the font's baseline handling.
        try:
            widths = [draw.textlength(ch, font=font) for ch in text]
        except Exception:
            widths = [font.size * 0.6 for _ in text]
        total = sum(widths) + trk * max(0, len(text) - 1)
        ha = anchor[0] if anchor else "l"
        cx = x - total if ha == "r" else (x - total / 2.0 if ha == "m" else x)
        va = anchor[1] if len(anchor) > 1 else "t"
        cur = cx
        for ch, wch in zip(text, widths):
            kw = {"font": font, "fill": color, "anchor": "l" + va}
            if outline:
                kw["stroke_width"] = 1
                kw["stroke_fill"] = INK
            draw.text((cur, y), ch, **kw)
            cur += wch + trk

    @staticmethod
    def _cv_fallback(frame_bgr, x, y, text, color, size, anchor, outline) -> None:
        scale = max(0.32, size / 30.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ax = x - tw if anchor and anchor[0] == "r" else (x - tw // 2 if anchor and anchor[0] == "m" else x)
        ay = y + th // 2 if anchor and len(anchor) > 1 and anchor[1] == "m" else y + th
        if outline:
            cv2.putText(frame_bgr, text, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(INK), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(color), 1, cv2.LINE_AA)
