"""WARNING HUD — Share Tech Mono fonts + monospace text metrics.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor).  Owns the
optional Pillow import (``_PIL_OK`` / ``Image`` / ``ImageDraw`` / ``ImageFont``)
that the rest of the kit shares, the bundled-font discovery, the font cache, and
the fixed-pitch advance / text-width helpers.

Mutable module state (``_FONT_CACHE`` / ``_FONT_PATH``) and every function that
reads or writes it live together in this one module so the cache stays coherent.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False
    # Keep the names bound (as ``None``) so sibling modules can ``from`` -import
    # them unconditionally; they are only ever *used* when ``_PIL_OK`` is True,
    # matching the original single-module guarded behaviour.
    Image = ImageDraw = ImageFont = None


# ───────────────────────────────── fonts ──────────────────────────────────────
# Share Tech Mono.  A copy is bundled next to this module so the runtime works in
# the Docker container without arcv's resources; arcv's copy + system monospace
# fonts are fallbacks.
def _font_candidates() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cands: List[str] = [os.path.join(here, "ShareTechMono-Regular.ttf")]   # bundled
    try:
        import arcv
        cands.append(os.path.join(os.path.dirname(arcv.__file__),
                                  "resources", "fonts", "ShareTechMono-Regular.ttf"))
    except Exception:  # pragma: no cover
        pass
    cands += [
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    return cands


_FONT_CACHE: Dict[int, "ImageFont.FreeTypeFont"] = {}
_FONT_PATH: Optional[str] = None


def load_font(size: int):
    """Return a cached Share Tech Mono PIL font at ``size`` px."""
    global _FONT_PATH
    size = max(6, int(round(size)))
    f = _FONT_CACHE.get(size)
    if f is not None:
        return f
    for path in ([_FONT_PATH] if _FONT_PATH else []) + _font_candidates():
        if not path:
            continue
        try:
            f = ImageFont.truetype(path, size)
            _FONT_PATH = path
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _advance(size: int) -> float:
    """Mono per-glyph advance in px (Share Tech Mono is fixed-pitch)."""
    f = load_font(size)
    try:
        return float(f.getlength("M"))
    except Exception:
        return size * 0.6


def text_width(s: str, size: int, tracking: float = 0.0) -> float:
    if not s:
        return 0.0
    return len(s) * (_advance(size) + tracking) - tracking
