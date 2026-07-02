"""Resolved color/glyph capabilities + paint helpers (btop-style UI toolkit).

The ``Theme`` dataclass carries the detected color level and unicode capability
and provides the ``paint``/``gradient`` helpers every component uses. See
``term_ui`` for the public facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .ansi import (
    _ASCII_BOX,
    _BoxChars,
    _ESC,
    _UNICODE_BOX,
    BOX_ACCENTS,
    NONE,
    PALETTE,
    TRUECOLOR,
    _bg_code,
    _enable_windows_vt,
    _fg_code,
    _hex_to_rgb,
    detect_color_level,
    detect_unicode,
    gradient_rgb,
)

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
