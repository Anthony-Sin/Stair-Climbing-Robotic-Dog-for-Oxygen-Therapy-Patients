"""ANSI-aware width helpers for the btop-style terminal UI toolkit.

Measure/truncate/pad by *visible* columns, ignoring ANSI escapes and the
zero-width box notch so colored panels still line up. See ``term_ui`` for the
public facade.
"""

from __future__ import annotations

from .ansi import _ANSI_RE, _ESC

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
