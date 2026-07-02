"""Status glyphs and chrome bars for the btop-style terminal UI toolkit.

Launcher stage glyphs, full-width status/title bars, and the single-row status
line used by the live dashboard. See ``term_ui`` for the public facade.
"""

from __future__ import annotations

from typing import Optional

from .text import pad, truncate, visible_len
from .theme import Theme, default_theme

# ---------------------------------------------------------------------------
# Status glyphs
# ---------------------------------------------------------------------------

#: launcher stage states -> (glyph, color). ASCII fallbacks chosen by Theme.
_STATE_STYLE = {
    "start": ("▶", "blue"),
    "running": ("▶", "blue"),
    "notice": ("•", "yellow"),
    "warning": ("▲", "warning"),
    "ready": ("✔", "success"),
    "complete": ("✔", "success"),
    "ok": ("✔", "success"),
    "cleanup": ("◦", "muted"),
    "pruned": ("◦", "muted"),
    "skipped": ("–", "muted"),
    "dry-run": ("◦", "muted"),
    "failed": ("✖", "error"),
    "error": ("✖", "error"),
}
_STATE_ASCII = {
    "▶": ">", "•": "*", "▲": "!", "✔": "OK", "◦": "-", "–": "-", "✖": "X",
}


def state_glyph(state: str, theme: Optional[Theme] = None) -> str:
    theme = theme or default_theme()
    glyph, color = _STATE_STYLE.get(state, ("•", "fg"))
    if not theme.unicode:
        glyph = _STATE_ASCII.get(glyph, glyph)
    return theme.paint(glyph, fg=color, bold=True)


def bar(left: str, right: str = "", width: int = 80, theme: Optional[Theme] = None,
        bg="net", fg="bg", bold: bool = True) -> str:
    """A full-width status/title bar: ``left`` text, ``right`` right-justified, one bg.

    Classic TUI chrome (header/footer). Pass plain text; the whole line is painted
    one background color so it reads as a solid bar.
    """
    theme = theme or default_theme()
    avail = max(0, width - visible_len(right) - 1)
    if visible_len(left) > avail:
        left = truncate(left, avail)
    space = max(0, width - visible_len(left) - visible_len(right))
    return theme.paint(left + " " * space + right, fg=fg, bg=bg, bold=bold)


def status_line(timestamp: str, stage: str, state: str, message: str,
                theme: Optional[Theme] = None) -> str:
    """A single colorized launcher status row (used by the live dashboard)."""
    theme = theme or default_theme()
    ts = theme.paint(timestamp, fg="muted")
    gl = state_glyph(state, theme)
    stg = theme.paint(pad(stage, 11), fg="secondary", bold=True)
    _, color = _STATE_STYLE.get(state, ("•", "fg"))
    msg = theme.paint(message, fg=color if state in ("failed", "error") else "fg")
    return f"{ts} {gl} {stg} {msg}"
