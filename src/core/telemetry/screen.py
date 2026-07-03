"""Live-screen helpers for the btop-style terminal UI toolkit.

An alt-screen controller for flicker-free in-place dashboard repaints plus the
cursor show/hide escapes. See ``term_ui`` for the public facade.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional

from .ansi import _ESC
from .theme import Theme, default_theme

# ---------------------------------------------------------------------------
# Live screen helper (alt-screen, flicker-free repaint)
# ---------------------------------------------------------------------------


class Screen:
    """Minimal alt-screen controller for in-place dashboard repaints."""

    def __init__(self, stream=None, theme: Optional[Theme] = None):
        self.stream = stream if stream is not None else sys.stdout
        self.theme = theme or default_theme()
        self._active = False

    def __enter__(self) -> "Screen":
        if self.theme.enabled:
            self.stream.write(f"{_ESC}[?1049h{_ESC}[?25l{_ESC}[2J")
            self.stream.flush()
            self._active = True
        return self

    def __exit__(self, *exc) -> None:
        if self._active:
            self.stream.write(f"{_ESC}[?25h{_ESC}[?1049l")
            self.stream.flush()
            self._active = False

    def render(self, lines: Iterable[str]) -> None:
        """Repaint from the home position, clearing each line to EOL.

        In alt-screen mode the final line gets NO trailing newline: writing a
        newline after the last row of a full-height frame scrolls the buffer up
        one line, pushing the top row (the header) off-screen.
        """
        rows = list(lines)
        buf = [f"{_ESC}[H"] if self._active else []
        for i, ln in enumerate(rows):
            buf.append(ln + (f"{_ESC}[K" if self._active else ""))
            if not self._active or i < len(rows) - 1:
                buf.append("\n")
        if self._active:
            buf.append(f"{_ESC}[J")  # clear below
        self.stream.write("".join(buf))
        self.stream.flush()


def hide_cursor(stream=None) -> str:
    return f"{_ESC}[?25l"


def show_cursor(stream=None) -> str:
    return f"{_ESC}[?25h"
