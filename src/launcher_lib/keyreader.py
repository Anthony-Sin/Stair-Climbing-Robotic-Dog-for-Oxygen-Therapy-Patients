"""Cross-platform single-key reader.

Extracted verbatim from ``launcher.py``. Self-contained: only depends on
``os``/``sys`` plus lazily-imported platform modules (msvcrt / termios / tty /
select).
"""

from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# Cross-platform single-key reader
# ---------------------------------------------------------------------------


class KeyReader:
    """Read logical keys ('up','down','left','right','enter','space','tab',char)."""

    def __init__(self):
        self.is_windows = os.name == "nt"
        self._fd = None
        self._old = None

    def __enter__(self):
        if not self.is_windows:
            try:
                if sys.stdin.isatty():
                    import termios
                    import tty
                    self._fd = sys.stdin.fileno()
                    self._old = termios.tcgetattr(self._fd)
                    tty.setcbreak(self._fd)  # leaves ISIG on, so Ctrl-C still works
            except Exception:
                self._old = None
        return self

    def __exit__(self, *exc):
        if not self.is_windows and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        """Non-blocking: return a key if one is buffered, else None."""
        try:
            if self.is_windows:
                import msvcrt
                if msvcrt.kbhit():
                    return self.read()
                return None
            import select
            if not sys.stdin.isatty():
                return None
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return self.read()
        except Exception:
            return None
        return None

    def read(self) -> str:
        if self.is_windows:
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                return {"H": "up", "P": "down", "K": "left", "M": "right",
                        "I": "pgup", "Q": "pgdn", "G": "home", "O": "end"}.get(code, "")
            return self._classify(ch)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(1)
            if nxt == "[":
                code = sys.stdin.read(1)
                if code.isdigit():  # ESC [ <n> ~  (PgUp/PgDn/Home/End)
                    seq = code
                    while True:
                        c = sys.stdin.read(1)
                        if c == "~" or not c:
                            break
                        seq += c
                    return {"5": "pgup", "6": "pgdn", "1": "home", "7": "home",
                            "4": "end", "8": "end"}.get(seq, "")
                return {"A": "up", "B": "down", "C": "right", "D": "left",
                        "H": "home", "F": "end"}.get(code, "")
            return "esc"
        return self._classify(ch)

    @staticmethod
    def _classify(ch: str) -> str:
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\t":
            return "tab"
        if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            return "quit"
        return ch
