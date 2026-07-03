"""Renderable components for the btop-style terminal UI toolkit.

Panels (notched, colored titles), gradient meters, block sparklines, metric
rows, and the layout helpers that turn stacked boxes into a dashboard grid. See
``term_ui`` for the public facade.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .ansi import _BLOCK_RAMP, _TTY_RAMP, gradient_rgb
from .text import pad, visible_len
from .theme import Theme, default_theme

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def panel(title: str, lines: Sequence[str], width: int = 60,
          accent: str = "cpu", theme: Optional[Theme] = None,
          footer: Optional[str] = None) -> List[str]:
    """Render a btop box with a notched, colored title.

    ``╭─┐ title ┌──────────────╮``
    ``│  ...content...          │``
    ``╰──────────────────────────╯``

    *lines* are already-rendered content rows (may contain ANSI); each is
    padded/truncated to the interior width. *width* is the total outer width.
    """
    theme = theme or default_theme()
    bc = theme.box()
    accent_rgb = theme._resolve(accent) or theme._resolve("muted")
    width = max(width, visible_len(title) + 8)
    inner = width - 2

    def border(s: str) -> str:
        return theme.paint(s, fg=accent_rgb)

    # Top: ╭─┐ title ┌───╮
    title_txt = theme.paint(f" {title} ", fg="primary", bold=True)
    left_vis = 3 + visible_len(title_txt)  # tl + h + notch_l + title + notch_r
    fill = max(0, width - left_vis - 2)
    top = border(bc.tl + bc.h + bc.notch_l) + title_txt + border(
        bc.notch_r + bc.h * fill + bc.tr)

    out = [top]
    for ln in lines:
        body = pad(" " + ln, inner)
        out.append(border(bc.v) + body + border(bc.v))

    if footer:
        foot_txt = theme.paint(f" {footer} ", fg="muted")
        ffill = max(0, width - 3 - visible_len(foot_txt) - 2)
        out.append(border(bc.bl + bc.h * ffill + bc.notch_l) + foot_txt
                   + border(bc.notch_r + bc.h + bc.br))
    else:
        out.append(border(bc.bl + bc.h * inner + bc.br))
    return out


def block_width(lines: Sequence[str]) -> int:
    """Max visible width of a block of (possibly ANSI-styled) lines."""
    return max((visible_len(ln) for ln in lines), default=0)


def hjoin(blocks: Sequence[Sequence[str]], gap: int = 2) -> List[str]:
    """Place rendered blocks (e.g. panels) side by side into one block.

    This is what turns the stacked-boxes look into a btop dashboard grid: each
    block is padded to its own width and to the tallest block's height, then the
    rows are concatenated with ``gap`` spaces between columns.
    """
    cols = [list(b) for b in blocks if b]
    if not cols:
        return []
    widths = [block_width(b) for b in cols]
    height = max(len(b) for b in cols)
    sep = " " * gap
    out: List[str] = []
    for r in range(height):
        cells = []
        for b, w in zip(cols, widths):
            cells.append(pad(b[r] if r < len(b) else "", w))
        out.append(sep.join(cells))
    return out


def fit_height(lines: List[str], rows: int) -> List[str]:
    """Pad or truncate a block to exactly *rows* lines (dynamic full-height fill)."""
    lines = list(lines)
    if len(lines) >= rows:
        return lines[:rows]
    return lines + [""] * (rows - len(lines))


def kv(label: str, value: str, theme: Optional[Theme] = None,
       label_w: int = 12, value_color="primary", bold_value: bool = True) -> str:
    """A muted ``label`` followed by a colored ``value`` (btop metric row)."""
    theme = theme or default_theme()
    lbl = theme.paint(pad(label, label_w), fg="muted")
    val = theme.paint(value, fg=value_color, bold=bold_value)
    return f"{lbl}{val}"


def meter(frac: float, width: int = 20, theme: Optional[Theme] = None,
          mode: str = "meter") -> str:
    """A gradient-filled meter: ``[■■■■■■░░░░]``-style, colored per cell.

    ``mode``: ``meter`` (■/░), ``bar`` (█/░), or ``tty`` (#/.).
    """
    theme = theme or default_theme()
    width = max(1, width)
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    filled = int(round(frac * width))
    if theme.unicode and mode != "tty":
        full, empty = ("■", "░") if mode == "meter" else ("█", "░")
    else:
        full, empty = "#", "."
    cells = []
    for i in range(width):
        if i < filled:
            pos = i / max(1, width - 1)
            cells.append(theme.paint(full, fg=gradient_rgb(pos)))
        else:
            cells.append(theme.paint(empty, fg="muted"))
    return "".join(cells)


def sparkline(values: Sequence[float], theme: Optional[Theme] = None,
              width: Optional[int] = None, mode: str = "block") -> str:
    """A gradient block sparkline (``▁▂▃▅▇█``) sampled from *values*."""
    theme = theme or default_theme()
    vals = list(values)
    if not vals:
        return ""
    if width and len(vals) > width:
        # keep the most recent `width` samples (btop scrolls left)
        vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    ramp = _BLOCK_RAMP if (theme.unicode and mode == "block") else _TTY_RAMP
    out = []
    for v in vals:
        norm = 0.0 if rng <= 0 else (v - lo) / rng
        idx = min(len(ramp) - 1, int(round(norm * (len(ramp) - 1))))
        out.append(theme.paint(ramp[idx], fg=gradient_rgb(norm)))
    return "".join(out)


# ---------------------------------------------------------------------------
# Claude Code components (DESIGN.md §4–§8): dashed input box + thinking spinner
# ---------------------------------------------------------------------------

#: Signature reverse-mirror spinner cycle (DESIGN.md §8). ASCII falls back to a
#: plain rotator.
_SPIN_FRAMES = "·✢✳✶✻✽✻✶✳✢"
_SPIN_ASCII = "|/-\\"

#: A handful of the whimsical thinking verbs (DESIGN.md §5). Kept short and
#: index-selected (no RNG) so renders stay deterministic/testable.
THINKING_VERBS = (
    "Percolating", "Cogitating", "Ruminating", "Simmering", "Noodling",
    "Whirring", "Marinating", "Tinkering", "Conjuring", "Assembling",
)


def spinner(t: float, theme: Optional[Theme] = None) -> str:
    """One shimmering terracotta spinner frame sampled at time *t* (seconds)."""
    theme = theme or default_theme()
    if theme.unicode:
        return theme.paint(_SPIN_FRAMES[int(t * 8) % len(_SPIN_FRAMES)],
                           fg="primary", bold=True)
    return theme.paint(_SPIN_ASCII[int(t * 8) % len(_SPIN_ASCII)], fg="primary", bold=True)


def thinking(t: float, theme: Optional[Theme] = None, verb: Optional[str] = None,
             suffix: str = "") -> str:
    """Spinner + a whimsical verb, e.g. ``✳ Percolating…  <suffix>``."""
    theme = theme or default_theme()
    if verb is None:
        verb = THINKING_VERBS[int(t) % len(THINKING_VERBS)]
    line = spinner(t, theme) + " " + theme.paint(f"{verb}…", fg="primary")
    if suffix:
        line += "  " + theme.paint(suffix, fg="muted")
    return line


def _dashed(width: int, theme: Theme) -> str:
    """The DESIGN.md dashed rule (``- - - -``), muted gray — NOT box-drawing."""
    n = max(1, width // 2)
    return theme.paint(("- " * n)[:width].rstrip(), fg="muted")


def input_box(text: str, cursor: int, theme: Optional[Theme] = None,
              width: int = 60, prompt: str = "›") -> List[str]:
    """The signature Claude Code input: dashed border, ``›`` prompt, block cursor.

    Horizontally scrolls so *cursor* stays visible on long lines. Returns three
    rows (top dash, input, bottom dash).
    """
    theme = theme or default_theme()
    width = max(20, width)
    cursor = max(0, min(cursor, len(text)))
    avail = width - 4  # " › " + a trailing cell

    # Scroll a window so the cursor is always shown.
    start = 0
    if len(text) > avail:
        start = max(0, cursor - avail + 1)
    win = text[start:start + avail]
    cur = cursor - start

    if cur >= len(win):
        body = theme.paint(win, fg="fg") + theme.paint(" ", fg="bg", bg="primary")
    else:
        body = (theme.paint(win[:cur], fg="fg")
                + theme.paint(win[cur] or " ", fg="bg", bg="primary")
                + theme.paint(win[cur + 1:], fg="fg"))
    line = " " + theme.paint(prompt, fg="primary", bold=True) + " " + body
    dash = _dashed(width, theme)
    return [dash, line, dash]
