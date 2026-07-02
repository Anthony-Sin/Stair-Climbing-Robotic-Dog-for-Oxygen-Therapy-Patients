"""btop-style terminal UI toolkit (pure stdlib, cross-platform).

A small rendering library that draws the btop aesthetic described in the repo's
``DESIGN.md``: pure-black dashboard, rounded boxes with notched/colored titles
(``╭─┐ title ┌────╮``), TrueColor gradient meters (green → yellow → red), and
block sparklines. It has **no third-party dependencies** so the exact same code
styles the Windows/sim launcher, the Linux/real launcher and the shared
``core/main.py`` controller banner.

Design goals
------------
* **Degrade, never break.** When the stream is not a TTY, ``NO_COLOR`` is set,
  ``TERM=dumb``, or the user passes ``--no-color``, every helper falls back to
  plain ASCII with zero escape codes. The box characters fall back to ``+-|``.
* **TrueColor → 256 → 16.** Colors are emitted at the richest level the
  terminal advertises and converted down automatically, so the palette looks
  right in Windows Terminal, modern conhost, and a plain Linux console alike.
* **Measure visible width.** All padding/truncation ignores ANSI escapes and
  the zero-width box notch, so panels line up even when colored.

The public surface is intentionally tiny:

    theme = Theme.detect()              # or Theme.detect(force_color=False)
    print(theme.paint("hi", "green", bold=True))
    for line in panel("config", rows, width=48, accent="net", theme=theme):
        print(line)
    print(meter(0.62, 20, theme))
    print(sparkline([1, 3, 5, 8, 5, 2], theme))

This module is a thin facade: the implementation lives in cohesive sibling
modules (``ansi``, ``text``, ``theme``, ``components``, ``status``, ``screen``)
and is re-exported here so ``core.telemetry.term_ui`` keeps its historical
import surface.
"""

from __future__ import annotations

from .ansi import (  # noqa: F401
    ANSI16,
    ANSI256,
    BOX_ACCENTS,
    NONE,
    PALETTE,
    TRUECOLOR,
    _ANSI16,
    _ANSI_RE,
    _ASCII_BOX,
    _BLOCK_RAMP,
    _BoxChars,
    _ESC,
    _GRADIENT_STOPS,
    _TTY_RAMP,
    _UNICODE_BOX,
    _bg_code,
    _enable_windows_vt,
    _fg_code,
    _hex_to_rgb,
    _rgb_to_16,
    _rgb_to_256,
    detect_color_level,
    detect_unicode,
    gradient_rgb,
)
from .components import (  # noqa: F401
    block_width,
    fit_height,
    hjoin,
    kv,
    meter,
    panel,
    sparkline,
)
from .screen import (  # noqa: F401
    Screen,
    hide_cursor,
    show_cursor,
)
from .status import (  # noqa: F401
    _STATE_ASCII,
    _STATE_STYLE,
    bar,
    state_glyph,
    status_line,
)
from .text import (  # noqa: F401
    pad,
    strip_ansi,
    truncate,
    visible_len,
)
from .theme import (  # noqa: F401
    Theme,
    _DEFAULT,
    default_theme,
)
