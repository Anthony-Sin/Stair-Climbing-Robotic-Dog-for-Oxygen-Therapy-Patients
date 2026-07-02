"""ARCV/MUTEK-scanner HUD theme + primitives.

Single accent-core (**signal red** ``#E4362A``) carries all focus/active/lock and
warning meaning; everything else is neutral off-white / grey / hairline on a
near-pure-black bed.  The look is built from straight lines, open corner brackets
and right angles, with dot-matrix texture filling otherwise-empty panel regions and
a fixed technical chrome (crop-marks, REC, timestamp, frame counter).

Text is rendered with Pillow (true monospace + AA + a 1px dark legibility outline)
in a single deferred pass per frame via :class:`TextLayer`; vector linework (bars,
reticle, radar, brackets) is drawn with OpenCV's AA primitives.  Colours are stored
as RGB tuples (Pillow-native); :func:`bgr` converts to OpenCV's BGR order.

This module is a facade: the implementation now lives in cohesive siblings
(:mod:`core.hud.hud_theme`, :mod:`core.hud.hud_text`, :mod:`core.hud.hud_shapes`,
:mod:`core.hud.hud_widgets`); everything the previous ``hud_primitives`` exposed is
re-exported here unchanged so import paths keep working.
"""
# Design tokens + colour helpers.
from core.hud.hud_theme import (  # noqa: F401
    BG_BASE, BG_PANEL, HAIRLINE, TEXT, DIM, ACCENT, ALERT, TEXTURE, ACCENT_DK, INK,
    FG, AMBER, SLATE, SUCCESS, BRIGHT_SLATE, WARNING, ERROR, MUTED, SURFACE,
    DEEP_AMBER, BG_FILL, HUD_TEXT, HUD_FG, HUD_AMBER, HUD_SLATE, HUD_MUTED,
    HUD_SUCCESS, HUD_WARNING, HUD_ALERT, HUD_ERROR, HUD_SURFACE, HUD_INK,
    bgr, dim, hex4,
)

# Fonts + deferred text layer.
from core.hud.hud_text import (  # noqa: F401
    _PIL_OK, _FONT_CACHE, _FONT_FILE_CACHE, _matplotlib_ttf, _font_file, _font,
    _text_width, TextLayer,
)

# Vector compositing + geometry primitives.
from core.hud.hud_shapes import (  # noqa: F401
    fill_region, _bevel_pts, _fill_poly_alpha, texture_fill, scanlines,
    connection_line, panel, group_backing, sweep_line, scan_frame, sector_bar,
    radar_graph, leg_row, reticle, lead_label, corner_frame,
)

# Text-composite widgets.
from core.hud.hud_widgets import (  # noqa: F401
    group_title, kv_row, data_tile, status_pill, focal_tag, _SCRAMBLE, decode_text,
    label_stack, section_label, status_bar, button_glyph, status_dots,
)
