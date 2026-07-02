"""WARNING // Target-Acquisition HUD — reusable renderer kit.

A ground-up, *opaque* yellow HUD in the "ref1" hazard-panel style (bright angular
black-on-yellow WARNING cards) laid over a live OpenCV feed.  This is the on-frame
overlay for the O2-therapy follow dog: it drives BOTH the live runtime compositor
(:mod:`core.hud.visualization`, which maps the robot's telemetry into
:class:`Telemetry`) AND the standalone demo (``examples/warning_hud.py``).  It
replaced the earlier red "GO2 // TACTICAL" overlay.

Why opaque (and not arcv's ``Overlay``): arcv composites its HUD *additively*
(glow), which physically cannot draw black text on a bright panel — additive
light only brightens, it can never darken.  ref1 is fundamentally solid
black-on-yellow, so the panels here are alpha-composited opaquely with cv2 while
text is rendered with the **Share Tech Mono** face via Pillow (a copy is bundled
next to this module so the Docker container needs no font from arcv).

Everything drawn is real: status, target count, confidence, position, an
estimated range (labelled EST), fps and a smoothed signal quality.  No fabricated
telemetry.  Panels/cards spawn and collapse as the scene needs them and the
columns reflow so there is no dead space.

Public surface:
    * ``Telemetry`` / ``Target``      — the data the HUD draws
    * ``WarningHud(size).render(...)`` — draw one frame, returns BGR uint8
    * palette + ``load_font`` helpers  — for tests / reuse

This module is now a thin *facade* (Phase 2 structural refactor): the code was
split into single-responsibility siblings and re-exported here so every name that
was importable from :mod:`core.hud.warning_kit` before is still importable now:
    * :mod:`core.hud.warning_theme`  — palette, colour helpers, ``_out_cubic`` ease
    * :mod:`core.hud.warning_text`   — Share Tech Mono fonts + text metrics
    * :mod:`core.hud.warning_render` — ``Layer`` / ``blit`` / geometry primitives
    * :mod:`core.hud.warning_model`  — ``Target`` / ``Telemetry`` data model
    * :mod:`core.hud.warning_cards`  — the spawn/collapse card system
    * :mod:`core.hud.warning_hud`    — the ``WarningHud`` compositor
"""
from __future__ import annotations

from core.hud.warning_cards import (
    CardSpec,
    CardStack,
    _CardAnim,
    _render_card,
    _seg_bar,
)
from core.hud.warning_hud import WarningHud
from core.hud.warning_model import Target, Telemetry
from core.hud.warning_render import (
    Layer,
    blit,
    cut_poly,
    hazard_band,
)
from core.hud.warning_text import (
    _FONT_CACHE,
    _FONT_PATH,
    _PIL_OK,
    _advance,
    _font_candidates,
    load_font,
    text_width,
)
from core.hud.warning_theme import (
    ALERT,
    ALERT_DK,
    BLACK,
    FEED_DIM,
    GREY,
    PANEL_BLK,
    RGB,
    WHITE,
    YELLOW,
    YELLOW_DK,
    YELLOW_HOT,
    _mix,
    _out_cubic,
    bgr,
)

__all__ = ["Telemetry", "Target", "WarningHud", "CardSpec", "CardStack",
           "load_font", "bgr", "YELLOW", "BLACK", "ALERT"]
