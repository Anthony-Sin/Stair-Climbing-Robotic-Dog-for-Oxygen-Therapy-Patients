"""On-frame HUD composition — the "GO2 // TACTICAL" operator overlay.

An original tactical-ops layout for the stair-climbing O2-therapy follow dog:
chamfered (cut-corner) bracket panels along the darkened frame border, a lock
reticle on the tracked patient, an instrument-style readout grid, a 4-leg gait
diagram, a forward LiDAR sector radar, and a raster depth inset — all in a
single signal-red accent on near-black.  Nothing here is decorative filler;
every value drawn is live telemetry resolved by :func:`derive`.

This is NOT a port of any arcv example or the previous "constellation" HUD —
it is a ground-up design.  Rendering goes through arcv's real ``Overlay`` (GPU
vector/text batches + HDR bloom) via :mod:`core.hud.gl_hud`; the composited
result is additively blended onto the live camera frame, so panels are drawn as
glowing OUTLINES + text on a near-black bed (a filled panel would add-wash the
video — legibility instead comes from the compositor's edge vignette, under
which the border panels sit).  Colours are RGBA floats in ``[0, 1]`` (arcv's
convention) — NOT the RGB int tuples the raster ``core/hud`` helpers use.

This module is a facade: the implementation now lives in cohesive siblings
(:mod:`core.hud.hud_layout_state`, :mod:`core.hud.hud_layout_widgets`,
:mod:`core.hud.hud_layout_sections`); the public entrypoints and every name the
previous ``hud_layout`` exposed are re-exported here unchanged.
"""
from __future__ import annotations

# State + palette (RGBA-float tokens, the alpha helper, the HudState snapshot).
from core.hud.hud_layout_state import (  # noqa: F401
    Color, WHITE, GREY, GREY_DIM, FRAME, RED, RED_HOT, RED_DIM, _TAU, _CUT,
    _fa, HudState, derive,
)

# Reusable arcv draw primitives.
from core.hud.hud_layout_widgets import (  # noqa: F401
    _fmt_m, _fill, _panel, _led, _row, _bar, _bipolar,
)

# Section builders + entrypoints.
from core.hud.hud_layout_sections import (  # noqa: F401
    _topbar, _reticle, _left, _right, _depth_frame, _bottom, _SECTORS, _radar,
    _flash_cue, _fall_banner, _alert_border, build, build_paused,
)

__all__ = ["HudState", "derive", "build", "build_paused"]
