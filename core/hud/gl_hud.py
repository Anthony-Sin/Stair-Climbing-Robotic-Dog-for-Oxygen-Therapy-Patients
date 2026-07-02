"""Lazy singleton GPU context that renders the on-frame HUD through arcv's real
``Overlay`` (ModernGL vector/text batches + HDR bloom composite) — see
:mod:`core.hud.hud_layout` for the actual composition.

One standalone context + ``Overlay`` + output ``Target`` is created for the
process lifetime (GL object creation is comparatively expensive; per-frame
rendering is not) and reused every frame.  ``render()`` draws the HUD onto a
near-black bed and reads the result back as a BGR uint8 array sized to the
video frame; the caller saturating-adds it onto the live camera frame, so the
near-black bed contributes ~nothing and glowing strokes bloom on top of the
video — the same additive relationship arcv's own composite pass uses between
its HUD layer and its (flat-colour) scene.

This pulls in a real GPU/GL context.  ``core/hud`` is imported from BOTH the
robot's Docker container AND the Isaac Sim host process (Isaac puts ``core/``
on its own ``sys.path``).  Standalone context creation is verified stand-alone
on a dev GPU here; it is NOT yet verified alongside Isaac Kit's own RTX
context in the same process, nor on real Jetson Orin hardware.  If context
creation fails outright, this raises loudly — no silent CPU fallback — so the
failure is visible instead of masked.

Headless-Linux note (confirmed on this project's Docker Desktop/WSL2 setup):
``moderngl``'s default backend auto-detection tries X11/GLX on Linux, which
raises ``XOpenDisplay: cannot open display`` with no X server -- ``_ensure()``
below tries the EGL backend first (works headless) and only falls back to
auto-detect for platforms where EGL isn't the right choice (e.g. Windows dev,
which auto-detects WGL correctly).  Separately: this container currently has
no NVIDIA EGL vendor ICD registered (``nvidia-smi`` sees the GPU fine --
that's the CUDA path -- but ``/usr/share/glvnd/egl_vendor.d/`` is empty), so
EGL falls back to Mesa's ``llvmpipe`` SOFTWARE renderer.  Measured on this
workload: ~30ms/frame (33 fps ceiling) under llvmpipe inside the container --
comfortably fast enough since this HUD only runs at ~6 fps, so this is left
as-is rather than chased further.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

_ctx = None
_ov = None
_target = None
_size: Optional[Tuple[int, int]] = None


def _build_theme():
    from arcv.theme import make_theme
    # NOTE: make_theme() always derives `glow` from `hue` internally (passing
    # glow= as an override collides -- Theme() gets it twice); the "red" preset's
    # hue=2.0 already gives a warm red-white bloom tint, which is what we want.
    return make_theme(
        "red",
        base=(0.026, 0.024, 0.026, 1.0),     # near-pure-black bed -> ~0 contribution on add-composite
        bloom_intensity=2.0, bloom_threshold=0.20, exposure=1.30,
        scanline_strength=0.0, sweep_strength=0.0,   # our own cv2 pass covers the whole frame incl. video
    )


def _ensure(size: Tuple[int, int]):
    global _ctx, _ov, _target, _size
    import moderngl
    from arcv.overlay import Overlay
    from arcv.passes.base import Target

    if _ctx is None:
        try:
            # Headless Linux (Docker/Jetson): X11/GLX auto-detect fails with no
            # display attached; EGL doesn't need one and is the right choice here.
            _ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception:
            # Platforms where EGL isn't the right backend (e.g. Windows dev,
            # which correctly auto-detects WGL) -- let moderngl pick.
            _ctx = moderngl.create_standalone_context(require=330)
    if _ov is None:
        theme = _build_theme()
        _ov = Overlay(_ctx, size, theme=theme, base_color=theme.base, grid=False)
        _target = Target(_ctx, size, components=4, dtype="f1")
        _size = size
    elif _size != size:
        _ov.resize(size[0], size[1])
        _target.resize(size)
        _size = size
    return _ov, _target


def render(build_fn: Callable[[object, int, int, float], None],
           size: Tuple[int, int], t: float) -> np.ndarray:
    """Run ``build_fn(ov, w, h, t)`` against a fresh frame and return a BGR
    uint8 ``(H, W, 3)`` array ready to composite onto the live video frame."""
    ov, target = _ensure(size)
    ov.begin()
    build_fn(ov, size[0], size[1], t)
    ov.render(t, target=target.fbo)
    rgb = ov.read_pixels(target.fbo)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def release() -> None:
    """Release the GL context (tests / clean shutdown)."""
    global _ctx, _ov, _target, _size
    if _ctx is not None:
        try:
            _ctx.release()
        except Exception:
            pass
    _ctx = _ov = _target = None
    _size = None
