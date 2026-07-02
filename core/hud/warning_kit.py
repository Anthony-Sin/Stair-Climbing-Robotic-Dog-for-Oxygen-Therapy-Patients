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
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

# arcv is used for its animation-timing model (staggered assemble / eases).  The
# import is optional so the kit still renders (static) if arcv is unavailable.
try:
    from arcv.overlay.anim import out_cubic as _out_cubic
except Exception:  # pragma: no cover
    def _out_cubic(t: float) -> float:
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        f = t - 1.0
        return f * f * f + 1.0


RGB = Tuple[int, int, int]

# ───────────────────────────────── palette ────────────────────────────────────
# Acid-yellow hazard scheme lifted from the ref1 art: bright yellow panels, black
# ink, a single hot-red reserved for danger (target lost / fault).
YELLOW      = (247, 221, 8)      # primary accent / panel fill  (#F7DD08)
YELLOW_HOT  = (255, 238, 60)     # brighter yellow for pulses / lock
YELLOW_DK   = (150, 134, 0)      # dim yellow (inactive strokes, ticks)
BLACK       = (10, 11, 6)        # ink on yellow / black bars
PANEL_BLK   = (18, 19, 12)       # black-card fill
GREY        = (128, 126, 96)     # muted labels on black cards
WHITE       = (238, 238, 224)    # bright readout on black cards
ALERT       = (255, 66, 40)      # danger red — used sparingly (lost / fault)
ALERT_DK    = (120, 26, 16)
FEED_DIM    = 0.34               # how far the non-feed frame is darkened


def bgr(c: RGB) -> Tuple[int, int, int]:
    """RGB → OpenCV BGR."""
    return (int(c[2]), int(c[1]), int(c[0]))


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore


# ───────────────────────────────── fonts ──────────────────────────────────────
# Share Tech Mono.  A copy is bundled next to this module so the runtime works in
# the Docker container without arcv's resources; arcv's copy + system monospace
# fonts are fallbacks.
def _font_candidates() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cands: List[str] = [os.path.join(here, "ShareTechMono-Regular.ttf")]   # bundled
    try:
        import arcv
        cands.append(os.path.join(os.path.dirname(arcv.__file__),
                                  "resources", "fonts", "ShareTechMono-Regular.ttf"))
    except Exception:  # pragma: no cover
        pass
    cands += [
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    return cands


_FONT_CACHE: Dict[int, "ImageFont.FreeTypeFont"] = {}
_FONT_PATH: Optional[str] = None


def load_font(size: int):
    """Return a cached Share Tech Mono PIL font at ``size`` px."""
    global _FONT_PATH
    size = max(6, int(round(size)))
    f = _FONT_CACHE.get(size)
    if f is not None:
        return f
    for path in ([_FONT_PATH] if _FONT_PATH else []) + _font_candidates():
        if not path:
            continue
        try:
            f = ImageFont.truetype(path, size)
            _FONT_PATH = path
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _advance(size: int) -> float:
    """Mono per-glyph advance in px (Share Tech Mono is fixed-pitch)."""
    f = load_font(size)
    try:
        return float(f.getlength("M"))
    except Exception:
        return size * 0.6


def text_width(s: str, size: int, tracking: float = 0.0) -> float:
    if not s:
        return 0.0
    return len(s) * (_advance(size) + tracking) - tracking


# ─────────────────────────── layer: bgr + alpha + text ─────────────────────────
class Layer:
    """A drawing surface: an opaque BGR image, a companion alpha mask (0..255),
    and a Pillow text pass baked on ``flush()``.  cv2 shapes write BGR *and*
    stamp the alpha; text is deferred so labels always sit on top."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = int(w), int(h)
        self.bgr = np.zeros((self.h, self.w, 3), np.uint8)
        self.a = np.zeros((self.h, self.w), np.uint8)
        self._txt = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)) if _PIL_OK else None
        self._draw = ImageDraw.Draw(self._txt) if _PIL_OK else None

    # -- vector (opaque) --------------------------------------------------------
    def fill_poly(self, pts, color: RGB, a: int = 255) -> None:
        p = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(self.bgr, [p], bgr(color), cv2.LINE_AA)
        cv2.fillPoly(self.a, [p], int(a), cv2.LINE_AA)

    def poly(self, pts, color: RGB, width: float = 2.0, closed: bool = True, a: int = 255) -> None:
        p = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.bgr, [p], closed, bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.polylines(self.a, [p], closed, int(a), int(round(width)), cv2.LINE_AA)

    def line(self, p0, p1, color: RGB, width: float = 2.0, a: int = 255) -> None:
        p0 = (int(round(p0[0])), int(round(p0[1])))
        p1 = (int(round(p1[0])), int(round(p1[1])))
        cv2.line(self.bgr, p0, p1, bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.line(self.a, p0, p1, int(a), int(round(width)), cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, color: RGB, width: float = 2.0) -> None:
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, width, closed=True)

    def fill_rect(self, x0, y0, x1, y1, color: RGB, a: int = 255) -> None:
        self.fill_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color, a)

    def disc(self, cx, cy, r, color: RGB, a: int = 255) -> None:
        cv2.circle(self.bgr, (int(cx), int(cy)), int(r), bgr(color), -1, cv2.LINE_AA)
        cv2.circle(self.a, (int(cx), int(cy)), int(r), int(a), -1, cv2.LINE_AA)

    def ring(self, cx, cy, r, color: RGB, width: float = 2.0, a: int = 255,
             a0: float = 0.0, a1: float = 360.0) -> None:
        cv2.ellipse(self.bgr, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1,
                    bgr(color), int(round(width)), cv2.LINE_AA)
        cv2.ellipse(self.a, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1,
                    int(a), int(round(width)), cv2.LINE_AA)

    def paste_bgr(self, sub: np.ndarray, x: int, y: int, a: int = 255) -> None:
        h, w = sub.shape[:2]
        x, y = int(x), int(y)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        self.bgr[y0:y1, x0:x1] = sub[y0 - y:y1 - y, x0 - x:x1 - x]
        self.a[y0:y1, x0:x1] = np.maximum(self.a[y0:y1, x0:x1], a)

    # -- text (deferred) --------------------------------------------------------
    def text(self, s: str, x: float, y: float, size: int, color: RGB,
             anchor: str = "lt", tracking: float = 0.0, alpha: int = 255) -> None:
        if not s or self._draw is None:
            return
        h, v = anchor[0], anchor[1]
        w = text_width(s, size, tracking)
        if h == "c":
            x -= w * 0.5
        elif h == "r":
            x -= w
        vmap = {"t": "t", "m": "m", "b": "s"}
        f = load_font(size)
        adv = _advance(size) + tracking
        col = (int(color[0]), int(color[1]), int(color[2]), int(alpha))
        cx = x
        for ch in s:
            if ch != " ":
                self._draw.text((cx, y), ch, font=f, fill=col, anchor="l" + vmap.get(v, "t"))
            cx += adv

    def flush(self) -> None:
        """Bake the text layer down onto ``bgr``/``a`` (text on top, inside mask)."""
        if self._txt is None:
            return
        rgba = np.asarray(self._txt)  # (H,W,4) RGBA
        ta = rgba[:, :, 3].astype(np.float32) / 255.0
        if ta.max() <= 0:
            return
        tbgr = rgba[:, :, 2::-1]  # RGB→BGR
        m = ta[:, :, None]
        self.bgr[:] = (tbgr.astype(np.float32) * m + self.bgr.astype(np.float32) * (1.0 - m)).astype(np.uint8)
        self.a[:] = np.maximum(self.a, (ta * 255).astype(np.uint8))


def blit(dst: np.ndarray, layer: Layer, x: int, y: int, alpha: float = 1.0,
         reveal: float = 1.0, edge: Optional[RGB] = None) -> None:
    """Alpha-composite ``layer`` onto ``dst`` (BGR) at (x,y).

    ``reveal`` (0..1) roll-opens the layer from the top (a bright ``edge`` line is
    drawn at the wipe front).  ``alpha`` fades the whole layer."""
    layer.flush()
    vis_h = int(round(layer.h * max(0.0, min(1.0, reveal))))
    if vis_h <= 0 or alpha <= 0.0:
        return
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + layer.w), min(dst.shape[0], y + vis_h)
    if x1 <= x0 or y1 <= y0:
        return
    src_bgr = layer.bgr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    src_a = (layer.a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0) * float(alpha)
    m = src_a[:, :, None]
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (src_bgr * m + roi * (1.0 - m)).astype(np.uint8)
    if edge is not None and 0.02 < reveal < 0.995:
        ey = y1 - 1
        if 0 <= ey < dst.shape[0]:
            cv2.line(dst, (x0, ey), (x1 - 1, ey), bgr(edge), 2, cv2.LINE_AA)


# ───────────────────────────── shared geometry ────────────────────────────────
def cut_poly(x0, y0, x1, y1, cut: float, corners: str = "tr,bl") -> List[Tuple[float, float]]:
    """Rectangle with chamfered (cut) corners — the ref1 panel silhouette.

    ``corners`` picks which corners are cut: any of tl,tr,br,bl."""
    c = cut
    cs = set(s.strip() for s in corners.split(",")) if corners else set()
    pts: List[Tuple[float, float]] = []
    # TL
    pts += [(x0 + c, y0)] if "tl" in cs else [(x0, y0)]
    # TR
    pts += [(x1 - c, y0), (x1, y0 + c)] if "tr" in cs else [(x1, y0)]
    # BR
    pts += [(x1, y1 - c), (x1 - c, y1)] if "br" in cs else [(x1, y1)]
    # BL
    pts += [(x0 + c, y1), (x0, y1 - c)] if "bl" in cs else [(x0, y1)]
    # close TL
    if "tl" in cs:
        pts += [(x0, y0 + c)]
    return pts


def hazard_band(w: int, h: int, stripe: int = 13, fg: RGB = BLACK, bg: RGB = YELLOW,
                phase: float = 0.0) -> np.ndarray:
    """A yellow/black diagonal hazard-stripe tile (BGR)."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bgr(bg)
    step = stripe * 2
    off = int(phase * step) % step
    for x in range(-h - step, w + step, step):
        p0 = (x + off, 0)
        p1 = (x + off + h, h)
        cv2.line(img, p0, p1, bgr(fg), stripe, cv2.LINE_AA)
    return img


# ───────────────────────────────── data model ─────────────────────────────────
@dataclass
class Target:
    cx: float           # centre x, normalised [0,1] (image space, top-left origin)
    cy: float           # centre y, normalised
    w: float            # width, normalised
    h: float            # height, normalised
    score: float = 1.0  # detector / tracking confidence 0..1
    dist_m: Optional[float] = None   # estimated range (labelled EST), or None
    tid: int = 1        # contact id
    locked: bool = False


@dataclass
class Telemetry:
    state: str                       # BOOTING/ACQUIRING/LOCKED/TRACKING/TARGET LOST/REACQUIRE
    targets: List[Target] = field(default_factory=list)   # primary first
    fps: float = 0.0
    signal: float = 0.0              # smoothed link/track quality 0..1 (feeds RADAR link, not a card)
    boot: float = 1.0                # chrome assemble progress 0..1
    lost_for: Optional[float] = None # seconds since lock lost (drives ALERT card)
    rec_s: int = 0
    frame_no: int = 0
    sim: bool = False                # feed is simulated (not a real camera)?
    code: str = "17-WW-22-000"
    # instrument-cluster data (persistent yellow visualisations)
    depth: Optional[np.ndarray] = None                 # depth image in mm, or None (no depth cam)
    stairs: Optional[Tuple[bool, float, Optional[Tuple[float, float, float, float]]]] = None  # (det, conf, bbox_norm)
    lidar: Optional[dict] = None                       # {"ranges_m":[...], "view_range_m":..} fwd profile
    # transient UI (spawn-in → hold → spawn-out)
    show_detail: bool = False                          # TARGET detail card currently surfaced?
    toasts: List[Tuple[str, str]] = field(default_factory=list)  # (message, kind) active this frame
    warn: bool = False                                 # explicit hazard alarm (caller-set, e.g. robot fell)
    drive: Optional[str] = None                        # active locomotion backend label (PGTT / BLIND RL)

    NEAR_RANGE_M = 0.8                                  # patient closer than this → proximity colour (not the WARNING)

    @property
    def primary(self) -> Optional[Target]:
        return self.targets[0] if self.targets else None

    @property
    def present(self) -> bool:
        return bool(self.targets)

    @property
    def hazard(self) -> bool:
        """The top WARNING alarm — ONLY when the target is lost (or the robot
        fell).  Proximity does NOT raise it; the HUD stays calm while following."""
        return self.warn or self.state == "TARGET LOST"

    @property
    def near(self) -> bool:
        p = self.primary
        return p is not None and p.dist_m is not None and p.dist_m < self.NEAR_RANGE_M

    @property
    def is_blind_rl(self) -> bool:
        return bool(self.drive) and "BLIND" in self.drive.upper()

    @property
    def accent(self) -> RGB:
        return ALERT if self.hazard else YELLOW


# ──────────────────────────────── card system ─────────────────────────────────
@dataclass
class CardSpec:
    key: str
    title: str
    style: str = "panel"                 # panel | solid | alert
    big: Optional[Tuple[str, RGB]] = None
    rows: List[Tuple[str, str, RGB]] = field(default_factory=list)  # (label, value, value_col)
    bar: Optional[Tuple[float, RGB]] = None
    note: Optional[str] = None
    height: int = 84


class _CardAnim:
    __slots__ = ("spec", "anim")

    def __init__(self, spec: CardSpec) -> None:
        self.spec = spec
        self.anim = 0.0


class CardStack:
    """A column of cards that spawn (roll open) / collapse (roll shut) as the
    active set changes, reflowing so there is no dead space."""

    APPEAR = 0.26   # seconds to open
    VANISH = 0.20   # seconds to close

    def __init__(self, side: str) -> None:
        self.side = side                       # "L" | "R"
        self._cards: Dict[str, _CardAnim] = {}
        self._order: List[str] = []

    def update(self, specs: List[CardSpec], dt: float) -> None:
        want = {s.key: s for s in specs}
        for key, spec in want.items():
            c = self._cards.get(key)
            if c is None:
                c = _CardAnim(spec)
                self._cards[key] = c
                self._order.append(key)
            else:
                c.spec = spec
        for key, c in self._cards.items():
            if key in want:
                c.anim = min(1.0, c.anim + dt / self.APPEAR)
            else:
                c.anim = max(0.0, c.anim - dt / self.VANISH)
        # drop fully-collapsed cards that are no longer wanted
        dead = [k for k, c in self._cards.items() if k not in want and c.anim <= 0.001]
        for k in dead:
            self._cards.pop(k, None)
            self._order.remove(k)
        # keep order: wanted specs define ordering; closing cards keep last position
        wanted_order = [s.key for s in specs]
        self._order = wanted_order + [k for k in self._order if k not in want]

    def draw(self, dst: np.ndarray, x: int, y_top: int, w: int, gap: int,
             t: float) -> None:
        y = y_top
        for key in self._order:
            c = self._cards.get(key)
            if c is None:
                continue
            e = _out_cubic(c.anim)
            layer = _render_card(c.spec, w, t)
            vis = e
            blit(dst, layer, x, y, alpha=min(1.0, c.anim * 1.4), reveal=vis,
                 edge=YELLOW_HOT if c.spec.style != "alert" else ALERT)
            y += int(round(layer.h * e)) + gap


def _render_card(spec: CardSpec, w: int, t: float) -> Layer:
    """Render one card to its own opaque tile (ref1 hazard-panel look).

    Height is derived from the content (header + optional big line + rows +
    optional bar) so nothing ever clips and the column can pack tightly."""
    s = w / 275.0                        # scale relative to a 275px reference card
    pad = int(12 * s)
    header_h = int(30 * s)
    title_sz = max(11, int(15 * s))
    big_sz = max(20, int(33 * s))
    big_lh = int(43 * s)
    label_sz = max(10, int(13 * s))
    row_h = int(24 * s)
    bar_zone = int(24 * s)

    style = spec.style                       # solid | panel | alert | bracket | underline
    solid = style == "solid"
    is_alert = style == "alert"
    open_style = style in ("bracket", "underline")

    # ---- measure -> height
    h = header_h + int(9 * s)
    if spec.big is not None:
        h += big_lh
    h += row_h * len(spec.rows)
    if spec.bar is not None:
        h += bar_zone
    h += int(10 * s)
    L = Layer(w, h)
    cut = max(6, int(w * 0.05))

    fill = YELLOW if solid else PANEL_BLK
    accent = ALERT if is_alert else YELLOW
    label_col = _mix(YELLOW, BLACK, 0.6) if solid else GREY
    val_default = BLACK if solid else YELLOW
    body = cut_poly(1, 1, w - 1, h - 1, cut, corners="tr,bl")
    L.fill_poly(body, fill)
    ind = pad + (int(9 * s) if style == "bracket" else 0)   # content indent (past the spine)

    # ---- frame + title, distinct per style ------------------------------------
    if style == "bracket":
        # dark card framed by accent corner L-brackets, a left accent spine, and
        # a cut-corner title TAB (reads as a targeting readout).
        L.poly(body, _mix(YELLOW, PANEL_BLK, 0.4), 1, closed=True)
        leg = int(18 * s)
        for bx, by, sx, sy in ((3, 3, 1, 1), (w - 3, 3, -1, 1), (3, h - 3, 1, -1), (w - 3, h - 3, -1, -1)):
            L.poly([(bx + sx * leg, by), (bx, by), (bx, by + sy * leg)], accent, 2, closed=False)
        L.fill_rect(3, header_h * 0.5, int(4 * s) + 3, h - 4, accent, 255)      # left spine
        tabw = int(text_width(spec.title, title_sz, 1.6)) + int(18 * s)
        L.fill_poly(cut_poly(3, 3, 3 + tabw, header_h, int(8 * s), corners="br"), accent)
        L.text(spec.title, pad, header_h * 0.5, title_sz, BLACK, anchor="lm", tracking=1.6)
    elif style == "underline":
        # dark card, title with an accent underline (no header bar) + corner ticks.
        L.poly(body, _mix(YELLOW, PANEL_BLK, 0.4), 1, closed=True)
        L.text(spec.title, pad, header_h * 0.5, title_sz, accent, anchor="lm", tracking=1.8)
        L.line((pad, header_h - int(3 * s)), (w - pad, header_h - int(3 * s)),
               _mix(accent, PANEL_BLK, 0.15), 2)
        L.line((w - 3, 3), (w - 3 - int(12 * s), 3), accent, 2)
        L.line((w - 3, 3), (w - 3, 3 + int(12 * s)), accent, 2)
    elif is_alert:
        L.poly(body, ALERT, max(2, int(2 * s)), closed=True)
        band = hazard_band(w - 3, header_h - 2, stripe=int(10 * s), phase=t * 30.0)
        L.paste_bgr(band, 2, 2)
        blink = 0.55 + 0.45 * math.sin(t * 12.0)
        tw = int(text_width(spec.title, title_sz, 2.0)) + int(16 * s)
        L.fill_rect(w * 0.5 - tw * 0.5, 3, w * 0.5 + tw * 0.5, header_h - 1, BLACK, 235)
        L.text(spec.title, w * 0.5, header_h * 0.5, title_sz, _mix(ALERT, YELLOW, blink),
               anchor="cm", tracking=2.0)
    else:                                    # solid | panel — filled header bar
        L.poly(body, (BLACK if solid else YELLOW), max(2, int(2 * s)), closed=True)
        head_bg = BLACK if solid else YELLOW
        head_fg = YELLOW if solid else BLACK
        L.fill_rect(2, 2, w - 2, header_h, head_bg, 255)
        L.text(spec.title, pad, header_h * 0.5, title_sz, head_fg, anchor="lm", tracking=1.6)
        for i in range(3):                   # index ticks on the header right
            tx = w - pad - i * int(7 * s)
            L.line((tx, 6), (tx, header_h - 5), _mix(head_fg, head_bg, 0.45), 2)

    # ---- content (big value / rows / bar) — shared -----------------------------
    cy = header_h + int(9 * s)
    if spec.big is not None:
        txt, col = spec.big
        L.text(txt, ind, cy, big_sz, col, anchor="lt", tracking=1.0)
        cy += big_lh
    for (label, value, vcol) in spec.rows:
        L.text(label, ind, cy + row_h * 0.5, label_sz, label_col, anchor="lm", tracking=1.0)
        L.text(value, w - pad, cy + row_h * 0.5, label_sz + int(2 * s),
               vcol if vcol is not None else val_default, anchor="rm", tracking=0.5)
        cy += row_h
    if spec.bar is not None:
        frac, col = spec.bar
        _seg_bar(L, ind, cy + int(3 * s), w - pad - ind, int(9 * s), frac, col,
                 track=_mix(YELLOW, BLACK, 0.55) if solid else _mix(PANEL_BLK, YELLOW_DK, 0.3))
    if spec.note:
        note_col = accent if open_style else (YELLOW if solid else BLACK)
        L.text(spec.note, w - pad, header_h * 0.5, label_sz - 1, note_col, anchor="rm", tracking=0.5)
    return L


def _seg_bar(L: Layer, x, y, w, h, frac: float, col: RGB, track: RGB, segs: int = 16) -> None:
    frac = max(0.0, min(1.0, frac))
    gap = max(1, int(w * 0.008))
    sw = (w - gap * (segs - 1)) / segs
    lit = int(round(frac * segs))
    for i in range(segs):
        sx = x + i * (sw + gap)
        L.fill_rect(sx, y, sx + sw, y + h, col if i < lit else track, 255)


# ──────────────────────────────── the HUD ─────────────────────────────────────
class WarningHud:
    """Stateful compositor.  Construct once at a size, call :meth:`render` per
    frame with a live BGR frame + :class:`Telemetry`.

    The camera **is** the whole screen; the HUD floats opaque over it — a top
    WARNING bar, a thin footer, target brackets/reticle on the live feed, two
    persistent yellow instrument panels (DEPTH raster + forward LiDAR radar), a
    small persistent STATUS card, plus transient cards/toasts that spawn-in and
    spawn-out (info you don't need at a glance)."""

    def __init__(self, size: Tuple[int, int] = (1280, 720)) -> None:
        self.W, self.H = int(size[0]), int(size[1])
        self.left = CardStack("L")
        self.right = CardStack("R")
        self._vig: Optional[np.ndarray] = None
        self.col_w = int(self.W * 0.220)
        self._toasts: Dict[str, List] = {}       # msg -> [anim, kind]
        # instrument panels (bottom corners, over the full-screen feed)
        pw, ph = int(self.W * 0.232), int(self.H * 0.300)
        self.depth_rect = (int(self.W * 0.020), int(self.H * 0.930) - ph,
                           int(self.W * 0.020) + pw, int(self.H * 0.930))
        rw, rh = int(self.W * 0.248), int(self.H * 0.300)
        self.radar_rect = (int(self.W * 0.980) - rw, int(self.H * 0.930) - rh,
                           int(self.W * 0.980), int(self.H * 0.930))

    # -- helpers ----------------------------------------------------------------
    def _vignette(self, frame: np.ndarray) -> None:
        if self._vig is None or self._vig.shape[:2] != frame.shape[:2]:
            h, w = frame.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            nx = (xx / w - 0.5) * 2.0
            ny = (yy / h - 0.5) * 2.0
            rad = np.sqrt(nx * nx * 0.9 + ny * ny)
            v = 1.0 - 0.5 * np.clip((rad - 0.45) / 0.9, 0.0, 1.0) ** 1.5
            self._vig = v[:, :, None].astype(np.float32)
        frame[:] = (frame.astype(np.float32) * self._vig).astype(np.uint8)

    @staticmethod
    def _scanlines(frame: np.ndarray, step: int = 3, strength: float = 0.06) -> None:
        frame[::step] = (frame[::step].astype(np.float32) * (1.0 - strength)).astype(np.uint8)

    @staticmethod
    def _depth_color(depth: np.ndarray, near: float = 280.0, far: float = 5200.0) -> np.ndarray:
        """Colour-map a depth image (mm) so structure is actually readable —
        near = warm (red/orange), far = cool (blue), via TURBO.  Returns BGR
        uint8.  (A flat yellow ramp hid the shape; this keeps the panel useful
        while the yellow chrome around it carries the HUD's vibe.)"""
        d = np.clip((depth.astype(np.float32) - near) / (far - near), 0.0, 1.0)
        near_hot = np.clip((1.0 - d) * 255.0, 0, 255).astype(np.uint8)   # near -> 255 (warm end)
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        return cv2.applyColorMap(near_hot, cmap)

    # -- main -------------------------------------------------------------------
    def render(self, frame: np.ndarray, tel: Telemetry, t: float, dt: float) -> np.ndarray:
        if frame.shape[1] != self.W or frame.shape[0] != self.H:
            frame = cv2.resize(frame, (self.W, self.H))
        else:
            frame = frame.copy()
        W, H = self.W, self.H
        boot = tel.boot

        # 1) the feed is the ENTIRE screen — just condition it for legibility
        self._vignette(frame)
        frame[:] = (frame.astype(np.float32) * FEED_DIM).astype(np.uint8)
        self._scanlines(frame)

        # 2) chrome + on-feed target reticle (drawn onto one full-frame layer)
        top = Layer(W, H)
        self._header(top, tel, t, boot)
        self._footer(top, tel, t, boot)
        self._targets(top, frame, tel, t, boot)
        blit(frame, top, 0, 0, alpha=1.0, reveal=1.0)

        # 3) persistent instrument panels (bottom corners)
        self._depth_panel(frame, tel, t, boot)
        self._radar_panel(frame, tel, t, boot)

        # 4) dynamic cards — STATUS persistent (L), transient detail/alert (R)
        left_specs, right_specs = self._card_specs(tel)
        self.left.update(left_specs, dt)
        self.right.update(right_specs, dt)
        cyt = int(H * 0.100)
        self.left.draw(frame, int(W * 0.020), cyt, self.col_w, int(H * 0.016), t)
        self.right.draw(frame, W - int(W * 0.020) - self.col_w, cyt, self.col_w,
                        int(H * 0.016), t)

        # 5) transient toasts (spawn-in → hold → spawn-out) + lost banner
        self._draw_toasts(frame, tel, t, dt)
        if tel.state == "TARGET LOST":
            self._lost_banner(frame, tel, t)
        return frame

    # -- chrome pieces ----------------------------------------------------------
    def _header(self, L: Layer, tel: Telemetry, t: float, boot: float) -> None:
        W, H = self.W, self.H
        y0, y1 = int(H * 0.020), int(H * 0.086)
        x0, x1 = int(W * 0.020), int(W * 0.980)
        rv = _out_cubic(min(1.0, boot / 0.5))
        if rv <= 0:
            return
        # The big word is a STATUS/ALARM word: "WARNING" (red, blinking) appears
        # ONLY when the target is lost (or the robot fell); during normal following
        # it reads a calm status word — so WARNING is not sitting on-screen by default.
        haz = tel.hazard
        acc = tel.accent
        L.fill_rect(x0, y0, x1, y1, BLACK, int(240 * rv))
        L.rect(x0, y0, x1, y1, acc, 2)
        if haz:
            hb = hazard_band(int(W * 0.020), y1 - y0, stripe=9, fg=ALERT, bg=BLACK, phase=t * 34.0)
            ww = 0.5 + 0.5 * math.sin(t * 9.0)
            word, warn_col = "WARNING", _mix(_mix(BLACK, ALERT, 0.35), ALERT, ww)
        else:
            hb = hazard_band(int(W * 0.020), y1 - y0, stripe=9, phase=t * 26.0)
            word = "STANDBY" if tel.state == "STANDBY" else ("TRACKING" if tel.present else "SCANNING")
            warn_col = YELLOW
        L.paste_bgr(hb, x0, y0)
        wx = x0 + int(W * 0.020) + 14
        L.text(word, wx, (y0 + y1) * 0.5, int(H * 0.052), warn_col, anchor="lm", tracking=3.0)
        sub_x = wx + int(text_width("SCANNING", int(H * 0.052), 3.0)) + 22   # fixed → subtitle doesn't jump
        L.line((sub_x - 11, y0 + 8), (sub_x - 11, y1 - 8), YELLOW_DK, 2)
        L.text("TARGET ACQUISITION SYSTEM", sub_x, y0 + (y1 - y0) * 0.30,
               int(H * 0.021), YELLOW, anchor="lm", tracking=2.0)
        L.text("O2-THERAPY ESCORT // STAIR-ASSIST DOG", sub_x, y0 + (y1 - y0) * 0.72,
               int(H * 0.017), GREY, anchor="lm", tracking=1.5)
        self._drive_chip(L, tel, t, y0, y1)
        # right: chevrons + status code + REC
        chev_x = x1 - int(W * 0.020) - 8
        for i in range(4):
            cx = chev_x - i * 15
            col = acc if ((int(t * 6) - i) % 4 == 0) else YELLOW_DK
            L.poly([(cx - 8, y0 + 12), (cx, (y0 + y1) * 0.5), (cx - 8, y1 - 12)], col, 3, closed=False)
        L.text(f"CODE {tel.code}", x1 - int(W * 0.020) - 78, y0 + (y1 - y0) * 0.30,
               int(H * 0.017), YELLOW, anchor="rm", tracking=1.5)
        blink = (t % 1.0) < 0.5
        L.text(f"REC {tel.rec_s:04d}", x1 - int(W * 0.020) - 78, y0 + (y1 - y0) * 0.72,
               int(H * 0.017), ALERT if not blink else _mix(ALERT, BLACK, 0.4),
               anchor="rm", tracking=1.5)

    _ORANGE = (255, 140, 20)                      # experimental / blind-RL accent (distinct from yellow)

    def _drive_chip(self, L: Layer, tel: Telemetry, t: float, y0: int, y1: int) -> None:
        """A prominent header chip for the active locomotion backend.  BLIND-RL is
        highlighted (orange + pulse) because it is the experimental climber, so it
        stands out clearly from PGTT."""
        if not tel.drive:
            return
        H, W = self.H, self.W
        blind = tel.is_blind_rl
        cs = int(H * 0.021)
        tagw = int(text_width("DRIVE", int(H * 0.014), 1.0)) + int(W * 0.012)
        lw = int(text_width(tel.drive, cs, 1.5))
        w = tagw + lw + int(W * 0.020)
        cxp = int(W * 0.635)
        px0, px1 = cxp - w // 2, cxp + w // 2
        py0, py1 = y0 + int((y1 - y0) * 0.17), y1 - int((y1 - y0) * 0.17)
        if blind:
            pulse = 0.55 + 0.45 * math.sin(t * 6.5)
            col = _mix(_mix(BLACK, self._ORANGE, 0.5), self._ORANGE, pulse)
        else:
            col = YELLOW
        poly = cut_poly(px0, py0, px1, py1, int(H * 0.010), corners="tr,bl")
        L.fill_poly(poly, BLACK, 235)
        L.poly(poly, col, 2, closed=True)
        L.fill_rect(px0 + 2, py0 + 2, px0 + tagw, py1 - 2, col, 255)          # "DRIVE" tag
        L.text("DRIVE", px0 + int(W * 0.006), (y0 + y1) * 0.5, int(H * 0.014), BLACK,
               anchor="lm", tracking=1.0)
        L.text(tel.drive, px0 + tagw + int(W * 0.006), (y0 + y1) * 0.5, cs, col,
               anchor="lm", tracking=1.5)
        if blind:                                 # a small blinking hazard dot to draw the eye
            r = 3 + 1.5 * (0.5 + 0.5 * math.sin(t * 6.5))
            L.disc(px1 + int(W * 0.010), (y0 + y1) * 0.5, r, self._ORANGE)

    def _footer(self, L: Layer, tel: Telemetry, t: float, boot: float) -> None:
        W, H = self.W, self.H
        rv = _out_cubic(min(1.0, max(0.0, boot - 0.25) / 0.5))
        if rv <= 0:
            return
        y = int(H * 0.955)
        x0, x1 = int(W * 0.020), int(W * 0.980)
        L.line((x0, y), (x0 + int((x1 - x0) * rv), y), YELLOW_DK, 2)
        if rv > 0.98:                              # subtle tick ruler
            for i in range(41):
                tx = x0 + (x1 - x0) * (i / 40.0)
                th = 7 if i % 5 == 0 else 4
                L.line((tx, y), (tx, y - th), YELLOW_DK, 1)
        # one legible row, all Share Tech Mono: device id · fps · feed source.
        fs = int(H * 0.024)
        yb = y + int(H * 0.020)
        L.text(tel.code, x0, yb, fs, YELLOW, anchor="lm", tracking=1.5)
        if tel.fps > 0:
            L.text(f"{tel.fps:4.1f} FPS", W * 0.5, yb, fs, YELLOW, anchor="cm", tracking=1.5)
        mode = "SIMULATED FEED" if tel.sim else "LIVE OPTICAL FEED // CAM-01"
        L.text(mode, x1, yb, fs, YELLOW if not tel.sim else ALERT, anchor="rm", tracking=1.5)

    def _targets(self, L: Layer, frame: np.ndarray, tel: Telemetry, t: float, boot: float) -> None:
        if boot < 0.55:
            return
        W, H = self.W, self.H
        y_lo, y_hi = int(H * 0.100), int(H * 0.925)      # keep clear of header/footer

        # The bbox can be enormous when the patient is right in front of the dog,
        # so the reticle is a CAPPED targeting box (a modest, consistent size that
        # sits on the target centre) rather than a frame around the whole body,
        # and its centre is clamped to stay on-screen when the patient is off to a
        # side — both fix the "too big / looks weird at the edges" problems.
        def to_px(tg: Target, cap: bool) -> Tuple[int, int, int, int]:
            hw = tg.w * W * 0.5
            hh = tg.h * H * 0.5
            if cap:
                hw = min(max(hw, 0.045 * W), 0.105 * W)
                hh = min(max(hh, 0.070 * H), 0.180 * H)
            else:
                hw, hh = max(14, hw), max(18, hh)
            cx = float(np.clip(tg.cx * W, 0.06 * W, 0.94 * W))
            cy = float(np.clip(tg.cy * H, y_lo + hh, y_hi - hh))
            return int(cx), int(cy), int(hw), int(hh)

        # secondary targets: light corner brackets
        for tg in tel.targets[1:]:
            cx, cy, hw, hh = to_px(tg, cap=True)
            for ox, oy, sx, sy in ((-hw, -hh, 1, 1), (hw, -hh, -1, 1),
                                   (-hw, hh, 1, -1), (hw, hh, -1, -1)):
                L.poly([(cx + ox + sx * 12, cy + oy), (cx + ox, cy + oy),
                        (cx + ox, cy + oy + sy * 12)], YELLOW_DK, 2, closed=False)
            L.text(f"C-{tg.tid:02d}", cx, cy - hh - 8, int(H * 0.017), YELLOW,
                   anchor="cb", tracking=1.0)

        p = tel.primary
        if p is None:
            self._scan_reticle(L, W // 2, H // 2, int(min(W, H) * 0.13), t, tel.accent)
            return

        cx, cy, hw, hh = to_px(p, cap=True)
        acc = ALERT if tel.hazard else (YELLOW_HOT if p.locked else YELLOW)
        pulse = 1.0 if p.locked else (1.0 + 0.05 * math.sin(t * 5.0))
        hw2, hh2 = int(hw * pulse), int(hh * pulse)
        leg = int(min(30, max(14, min(hw2, hh2) * 0.4)))
        for ox, oy, sx, sy in ((-hw2, -hh2, 1, 1), (hw2, -hh2, -1, 1),
                               (-hw2, hh2, 1, -1), (hw2, hh2, -1, -1)):
            bx, by = cx + ox, cy + oy
            L.poly([(bx + sx * leg, by), (bx, by), (bx, by + sy * leg)], acc, 3, closed=False)
            if p.locked:
                L.line((bx, by), (bx - sx * 9, by - sy * 9), _mix(acc, BLACK, 0.4), 2)
        # crosshair + centre dot
        g = 9
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            L.line((cx + dx * g, cy + dy * g), (cx + dx * g * 2.4, cy + dy * g * 2.4), acc, 2)
        L.disc(cx, cy, 3, acc)
        if p.locked:                              # a modest lock ring, never the giant circle
            r = int(min(min(hw2, hh2) * 0.85, 0.085 * H))
            L.ring(cx, cy, r, _mix(acc, BLACK, 0.45), 1)
        # callout BELOW the target, clamped on-screen — keeps top-centre clear for toasts
        tag = "PATIENT-01 // LOCK" if p.locked else "PATIENT-01 // TRACK"
        if tel.state == "TARGET LOST":
            tag = "CONTACT DROPPED"
        ty = min(int(H * 0.880), cy + hh2 + int(H * 0.014))
        tw = int(text_width(tag, int(H * 0.020), 1.5)) + 18
        # clamp the callout on-screen; when it drops into the bottom-panel row,
        # keep it clear of the DEPTH/LiDAR columns so it never hides behind them.
        lo, hi = tw // 2 + int(0.02 * W), W - tw // 2 - int(0.02 * W)
        if ty + int(H * 0.026) > 0.62 * H:
            lo = max(lo, int(0.255 * W) + tw // 2 + 4)
            hi = min(hi, int(0.735 * W) - tw // 2 - 4)
        tcx = W // 2 if lo > hi else int(np.clip(cx, lo, hi))
        L.fill_poly(cut_poly(tcx - tw // 2, ty, tcx + tw // 2, ty + int(H * 0.026),
                             int(H * 0.012), corners="tr,bl"), acc)
        L.text(tag, tcx, ty + int(H * 0.013), int(H * 0.020), BLACK, anchor="cm", tracking=1.5)
        if p.dist_m is not None:                  # range reddens on proximity (not the top WARNING)
            L.text(f"{p.dist_m:0.2f}M EST", tcx, ty + int(H * 0.031),
                   int(H * 0.020), ALERT if tel.near else acc, anchor="ct", tracking=1.5)

    def _scan_reticle(self, L: Layer, cx, cy, r, t: float, acc: RGB) -> None:
        L.ring(cx, cy, r, _mix(acc, BLACK, 0.15), 2)
        L.ring(cx, cy, int(r * 0.6), _mix(acc, BLACK, 0.5), 1)
        a = t * 2.0
        L.line((cx, cy), (cx + math.cos(a) * r, cy + math.sin(a) * r), acc, 2)
        for k in range(4):
            ka = k * math.pi / 2 + t * 0.5
            L.line((cx + math.cos(ka) * r * 1.08, cy + math.sin(ka) * r * 1.08),
                   (cx + math.cos(ka) * r * 1.24, cy + math.sin(ka) * r * 1.24), acc, 2)

    def _lost_banner(self, frame: np.ndarray, tel: Telemetry, t: float) -> None:
        W, H = self.W, self.H
        L = Layer(W, H)
        blink = 0.5 + 0.5 * math.sin(t * 10.0)
        bw, bh = int(W * 0.34), int(H * 0.11)
        cx, cy = W // 2, int(H * 0.30)
        poly = cut_poly(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2,
                        int(H * 0.02), corners="tr,bl")
        L.fill_poly(poly, BLACK, 235)
        L.poly(poly, ALERT, 3, closed=True)
        band = hazard_band(bw - 6, int(bh * 0.24), stripe=11, fg=ALERT, bg=BLACK, phase=t * 24.0)
        L.paste_bgr(band, cx - bw // 2 + 3, cy - bh // 2 + 3)
        L.text("TARGET LOST", cx, cy + int(bh * 0.05), int(H * 0.055),
               _mix(BLACK, ALERT, 0.5 + 0.5 * blink), anchor="cm", tracking=4.0)
        cd = f"REACQUIRING  {tel.lost_for:0.1f}S" if tel.lost_for is not None else "REACQUIRING"
        L.text(cd, cx, cy + int(bh * 0.34), int(H * 0.022), YELLOW, anchor="cm", tracking=2.0)
        blit(frame, L, 0, 0, alpha=1.0, reveal=1.0)

    # -- instrument panels ------------------------------------------------------
    def _panel_shell(self, L: Layer, w: int, h: int, title: str, t: float,
                     badge: Optional[Tuple[str, RGB]] = None) -> int:
        """Draw a black cut-corner instrument panel with a yellow header; return
        the content-top y."""
        cut = max(8, int(w * 0.05))
        body = cut_poly(1, 1, w - 1, h - 1, cut, corners="tr,bl")
        L.fill_poly(body, PANEL_BLK, 236)
        L.poly(body, YELLOW, 2, closed=True)
        hh = int(h * 0.135)
        L.fill_rect(2, 2, w - 2, hh, YELLOW, 255)
        L.text(title, 12, hh * 0.5, int(hh * 0.62), BLACK, anchor="lm", tracking=1.8)
        if badge is not None:
            L.text(badge[0], w - 12, hh * 0.5, int(hh * 0.55), badge[1], anchor="rm", tracking=1.0)
        return hh

    def _depth_panel(self, frame: np.ndarray, tel: Telemetry, t: float, boot: float) -> None:
        if tel.depth is None:
            return
        rv = _out_cubic(min(1.0, max(0.0, boot - 0.45) / 0.5))
        if rv <= 0:
            return
        x0, y0, x1, y1 = self.depth_rect
        w, h = x1 - x0, y1 - y0
        L = Layer(w, h)
        det = bool(tel.stairs and tel.stairs[0])
        conf = float(tel.stairs[1]) if tel.stairs else 0.0
        badge = (f"STAIRS {conf * 100:0.0f}%", YELLOW_HOT) if det else ("SCAN", GREY)
        hh = self._panel_shell(L, w, h, "DEPTH // D435", t, badge)
        # depth raster
        pad = int(w * 0.03)
        cx0, cy0 = pad, hh + pad
        cw, ch = w - 2 * pad, h - hh - 2 * pad
        ramp = self._depth_color(tel.depth)
        ramp = cv2.resize(ramp, (cw, ch), interpolation=cv2.INTER_LINEAR)
        L.paste_bgr(ramp, cx0, cy0)
        L.rect(cx0, cy0, cx0 + cw, cy0 + ch, YELLOW, 1)
        # NEAR/FAR legend on the colour ramp
        L.text("NEAR", cx0 + 3, cy0 + 2, int(h * 0.05), WHITE, anchor="lt", tracking=0.5)
        L.text("FAR", cx0 + cw - 3, cy0 + ch - 2, int(h * 0.05), WHITE, anchor="rb", tracking=0.5)
        # stair box overlay
        if det and tel.stairs and tel.stairs[2] is not None:
            bx0, by0, bx1, by1 = tel.stairs[2]
            rx0, ry0 = cx0 + bx0 * cw, cy0 + by0 * ch
            rx1, ry1 = cx0 + bx1 * cw, cy0 + by1 * ch
            L.rect(rx0, ry0, rx1, ry1, YELLOW_HOT, 2)
            L.text("STAIRS", (rx0 + rx1) * 0.5, ry0 - 2, int(h * 0.05), YELLOW_HOT,
                   anchor="cb", tracking=1.0)
        # scan sweep
        if rv > 0.98:
            sw = 0.5 + 0.5 * math.sin(t * 1.3)
            sx = cx0 + cw * sw
            L.line((sx, cy0), (sx, cy0 + ch), _mix(YELLOW, PANEL_BLK, 0.35), 1)
        blit(frame, L, x0, y0, alpha=1.0, reveal=rv, edge=YELLOW_HOT)

    def _radar_panel(self, frame: np.ndarray, tel: Telemetry, t: float, boot: float) -> None:
        """Forward obstacle field as a dense clearance bar-graph (the ref1
        bar-array vibe): one bar per bearing bin, height = clearance (tall =
        open, red dips = close obstacles), the tracked patient's bearing flagged,
        range gridlines + FWD/MIN/STATUS readouts.  Far more data than a radar
        disc, and it fills the panel."""
        if tel.lidar is None:
            return
        rv = _out_cubic(min(1.0, max(0.0, boot - 0.5) / 0.5))
        if rv <= 0:
            return
        x0, y0, x1, y1 = self.radar_rect
        w, h = x1 - x0, y1 - y0
        L = Layer(w, h)
        ranges = tel.lidar.get("ranges_m") or []
        view = max(0.5, float(tel.lidar.get("view_range_m", 6.0)))
        n = len(ranges)
        link = "LINK OK" if tel.signal > 0.35 else "WEAK"
        hh = self._panel_shell(L, w, h, "LIDAR // XT16", t,
                               (link, YELLOW if tel.signal > 0.35 else ALERT))
        sm = max(9, int(h * 0.050))
        pad = int(w * 0.030)
        gx0, gx1 = pad, w - pad
        gy0, gy1 = hh + int(h * 0.09), h - int(h * 0.190)
        gw, gh = gx1 - gx0, gy1 - gy0

        def range_at(ang: float) -> float:               # ang rad in [-pi/2, pi/2]
            if not n:
                return view
            idx = int(((ang + math.pi / 2) / math.pi) * (n - 1))
            v = float(ranges[max(0, min(n - 1, idx))])
            return v if v > 0 else view

        # range gridlines (dashed) at clearance fractions, labelled
        for rm in (2.0, 4.0):
            yy = gy1 - gh * min(1.0, rm / view)
            for xx in range(int(gx0), int(gx1), 10):
                L.line((xx, yy), (xx + 5, yy), _mix(YELLOW, PANEL_BLK, 0.62), 1)
            L.text(f"{rm:0.0f}M", gx1, yy - 2, sm - 1, GREY, anchor="rb", tracking=0.5)

        # tracked-target bearing bin
        tgt_bin = None
        p = tel.primary
        if p is not None and p.dist_m is not None:
            taz = max(-math.pi / 2, min(math.pi / 2, (p.cx - 0.5) * 1.05))
            tgt_bin = int(((taz + math.pi / 2) / math.pi) * 33)

        # clearance bars
        N = 34
        bw = gw / N
        near_val = view
        for i in range(N):
            ang = -math.pi / 2 + (i / (N - 1)) * math.pi
            rr = range_at(ang)
            near_val = min(near_val, rr)
            bh = max(2, min(1.0, rr / view) * gh)
            bx = gx0 + i * bw
            danger, caution = rr < 0.8, rr < 1.5
            col = ALERT if danger else (_mix(YELLOW, ALERT, 0.4) if caution else YELLOW)
            if tgt_bin == i:
                col = YELLOW_HOT
            L.fill_rect(bx + 1, gy1 - bh, bx + bw - 0.6, gy1, col)
            if tgt_bin == i:                              # target flag above its bar
                mx, my = bx + bw * 0.5, gy1 - bh - int(h * 0.018)
                L.fill_poly([(mx - 4, my - 5), (mx + 4, my - 5), (mx, my)], YELLOW_HOT)
                L.text("TGT", mx, my - 6, sm - 2, YELLOW_HOT, anchor="cb", tracking=0.5)

        # baseline + bearing axis
        L.line((gx0, gy1), (gx1, gy1), YELLOW, 1)
        L.line(((gx0 + gx1) * 0.5, gy1), ((gx0 + gx1) * 0.5, gy1 + 4), YELLOW, 1)
        L.text("L90", gx0, gy1 + 3, sm - 1, GREY, anchor="lt", tracking=0.5)
        L.text("FWD", (gx0 + gx1) * 0.5, gy1 + 3, sm - 1, YELLOW, anchor="ct", tracking=0.5)
        L.text("R90", gx1, gy1 + 3, sm - 1, GREY, anchor="rt", tracking=0.5)

        # sweep highlight column
        if rv > 0.98:
            sx = gx0 + gw * (0.5 + 0.5 * math.sin(t * 0.9))
            L.line((sx, gy0), (sx, gy1), _mix(YELLOW, PANEL_BLK, 0.35), 1)

        # numeric readout row
        status = ("DANGER", ALERT) if near_val < 0.8 else \
                 (("CAUTION", _mix(YELLOW, ALERT, 0.4)) if near_val < 1.5 else ("CLEAR", YELLOW))
        yb = h - int(h * 0.068)
        L.text(f"FWD {range_at(0.0):0.1f}M", gx0, yb, sm, YELLOW, anchor="lm", tracking=0.5)
        L.text(f"MIN {near_val:0.1f}M", (gx0 + gx1) * 0.5, yb, sm, WHITE, anchor="cm", tracking=0.5)
        L.text(status[0], gx1, yb, sm, status[1], anchor="rm", tracking=0.5)
        blit(frame, L, x0, y0, alpha=1.0, reveal=rv, edge=YELLOW_HOT)

    # -- transient toasts (spawn-in → hold → spawn-out) -------------------------
    def _draw_toasts(self, frame: np.ndarray, tel: Telemetry, t: float, dt: float) -> None:
        want = {msg: kind for (msg, kind) in tel.toasts}
        for msg, kind in want.items():
            e = self._toasts.get(msg)
            if e is None:
                self._toasts[msg] = [0.0, kind]
            else:
                e[1] = kind
        for msg, e in list(self._toasts.items()):
            e[0] = min(1.0, e[0] + dt / 0.18) if msg in want else max(0.0, e[0] - dt / 0.18)
            if e[0] <= 0.0 and msg not in want:
                self._toasts.pop(msg, None)
        if not self._toasts:
            return
        W, H = self.W, self.H
        y = int(H * 0.100)
        for msg, (anim, kind) in list(self._toasts.items())[-2:]:   # keep the stack short
            e = _out_cubic(anim)
            col = ALERT if kind == "alert" else (YELLOW_HOT if kind == "ok" else YELLOW)
            size = int(H * 0.024)
            tw = int(text_width(msg, size, 2.0)) + int(W * 0.03)
            th = int(H * 0.040)
            L = Layer(tw, th)
            poly = cut_poly(1, 1, tw - 1, th - 1, int(th * 0.32), corners="tr,bl")
            L.fill_poly(poly, BLACK, 232)
            L.poly(poly, col, 2, closed=True)
            L.fill_rect(1, 1, int(th * 0.16), th - 1, col, 255)   # accent spine
            L.text(msg, tw * 0.5 + int(th * 0.08), th * 0.5, size, col, anchor="cm", tracking=2.0)
            blit(frame, L, (W - tw) // 2, y, alpha=e, reveal=1.0)
            y += int(th * e) + int(H * 0.010)

    # -- card specs from telemetry ---------------------------------------------
    def _card_specs(self, tel: Telemetry) -> Tuple[List[CardSpec], List[CardSpec]]:
        # Deliberately MIXED card styles (solid / bracket / underline / alert) so
        # the panels read distinctly instead of all looking the same.  (DRIVE now
        # lives in the header chip, so it is not repeated here.)
        p = tel.primary
        left: List[CardSpec] = [                          # solid yellow — the headline plate
            CardSpec("status", "STATUS", style="solid",
                     big=(tel.state, ALERT if tel.state == "TARGET LOST" else BLACK),
                     rows=[("MODE", "FOLLOW+ASSIST", BLACK),
                           ("TARGETS", f"{len(tel.targets):02d}  O2 PATIENT", BLACK)],
                     height=int(self.H * 0.20)),
        ]

        right: List[CardSpec] = []
        if tel.state == "TARGET LOST":                    # hazard-header alert card
            right.append(CardSpec(
                "alert", "! ALERT", style="alert",
                big=("LOCK LOST", ALERT),
                rows=[("LAST SEEN", f"{tel.lost_for:0.1f}S" if tel.lost_for is not None else "--", ALERT),
                      ("ACTION", "SCAN SWEEP", WHITE)]))
        if p is not None and tel.show_detail:             # bracket / targeting-readout card
            dm = p.dist_m
            near = dm is not None and dm < tel.NEAR_RANGE_M
            right.append(CardSpec(
                "detail", "TARGET // PATIENT-01", style="bracket",
                big=(f"{p.score * 100:3.0f}%", YELLOW if p.score > 0.5 else ALERT),
                rows=[("RANGE EST", f"{dm:0.2f}M" if dm is not None else "--",
                       ALERT if near else YELLOW),
                      ("AZIMUTH", f"{(p.cx - 0.5) * 2:+.2f}", WHITE),
                      ("ELEV", f"{(0.5 - p.cy) * 2:+.2f}", WHITE)],
                bar=(p.score, YELLOW if p.score > 0.5 else ALERT)))
        if len(tel.targets) >= 2:                         # underline / open card
            s = tel.targets[1]
            right.append(CardSpec(
                "contact2", "CONTACT-02", style="underline",
                rows=[("CONF", f"{s.score * 100:3.0f}%", WHITE),
                      ("POS", f"{s.cx:+.2f},{s.cy:+.2f}", WHITE)],
                note="SECONDARY"))
        return left, right


__all__ = ["Telemetry", "Target", "WarningHud", "CardSpec", "CardStack",
           "load_font", "bgr", "YELLOW", "BLACK", "ALERT"]
