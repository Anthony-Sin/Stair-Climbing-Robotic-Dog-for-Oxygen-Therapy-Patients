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
"""
import math
import os

import cv2
import numpy as np
from typing import List, Optional, Sequence, Tuple

try:                                            # Pillow is the text backend
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:                               # pragma: no cover - degraded fallback
    _PIL_OK = False


# ───────────────────────────── Design tokens (RGB) ────────────────────────────
# Reference look ("ARCV / MUTEK" scanner): white-grey type on near-pure black with
# a SINGLE saturated red accent-core.  Red now carries all focus / active / warning
# meaning (connection lines, headers, the locked target); everything else is neutral
# off-white / grey / hairline.  Straight lines, open brackets, radius 0.
BG_BASE   = (7, 7, 8)        # #070708  near-pure black base
BG_PANEL  = (16, 15, 16)     # #100F10  raised panel fill
HAIRLINE  = (58, 56, 58)     # #3A383A  dividers / inactive borders (neutral grey)
TEXT      = (232, 232, 230)  # #E8E8E6  primary readout text (off-white)
DIM       = (146, 148, 150)  # #929496  secondary / labels / meta (neutral grey)
ACCENT    = (228, 54, 42)    # #E4362A  signal-red accent-core (active / focus / lock)
ALERT     = (255, 74, 58)    # #FF4A3A  hotter red — critical danger (blinks/fills)
TEXTURE   = (26, 18, 18)     # #1A1212  faint red-black dot-matrix fill
ACCENT_DK = (138, 42, 34)    # #8A2A22  dimmed red (connection lines, ticks, decode tail)
INK       = (0, 0, 0)        # text outline / shadow halo

# ── Back-compat aliases ────────────────────────────────────────────────────────
# Older call-sites + tests referenced the previous "Tac-Amber" names; map them onto
# the new two-colour system so nothing imports a missing symbol and the whole HUD
# reads in one accent.  AMBER/SUCCESS → cyan accent; SLATE/MUTED → neutral dim;
# WARNING/ERROR → alert red (the doc has no separate caution colour).
FG = TEXT
AMBER = ACCENT
SLATE = DIM
SUCCESS = ACCENT
BRIGHT_SLATE = TEXT
WARNING = ALERT
ERROR = ALERT
MUTED = DIM
SURFACE = HAIRLINE
DEEP_AMBER = ACCENT_DK
BG_FILL = BG_BASE
HUD_TEXT = HUD_FG = TEXT
HUD_AMBER = ACCENT
HUD_SLATE = DIM
HUD_MUTED = DIM
HUD_SUCCESS = ACCENT
HUD_WARNING = HUD_ALERT = HUD_ERROR = ALERT
HUD_SURFACE = HAIRLINE
HUD_INK = INK


def bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """RGB → OpenCV BGR."""
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def dim(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    """Scale a colour toward black by factor ``f`` (0..1)."""
    return (int(rgb[0] * f), int(rgb[1] * f), int(rgb[2] * f))


def hex4(value) -> str:
    """A 4-hex-digit code block cell (real value rendered as device-style hex)."""
    try:
        return f"{int(round(float(value))) & 0xFFFF:04X}"
    except Exception:
        return "----"


# ───────────────────────────────── Fonts ──────────────────────────────────────
_FONT_CACHE: dict = {}
_FONT_FILE_CACHE: dict = {}


def _matplotlib_ttf(name: str) -> Optional[str]:
    try:
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name)
    except Exception:
        return None


def _font_file(bold: bool, mono: bool) -> Optional[str]:
    """Resolve a font file.  ``mono`` → the data/monospace face (hex, decode,
    numbers); else the *display* face — a condensed techy sans (Bahnschrift →
    Agency FB → Segoe Semibold), falling back to DejaVu so the Jetson container
    still renders."""
    key = (bool(bold), bool(mono))
    if key in _FONT_FILE_CACHE:
        return _FONT_FILE_CACHE[key]
    cands: List[str] = []
    if mono:
        cands += [r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
                  r"C:\Windows\Fonts\CascadiaMono.ttf",
                  _matplotlib_ttf("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf" % ("-Bold" if bold else "")]
    else:
        cands += [r"C:\Windows\Fonts\bahnschrift.ttf",
                  r"C:\Windows\Fonts\AGENCYB.TTF" if bold else r"C:\Windows\Fonts\AGENCYR.TTF",
                  r"C:\Windows\Fonts\seguisb.ttf", r"C:\Windows\Fonts\tahoma.ttf",
                  _matplotlib_ttf("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")]
    found = next((c for c in cands if c and os.path.exists(c)), None)
    _FONT_FILE_CACHE[key] = found
    return found


def _font(size: int, bold: bool = False, mono: bool = False):
    key = (int(size), bool(bold), bool(mono))
    f = _FONT_CACHE.get(key)
    if f is not None:
        return f
    if _PIL_OK:
        path = _font_file(bold, mono)
        try:
            f = ImageFont.truetype(path, int(size)) if path else ImageFont.load_default()
        except Exception:
            f = ImageFont.load_default()
        if bold and not mono and path and path.lower().endswith("bahnschrift.ttf"):
            # Bahnschrift is a variable font — pick a heavier named instance.
            for name in ("Bold", "SemiBold"):
                try:
                    f.set_variation_by_name(name)
                    break
                except Exception:
                    pass
    else:
        f = None
    _FONT_CACHE[key] = f
    return f


def _text_width(text: str, size: int, bold: bool = True, mono: bool = False) -> int:
    """Pixel advance of ``text`` in the chosen face (exact via Pillow metrics,
    else an em-ratio estimate)."""
    if _PIL_OK:
        try:
            return int(round(_font(size, bold, mono).getlength(text)))
        except Exception:
            pass
    return int(round(len(text) * size * (0.6 if mono else 0.52)))


# ───────────────────────────── Deferred text layer ────────────────────────────
class TextLayer:
    """Collects text draws and renders them all in one Pillow pass per frame.

    Each ``add`` records a string + style; ``flush`` converts the BGR frame to a
    Pillow image once, draws every string with a true monospace font + a 1px dark
    outline (the only legibility aid the design permits — never a box), then writes
    the result back into the same ndarray in place.
    """

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def add(self, x: int, y: int, text: str, color=TEXT, *, size: int = 13,
            bold: bool = False, anchor: str = "lt", outline: bool = True,
            tracking: float = 0.0, mono: bool = False) -> None:
        if text is None or text == "":
            return
        self.items.append((int(x), int(y), str(text), color, int(size), bool(bold),
                           anchor, bool(outline), float(tracking), bool(mono)))

    def flush(self, frame_bgr: np.ndarray) -> None:
        if not self.items:
            return
        if not _PIL_OK:                          # OpenCV fallback (no true monospace)
            for x, y, text, color, size, bold, anchor, outline, _trk, _mono in self.items:
                self._cv_fallback(frame_bgr, x, y, text, color, size, anchor, outline)
            self.items = []
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for x, y, text, color, size, bold, anchor, outline, trk, mono in self.items:
            font = _font(size, bold, mono)
            kw = {"font": font, "fill": tuple(color), "anchor": anchor}
            if outline:
                kw["stroke_width"] = 1
                kw["stroke_fill"] = INK
            try:
                if trk:                           # mechanical letter-spacing (§2.2)
                    self._tracked(draw, x, y, text, font, tuple(color), anchor, outline, trk)
                else:
                    draw.text((x, y), text, **kw)
            except TypeError:                     # very old Pillow w/o anchor/stroke
                draw.text((x, y), text, font=font, fill=tuple(color))
        frame_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.items = []

    @staticmethod
    def _tracked(draw, x, y, text, font, color, anchor, outline, trk) -> None:
        # Per-glyph layout with extra advance; honours left/middle/right horizontal
        # anchor.  Vertical anchor falls back to the font's baseline handling.
        try:
            widths = [draw.textlength(ch, font=font) for ch in text]
        except Exception:
            widths = [font.size * 0.6 for _ in text]
        total = sum(widths) + trk * max(0, len(text) - 1)
        ha = anchor[0] if anchor else "l"
        cx = x - total if ha == "r" else (x - total / 2.0 if ha == "m" else x)
        va = anchor[1] if len(anchor) > 1 else "t"
        cur = cx
        for ch, wch in zip(text, widths):
            kw = {"font": font, "fill": color, "anchor": "l" + va}
            if outline:
                kw["stroke_width"] = 1
                kw["stroke_fill"] = INK
            draw.text((cur, y), ch, **kw)
            cur += wch + trk

    @staticmethod
    def _cv_fallback(frame_bgr, x, y, text, color, size, anchor, outline) -> None:
        scale = max(0.32, size / 30.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ax = x - tw if anchor and anchor[0] == "r" else (x - tw // 2 if anchor and anchor[0] == "m" else x)
        ay = y + th // 2 if anchor and len(anchor) > 1 and anchor[1] == "m" else y + th
        if outline:
            cv2.putText(frame_bgr, text, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(INK), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(color), 1, cv2.LINE_AA)


# ───────────────────────────── Compositing helpers ────────────────────────────
def fill_region(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                color_rgb: Tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a flat colour into a sub-rect."""
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame_bgr[y0:y1, x0:x1]
    layer = np.empty_like(sub)
    layer[:] = bgr(color_rgb)
    cv2.addWeighted(layer, alpha, sub, 1.0 - alpha, 0.0, sub)


def _bevel_pts(x0: int, y0: int, x1: int, y1: int, cut: int,
               corners: Tuple[str, ...]) -> List[Tuple[int, int]]:
    """Clockwise polygon for a rectangle with one or more 45° clipped corners
    (``'tl' 'tr' 'br' 'bl'``) — the angular cyberpunk panel shape, not a plain box."""
    p: List[Tuple[int, int]] = []
    p.append((x0 + cut, y0) if "tl" in corners else (x0, y0))
    if "tr" in corners:
        p += [(x1 - cut, y0), (x1, y0 + cut)]
    else:
        p.append((x1, y0))
    if "br" in corners:
        p += [(x1, y1 - cut), (x1 - cut, y1)]
    else:
        p.append((x1, y1))
    if "bl" in corners:
        p += [(x0 + cut, y1), (x0, y1 - cut)]
    else:
        p.append((x0, y1))
    if "tl" in corners:
        p.append((x0, y0 + cut))
    return p


def _fill_poly_alpha(frame_bgr: np.ndarray, pts: np.ndarray,
                     color_rgb: Tuple[int, int, int], alpha: float) -> None:
    """Alpha-fill an arbitrary polygon (used for beveled panel bodies); only the
    polygon's bounding sub-rect is touched, so it stays cheap."""
    h, w = frame_bgr.shape[:2]
    x0 = max(0, int(pts[:, 0].min()))
    y0 = max(0, int(pts[:, 1].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame_bgr[y0:y1, x0:x1]
    overlay = sub.copy()
    cv2.fillPoly(overlay, [pts - np.array([x0, y0])], bgr(color_rgb), cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0.0, sub)


def texture_fill(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, *,
                 alpha: float = 0.55, step: int = 6,
                 color_rgb: Tuple[int, int, int] = TEXTURE) -> None:
    """Dot-matrix texture inside a panel region (§2.4 'texture fill' — fills empty
    space so it reads as 'more data', never flat).  Cheap strided write."""
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < step or y1 - y0 < step:
        return
    sub = frame_bgr[y0:y1, x0:x1]
    dots = sub[1::step, 1::step].astype(np.float32)
    tint = np.array(bgr(color_rgb), dtype=np.float32)
    sub[1::step, 1::step] = (dots * (1.0 - alpha) + tint * alpha).astype(np.uint8)


def scanlines(frame_bgr: np.ndarray, *, alpha: float = 0.16, step: int = 3) -> None:
    """Faint CRT scan lines across the whole frame (fixed device chrome, §1)."""
    band = frame_bgr[::step]
    band[:] = (band.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def connection_line(frame_bgr: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int],
                    *, color=ACCENT_DK, alpha: float = 1.0) -> None:
    """A thin 1px line from a central node to a satellite element (§2.4)."""
    if alpha >= 1.0:
        cv2.line(frame_bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 bgr(color), 1, cv2.LINE_AA)
    else:
        cv2.line(frame_bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 bgr(dim(color, alpha)), 1, cv2.LINE_AA)


def panel(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, *,
          accent: Tuple[int, int, int] = ACCENT, texture: bool = True,
          alpha: float = 0.55, corners: Tuple[str, ...] = ("tr",),
          cut: int = 16) -> None:
    """Angular cyberpunk panel: a near-black body with one or more 45° clipped
    corners (NOT a plain rectangle) + a 1px hairline border, dot texture, accent
    bevel edges and open accent brackets on the square corners."""
    cut = max(8, min(cut, (x1 - x0) // 4, (y1 - y0) // 4))
    pts = np.array(_bevel_pts(x0, y0, x1, y1, cut, corners), np.int32)
    _fill_poly_alpha(frame_bgr, pts, BG_PANEL, alpha)
    if texture:
        texture_fill(frame_bgr, x0 + 2, y0 + 2, x1 - 2, y1 - 2)
    cv2.polylines(frame_bgr, [pts], True, bgr(HAIRLINE), 1, cv2.LINE_AA)
    c = bgr(accent)
    # accent stroke along each clipped (diagonal) corner
    diag = {"tl": ((x0 + cut, y0), (x0, y0 + cut)), "tr": ((x1 - cut, y0), (x1, y0 + cut)),
            "br": ((x1, y1 - cut), (x1 - cut, y1)), "bl": ((x0 + cut, y1), (x0, y1 - cut))}
    for cn in corners:
        a, b = diag[cn]
        cv2.line(frame_bgr, a, b, c, 2, cv2.LINE_AA)
    # open brackets on the remaining square corners
    leg = max(10, min(20, (x1 - x0) // 6))
    sq = {"tl": (x0, y0, 1, 1), "tr": (x1, y0, -1, 1), "br": (x1, y1, -1, -1), "bl": (x0, y1, 1, -1)}
    for cn in ("tl", "br"):
        if cn not in corners:
            bx, by, sx, sy = sq[cn]
            cv2.line(frame_bgr, (bx, by), (bx + sx * leg, by), c, 2, cv2.LINE_AA)
            cv2.line(frame_bgr, (bx, by), (bx, by + sy * leg), c, 2, cv2.LINE_AA)


def group_backing(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                  *, alpha: float = 0.55, border: bool = True,
                  accent: Optional[Tuple[int, int, int]] = ACCENT) -> None:
    """Back-compat shim → :func:`panel` (older call-sites/tests)."""
    panel(frame_bgr, x0, y0, x1, y1, accent=accent or ACCENT, texture=True, alpha=alpha)
    _ = border


# ───────────────────────────── Group title / tick ─────────────────────────────
def group_title(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, title: str,
                *, right: bool = False, size: int = 13, color=ACCENT) -> None:
    """Accent group header with a single open corner tick (top-left, or mirrored
    top-right).  Tracked uppercase, no box."""
    leg = 9
    c = bgr(color)
    if right:
        cv2.line(frame_bgr, (x, y), (x - leg, y), c, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (x, y), (x, y + leg), c, 2, cv2.LINE_AA)
        layer.add(x - leg - 6, y - 3, title.upper(), color, size=size, bold=True,
                  anchor="rt", tracking=1.0)
    else:
        cv2.line(frame_bgr, (x, y), (x + leg, y), c, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (x, y), (x, y + leg), c, 2, cv2.LINE_AA)
        layer.add(x + leg + 6, y - 3, title.upper(), color, size=size, bold=True,
                  anchor="lt", tracking=1.0)


def kv_row(layer: TextLayer, x_label: int, x_value: int, y: int, label: str,
           value: str, value_color=TEXT, *, label_color=DIM, size: int = 12,
           value_bold: bool = False, right: bool = False) -> None:
    """One label/value row: label (dim) left, value (text/accent) at the value column."""
    layer.add(x_label, y, label.upper(), label_color, size=size, anchor="lt")
    layer.add(x_value, y, value, value_color, size=size, bold=value_bold,
              anchor="rt" if right else "lt")


# ─────────────────────────────── Data tile (§2.4) ─────────────────────────────
def data_tile(frame_bgr: np.ndarray, layer: TextLayer, x0: int, y0: int, x1: int, y1: int,
              title: str, code_lines: Sequence[str],
              rows: Sequence[Tuple[str, str, Tuple[int, int, int]]], *,
              right: bool = False, accent: Tuple[int, int, int] = ACCENT,
              size: int = 12, progress: float = 1.0, seed: int = 0) -> None:
    """Bracket-cornered data tile: a stacked hex code-block on the left, an accent
    title + label/value rows on the right (§2.4 'data tile').  ``right`` mirrors it
    (code block on the right) for the right-hand status rail.  ``progress`` < 1
    drives the per-value decode-in animation and a load sweep.
    """
    panel(frame_bgr, x0, y0, x1, y1, accent=accent, corners=("tl",) if right else ("tr",))
    code_w = max(_text_width(c, size - 1) for c in code_lines) if code_lines else 0
    code_w = max(code_w, _text_width("0000", size - 1))
    pad = 8

    if not right:
        code_x = x0 + pad
        div_x = code_x + code_w + pad
        title_x, val_edge = div_x + pad, x1 - pad
        title_anchor_x, title_right = title_x, False
    else:
        code_x = x1 - pad - code_w
        div_x = code_x - pad
        title_x, val_edge = x0 + pad, div_x - pad
        title_anchor_x, title_right = x1 - pad, True

    # title spans the top; code block + rows start beneath it (no collision)
    group_title(frame_bgr, layer, title_anchor_x, y0 + 8, title, right=title_right, size=size + 1)
    body_y = y0 + 30
    cv2.line(frame_bgr, (div_x, body_y - 4), (div_x, y1 - 8), bgr(HAIRLINE), 1, cv2.LINE_AA)

    # hex code block (cyan, monospace, stacked)
    for i, code in enumerate(code_lines):
        layer.add(code_x, body_y + i * (size + 4), code, accent, size=size - 1,
                  bold=True, anchor="lt", tracking=1.0, mono=True)

    # rows (labels static, values decode-in)
    ry = body_y
    for lbl, val, col in rows:
        layer.add(title_x, ry, lbl, DIM, size=size, anchor="lt")
        decode_text(layer, val_edge, ry, val, col, progress=progress, size=size,
                    anchor="rt", seed=seed)
        ry += size + 6

    if progress < 1.0:
        sweep_line(frame_bgr, x0 + 1, body_y - 2, x1 - 1, y1 - 4, progress, color=accent)


# ────────────────────────────── Status pill ───────────────────────────────────
def status_pill(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, text: str,
                state: str = "scanning", *, size: int = 12) -> int:
    """Sharp-cornered (radius 0) lock/mode chip.

      ``locked``   → cyan fill,   filled ▣, ink text
      ``scanning`` → hairline fill, hollow ◌ ring (dim), text
      ``lost``     → red fill,     ✗ mark,   white bold text
    Returns the chip's right edge.
    """
    text = text.upper()
    pad = max(7, size // 2)
    gap = 6
    mark_box = size + 2
    char_w = _text_width("M", size, True)
    text_w = len(text) * char_w
    x1 = x + pad + mark_box + gap + text_w + pad
    h = size + 10
    y1 = y + h
    cy = (y + y1) // 2

    fills = {"locked": ACCENT, "lost": ALERT, "scanning": HAIRLINE}
    fill = fills.get(state, HAIRLINE)
    txt_col = INK if state == "locked" else (FG if state == "lost" else TEXT)
    mark_col = txt_col if state != "scanning" else DIM
    fill_region(frame_bgr, x, y, x1, y1, fill, 0.92 if state != "scanning" else 0.7)
    cv2.rectangle(frame_bgr, (x, y), (x1, y1), bgr(ACCENT if state == "scanning" else fill), 1, cv2.LINE_AA)

    mcx, ms = x + pad + mark_box // 2, max(4, size // 3)
    if state == "locked":
        cv2.rectangle(frame_bgr, (mcx - ms, cy - ms), (mcx + ms, cy + ms), bgr(mark_col), -1, cv2.LINE_AA)
    elif state == "lost":
        cv2.line(frame_bgr, (mcx - ms, cy - ms), (mcx + ms, cy + ms), bgr(mark_col), 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (mcx - ms, cy + ms), (mcx + ms, cy - ms), bgr(mark_col), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame_bgr, (mcx, cy), ms, bgr(mark_col), 1, cv2.LINE_AA)

    layer.add(x + pad + mark_box + gap, cy, text, txt_col, size=size, bold=True,
              anchor="lm", outline=False)
    return x1


# ───────────────────────────── Focal data tile ────────────────────────────────
def focal_tag(frame_bgr: np.ndarray, layer: TextLayer, cx: int, top: int,
              segments: List[Tuple[str, Tuple[int, int, int]]], *,
              accent: Tuple[int, int, int] = ACCENT, size: int = 14,
              pad_x: int = 14, pad_y: int = 7, sep: str = " / ",
              sep_color: Tuple[int, int, int] = DIM,
              progress: float = 1.0, seed: int = 0) -> int:
    """A centred bracket-cornered focal tile (§2.4 'bracket frame' / 'data tile').

    Renders ``LABEL / VALUE`` (each ``(text, colour)`` segment laid out in true
    monospace) on a near-black scrim with a 1px hairline border + open ``accent``
    corner brackets — the at-a-glance designator for a tracked subject.  Centred on
    ``cx`` with its top edge at ``top``; returns the tile's bottom ``y`` so callers
    can stack several down the middle of the frame.
    """
    laid: List[Tuple[str, Tuple[int, int, int], int]] = []
    for i, (text, col) in enumerate(segments):
        if i:
            laid.append((sep, sep_color, _text_width(sep, size, mono=True)))
        laid.append((text, col, _text_width(text, size, mono=True)))
    text_w = sum(w for _, _, w in laid)

    box_w = text_w + 2 * pad_x
    box_h = size + 2 * pad_y
    x0 = int(round(cx - box_w / 2.0))
    x1 = x0 + box_w
    y0 = int(top)
    y1 = y0 + box_h

    # Angled (non-square) tile: bevel the top-right + bottom-left corners.
    cut = max(7, box_h // 3)
    pts = np.array(_bevel_pts(x0, y0, x1, y1, cut, ("tr", "bl")), np.int32)
    _fill_poly_alpha(frame_bgr, pts, BG_BASE, 0.74)
    texture_fill(frame_bgr, x0 + 2, y0 + 2, x1 - 2, y1 - 2, alpha=0.4)
    cv2.polylines(frame_bgr, [pts], True, bgr(HAIRLINE), 1, cv2.LINE_AA)
    leg = max(9, box_h // 3)
    ac = bgr(accent)
    for bx, by, sx, sy in ((x0, y0, 1, 1), (x1, y1, -1, -1)):    # brackets on square corners
        cv2.line(frame_bgr, (bx, by), (bx + sx * leg, by), ac, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (bx, by), (bx, by + sy * leg), ac, 2, cv2.LINE_AA)
    cv2.line(frame_bgr, (x1 - cut, y0), (x1, y0 + cut), ac, 2, cv2.LINE_AA)   # accent bevels
    cv2.line(frame_bgr, (x0, y1 - cut), (x0 + cut, y1), ac, 2, cv2.LINE_AA)

    if progress < 1.0:
        sweep_line(frame_bgr, x0 + 1, y0 + 1, x1 - 1, y1 - 1, progress, color=accent)
    cyc = (y0 + y1) // 2
    cursor = x0 + pad_x
    for text, col, w in laid:
        decode_text(layer, cursor, cyc, text, col, progress=progress, size=size,
                    bold=True, anchor="lm", seed=seed)
        cursor += w
    return y1


# ─────────────────── Cyberpunk decode / load-in display kit ────────────────────
_SCRAMBLE = "ABCDEF0123456789#%&*<>/\\=+|:"


def decode_text(layer: TextLayer, x: int, y: int, text, color=TEXT, *,
                progress: float = 1.0, size: int = 13, bold: bool = False,
                anchor: str = "lt", seed: int = 0,
                scramble_color: Tuple[int, int, int] = ACCENT_DK) -> None:
    """Cyberpunk 'decrypting' reveal: a growing real prefix + a per-frame scrambled
    tail.  Monospace keeps the width stable so nothing reflows; ``progress`` >= 1
    renders the plain string (cheap)."""
    text = str(text)
    n = len(text)
    if n == 0:
        return
    if progress >= 1.0:
        layer.add(x, y, text, color, size=size, bold=bold, anchor=anchor, mono=True)
        return
    cw = _text_width("M", size, bold, mono=True)
    va = anchor[1] if len(anchor) > 1 else "t"
    if anchor and anchor[0] == "r":
        x0 = x - n * cw
    elif anchor and anchor[0] == "m":
        x0 = x - (n * cw) // 2
    else:
        x0 = x
    k = int(max(0.0, min(1.0, progress)) * n)
    revealed, rem = text[:k], text[k:]
    scr = "".join(" " if c == " " else _SCRAMBLE[(seed * 13 + (k + i) * 7 + ord(c)) % len(_SCRAMBLE)]
                  for i, c in enumerate(rem))
    if revealed:
        layer.add(x0, y, revealed, color, size=size, bold=bold, anchor="l" + va, mono=True)
    layer.add(x0 + len(revealed) * cw, y, scr, scramble_color, size=size, bold=bold,
              anchor="l" + va, mono=True)


def label_stack(layer: TextLayer, x: int, y: int, caption, value, *,
                value_color=TEXT, caption_color=DIM, caption_size: int = 10,
                value_size: int = 16, bold: bool = True, right: bool = False,
                progress: float = 1.0, seed: int = 0) -> int:
    """Cyberpunk caption-over-value: a tiny tracked dim caption with a larger value
    beneath (the 'INPUT DATA → ENEMY #1' motif).  Returns the y below the value."""
    anc = "r" if right else "l"
    layer.add(x, y, str(caption).upper(), caption_color, size=caption_size,
              anchor=anc + "t", tracking=1.5)
    vy = y + caption_size + 5
    decode_text(layer, x, vy, value, value_color, progress=progress, size=value_size,
                bold=bold, anchor=anc + "t", seed=seed)
    return vy + value_size + 7


def section_label(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
                  text, *, color=DIM) -> None:
    """Tiny dim section header with a hairline rule trailing to the right
    (the 'DMG TYPE / RESISTANCES' divider motif)."""
    t = str(text).upper()
    layer.add(x, y, t, color, size=10, anchor="lt", tracking=1.5)
    tw = _text_width(t, 10) + 14
    if x + tw < x + w:
        cv2.line(frame_bgr, (x + tw, y + 5), (x + w, y + 5), bgr(HAIRLINE), 1, cv2.LINE_AA)


def status_bar(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
               text, state: str = "tracking", *, progress: float = 1.0,
               size: int = 12) -> int:
    """Threat-style bar: a filled ``!`` marker square + a label bar whose fill grows
    to ``progress``.  locked→accent, lost→alert, else accent.  Returns bottom y."""
    h = size + 12
    col = {"locked": ACCENT, "lost": ALERT}.get(state, ACCENT)
    mb = h
    fill_region(frame_bgr, x, y, x + mb, y + h, col, 0.92)
    layer.add(x + mb // 2, y + h // 2, "!", INK, size=size, bold=True, anchor="mm", outline=False)
    bx0, bx1 = x + mb + 3, x + w
    fill_region(frame_bgr, bx0, y, bx1, y + h, BG_PANEL, 0.7)
    fw = int((bx1 - bx0) * max(0.0, min(1.0, progress)))
    if fw > 0:
        fill_region(frame_bgr, bx0, y, bx0 + fw, y + h, col, 0.22)
    cv2.rectangle(frame_bgr, (bx0, y), (bx1, y + h), bgr(col), 1, cv2.LINE_AA)
    layer.add(bx0 + 8, y + h // 2, str(text).upper(), TEXT, size=size, bold=True, anchor="lm")
    return y + h


def button_glyph(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, key, label,
                 *, size: int = 12, accent: Tuple[int, int, int] = ACCENT) -> int:
    """Circled key glyph + label (a command-list item).  Returns the right-edge x."""
    r = size // 2 + 3
    cyc = y + r
    cv2.circle(frame_bgr, (x + r, cyc), r, bgr(accent), 1, cv2.LINE_AA)
    layer.add(x + r, cyc, str(key)[:1].upper(), accent, size=size - 2, bold=True,
              anchor="mm", outline=False)
    lx = x + 2 * r + 7
    layer.add(lx, cyc, str(label), TEXT, size=size, anchor="lm")
    return lx + _text_width(str(label), size)


def status_dots(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
                items: Sequence[Tuple[str, bool]], *, size: int = 11) -> None:
    """A clean labelled status-pip row (filled cyan dot = ok, hollow dim = off) —
    replaces the old warning-triangle triad.  Evenly distributes ``items`` across
    ``w``."""
    n = max(1, len(items))
    step = w // n
    for i, (lbl, ok) in enumerate(items):
        cxp = x + i * step + 5
        cyp = y + size // 2
        if ok:
            cv2.circle(frame_bgr, (cxp, cyp), max(3, size // 3), bgr(ACCENT), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame_bgr, (cxp, cyp), max(3, size // 3), bgr(DIM), 1, cv2.LINE_AA)
        layer.add(cxp + size, cyp, str(lbl).upper(), TEXT if ok else DIM, size=size - 1,
                  anchor="lm")


def sweep_line(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               progress: float, *, color: Tuple[int, int, int] = ACCENT) -> None:
    """A bright scan line sweeping top→bottom across a panel while it loads
    (``progress`` 0..1); brighter early, fades out as it settles."""
    if progress <= 0.0 or progress >= 1.0:
        return
    yy = int(y0 + (y1 - y0) * progress)
    fill_region(frame_bgr, x0, max(y0, yy - 1), x1, min(y1, yy + 1), color,
                min(0.6, 0.30 + 0.4 * (1.0 - progress)))


def scan_frame(frame_bgr: np.ndarray, cx: int, cy: int, w: int, h: int, *,
               progress: float = 1.0, state: str = "scanning") -> None:
    """Large animated corner brackets around the scanned subject (the big Cyberpunk
    scan reticle) — the corners ease out from the centre as the subject locks."""
    col = ALERT if state == "danger" else ACCENT
    c = bgr(col)
    p = max(0.06, min(1.0, progress))
    hw, hh = int(w / 2), int(h / 2)
    leg = max(10, int(min(hw, hh) * 0.42))
    ox, oy = int(hw * p), int(hh * p)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        bx, by = cx + sx * ox, cy + sy * oy
        cv2.line(frame_bgr, (bx, by), (bx - sx * leg, by), c, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (bx, by), (bx, by - sy * leg), c, 2, cv2.LINE_AA)
    if progress >= 0.98:                            # settled side ticks
        for sx in (-1, 1):
            mx = cx + sx * ox
            cv2.line(frame_bgr, (mx, cy - oy // 4), (mx, cy + oy // 4), c, 1, cv2.LINE_AA)


# ───────────────────────────── Sector / signal bar ────────────────────────────
def sector_bar(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
               label: str, frac: float, value_str: str, *, danger: bool = False,
               highlight: bool = False, no_return: bool = False,
               label_size: int = 11) -> None:
    """Horizontal segmented signal bar (no container).  ``frac`` 0..1 fills accent
    (or red when ``danger``); empty cells render as faint hairline ticks."""
    label_w = 34
    val_w = 52
    bar_x0 = x + label_w
    bar_w = max(20, w - label_w - val_w)
    cells = 18
    gap = 2
    cw = (bar_w - (cells - 1) * gap) / cells
    cy = y
    fill_col = ALERT if danger else ACCENT
    n_fill = int(round(max(0.0, min(1.0, frac)) * cells))
    for i in range(cells):
        cx0 = int(bar_x0 + i * (cw + gap))
        cx1 = int(cx0 + cw)
        if no_return:
            cv2.rectangle(frame_bgr, (cx0, cy - 3), (cx1, cy + 3), bgr(dim(HAIRLINE, 0.6)), -1)
        elif i < n_fill:
            cv2.rectangle(frame_bgr, (cx0, cy - 3), (cx1, cy + 3), bgr(fill_col), -1)
        else:
            cv2.rectangle(frame_bgr, (cx0, cy - 4), (cx1, cy + 4), bgr(dim(HAIRLINE, 0.6)), 1)
    lbl_col = ACCENT if highlight else DIM
    layer.add(x, y - 6, label.upper(), lbl_col, size=label_size, bold=highlight, anchor="lt")
    val_col = ALERT if danger else (DIM if no_return else TEXT)
    layer.add(x + w, y - 6, value_str, val_col, size=label_size, anchor="rt")


# ───────────────────────────── Radar signal graph ─────────────────────────────
def radar_graph(frame_bgr: np.ndarray, layer: TextLayer, cx: int, cy: int, radius: int,
                sector_angles: Sequence[float], fracs: Sequence[Optional[float]], *,
                danger_flags: Optional[Sequence[bool]] = None,
                marker_rad: Optional[float] = None, marker_alert: bool = False) -> None:
    """Forward-180° polygon/radar signal graph (§2.4 'signal graph').

    ``sector_angles`` are bearings in radians (0 = forward/up, -=left, +=right);
    ``fracs`` are 0..1 obstacle ranges (None = no return → outer rim).  Draws range
    rings + spokes, a filled accent return polygon (red where ``danger_flags``), and
    an optional bearing marker on the rim.
    """
    danger_flags = list(danger_flags or [False] * len(sector_angles))

    def pt(ang: float, r: float) -> Tuple[int, int]:
        return (int(round(cx + r * math.sin(ang))), int(round(cy - r * math.cos(ang))))

    # range rings (upper half) + outer arc
    for f in (0.34, 0.67, 1.0):
        cv2.ellipse(frame_bgr, (cx, cy), (int(radius * f), int(radius * f)), 0, 180, 360,
                    bgr(HAIRLINE), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (cx - radius, cy), (cx + radius, cy), bgr(HAIRLINE), 1, cv2.LINE_AA)
    # spokes
    for ang in (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2):
        cv2.line(frame_bgr, (cx, cy), pt(ang, radius), bgr(dim(HAIRLINE, 0.8)), 1, cv2.LINE_AA)

    # return polygon
    poly = [(cx, cy)]
    danger_pts = []
    for ang, fr, dg in zip(sector_angles, fracs, danger_flags):
        rr = radius * (fr if fr is not None else 1.0)
        p = pt(ang, rr)
        poly.append(p)
        if dg and fr is not None:
            danger_pts.append(p)
    if len(poly) >= 3:
        overlay = frame_bgr.copy()
        cv2.fillPoly(overlay, [np.array(poly, np.int32)], bgr(dim(ACCENT, 0.5)))
        cv2.addWeighted(overlay, 0.30, frame_bgr, 0.70, 0.0, frame_bgr)
        cv2.polylines(frame_bgr, [np.array(poly[1:], np.int32)], False, bgr(ACCENT), 1, cv2.LINE_AA)
    for p in danger_pts:
        cv2.circle(frame_bgr, p, 3, bgr(ALERT), -1, cv2.LINE_AA)

    # forward apex tick + bearing marker
    cv2.circle(frame_bgr, (cx, cy), 2, bgr(ACCENT), -1, cv2.LINE_AA)
    if marker_rad is not None:
        mp = pt(float(marker_rad), radius)
        mc = bgr(ALERT if marker_alert else ACCENT)
        cv2.circle(frame_bgr, mp, 4, mc, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, mp, 4, bgr(INK), 1, cv2.LINE_AA)
    _ = layer


# ───────────────────────────── Leg phase slider ───────────────────────────────
def leg_row(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
            leg: str, frac: float, swing: bool, state_word: str,
            *, active: bool = True, size: int = 12) -> None:
    """Dot-and-track phase slider for one leg.  Accent dot for the active/swing leg."""
    track_x0 = x + 46
    track_x1 = x + w - 56
    ty = y
    track_col = DIM if active else dim(DIM, 0.6)
    cv2.line(frame_bgr, (track_x0, ty), (track_x1, ty), bgr(track_col), 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, (track_x1, ty), 4, bgr(track_col), 1, cv2.LINE_AA)
    cv2.circle(frame_bgr, (track_x0, ty), 4, bgr(track_col), 1, cv2.LINE_AA)
    f = max(0.0, min(1.0, frac))
    px = int(track_x0 + f * (track_x1 - track_x0))
    dot_col = ACCENT if swing else (TEXT if active else DIM)
    cv2.circle(frame_bgr, (px, ty), 5, bgr(dot_col), -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, (px, ty), 5, bgr(INK), 1, cv2.LINE_AA)
    layer.add(x, y - 6, leg.upper(), TEXT, size=size, anchor="lt")
    state_col = ACCENT if swing else (DIM if active else dim(DIM, 0.7))
    layer.add(x + w, y - 6, state_word.upper(), state_col, size=size, anchor="rt")


# ──────────────────────────────── Reticle ─────────────────────────────────────
def reticle(frame_bgr: np.ndarray, cx: int, cy: int, state: str = "scanning",
            *, r: int = 18) -> None:
    """Centre-gap crosshair + ring with a north tick.  Cyan while scanning/locked,
    red inside a danger/disagreement zone (the only non-accent state)."""
    col = {"danger": ALERT}.get(state, ACCENT)
    c = bgr(col)
    cv2.ellipse(frame_bgr, (cx, cy), (r, r), 0, -60, 240, c, 2, cv2.LINE_AA)
    cv2.line(frame_bgr, (cx, cy - r - 6), (cx, cy - r - 1), c, 2, cv2.LINE_AA)   # north tick
    if state == "locked":                         # solid corner brackets when locked
        bl = r - 4
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            bx, by = cx + sx * r, cy + sy * r
            cv2.line(frame_bgr, (bx, by), (bx - sx * bl, by), c, 1, cv2.LINE_AA)
            cv2.line(frame_bgr, (bx, by), (bx, by - sy * bl), c, 1, cv2.LINE_AA)
    gap, arm = 6, 11
    cv2.line(frame_bgr, (cx - gap - arm, cy), (cx - gap, cy), c, 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (cx + gap, cy), (cx + gap + arm, cy), c, 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (cx, cy - gap - arm), (cx, cy - gap), c, 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (cx, cy + gap), (cx, cy + gap + arm), c, 1, cv2.LINE_AA)


def lead_label(frame_bgr: np.ndarray, layer: TextLayer, ax: int, ay: int,
               lines: List[Tuple[str, Tuple[int, int, int]]], *, up: bool = True,
               size: int = 11) -> None:
    """Anchor dot in-frame + thin diagonal leadline to a small label stack."""
    dx = 26
    dy = -22 if up else 22
    lx, ly = ax + dx, ay + dy
    cv2.circle(frame_bgr, (ax, ay), 3, bgr(ACCENT), -1, cv2.LINE_AA)
    cv2.line(frame_bgr, (ax, ay), (lx, ly), bgr(ACCENT_DK), 1, cv2.LINE_AA)
    ty = ly - (len(lines) * (size + 3)) if up else ly
    for i, (txt, col) in enumerate(lines):
        layer.add(lx + 4, ty + i * (size + 3), txt, col, size=size, anchor="lt")


# ───────────────────────── Screen frame chrome (§2.3) ─────────────────────────
def corner_frame(frame_bgr: np.ndarray, w: int, h: int, *, m: int = 10, leg: int = 16) -> None:
    """Fixed device chrome: a thin 1px hairline outer border + open accent L-ticks
    at the four corners (leaves the centre clear)."""
    cv2.rectangle(frame_bgr, (m - 3, m - 3), (w - m + 3, h - m + 3), bgr(dim(HAIRLINE, 0.7)), 1, cv2.LINE_AA)
    c = bgr(ACCENT)
    for cx, cy, sx, sy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(frame_bgr, (cx, cy), (cx + sx * leg, cy), c, 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (cx, cy), (cx, cy + sy * leg), c, 1, cv2.LINE_AA)
