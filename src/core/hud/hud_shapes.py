"""Vector compositing + geometry primitives for the raster HUD.

Alpha-blend fills, the bevelled-corner polygon math, dot-matrix/scanline
textures, and the low-level angular chrome (panels, brackets, sweep line, scan
reticle, radar, sector/leg bars, lead labels, corner frame).  All drawn with
OpenCV's AA primitives on the near-black bed; higher-level text widgets in
:mod:`core.hud.hud_widgets` build on these.
"""
import math

import cv2
import numpy as np
from typing import List, Optional, Sequence, Tuple

from core.hud.hud_theme import (
    BG_BASE, BG_PANEL, HAIRLINE, TEXT, DIM, ACCENT, ALERT, TEXTURE, ACCENT_DK, INK,
    bgr, dim,
)
from core.hud.hud_text import TextLayer, _text_width


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
