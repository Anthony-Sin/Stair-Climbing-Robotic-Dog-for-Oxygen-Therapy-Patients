"""WARNING HUD — the stateful full-screen compositor.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor).  The
:class:`WarningHud` compositor: it conditions the live feed, draws the header /
footer chrome, on-feed target reticle, the persistent DEPTH + LiDAR instrument
panels, the dynamic :class:`CardStack` columns, and transient toasts / lost
banner — one frame per :meth:`WarningHud.render` call.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.hud.warning_cards import CardSpec, CardStack
from core.hud.warning_model import Target, Telemetry
from core.hud.warning_render import Layer, blit, cut_poly, hazard_band
from core.hud.warning_text import text_width
from core.hud.warning_theme import (
    ALERT,
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
)


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
