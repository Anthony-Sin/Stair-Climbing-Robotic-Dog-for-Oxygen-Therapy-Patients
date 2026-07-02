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
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from arcv.overlay.anim import Sequencer, linear, out_cubic

from core.vision.lidar_fusion import decode_lidar_profile

Color = Tuple[float, float, float, float]

# ------------------------------------------------------------------ palette
WHITE = (0.94, 0.95, 0.94, 1.0)      # primary readout
GREY = (0.60, 0.62, 0.64, 1.0)       # secondary / values-neutral
GREY_DIM = (0.40, 0.42, 0.44, 1.0)   # labels / captions
FRAME = (0.32, 0.34, 0.37, 1.0)      # panel frame linework (dim steel)
RED = (0.89, 0.21, 0.16, 1.0)        # signal-red accent-core (#E4362A)
RED_HOT = (1.00, 0.40, 0.26, 1.0)    # hotter alert red (danger / fall)
RED_DIM = (0.52, 0.16, 0.13, 1.0)    # dimmed accent (ticks, leaders)
_TAU = math.pi * 2.0
_CUT = 7.0                           # panel corner chamfer (px, pre-scale)


def _fa(c: Color, a: float) -> Color:
    return (c[0], c[1], c[2], c[3] * a)


# ------------------------------------------------------------------- state
@dataclass
class HudState:
    W: int
    H: int
    node: Tuple[float, float, float, float]
    present: bool
    locked: bool
    reacquire: bool
    fell: bool
    fall_type: str
    dist: Optional[float]
    bearing: Optional[float]
    stairs_det: bool
    stairs_conf: float
    stairs_depth: Optional[float]
    cam_ok: bool
    lidar_ok: bool
    imu_ok: bool
    roll: Optional[float]
    pitch: Optional[float]
    height: Optional[float]
    fusion_src: str
    loco_mode: str
    loco_gait: str
    loco_clear: Optional[float]
    loco_speed: Optional[float]
    loco_policy: str
    swing_legs: List[str]
    cmd_v: float
    cmd_w: float
    lidar_dec: Optional[Dict[str, Any]]
    person_bearing_rad: Optional[float]
    disagreement: bool
    flash_kind: str
    flash_amt: float
    conn_id: str
    rec_s: int
    frame_no: int
    clock_str: str
    camera_mode: str
    proc_fps: float = 0.0
    view_fps: float = 0.0
    depth_rect: Optional[Tuple[int, int, int, int]] = None

    @property
    def accent(self) -> Color:
        return RED_HOT if (self.fell or (not self.present and self.reacquire)) else RED

    @property
    def track_word(self) -> str:
        if self.fell:
            return "FALLEN"
        if self.locked:
            return "LOCKED"
        if self.present:
            return "TRACKING"
        if self.reacquire:
            return "SEARCHING"
        return "NO SIGNAL"


def derive(debug_info: Dict[str, Any], frame_meta: Optional[dict], node,
           W: int, H: int, reacquire: bool, camera_mode: str,
           trans_x_cmd: float, rotation_cmd: float, flash_kind: str, flash_amt: float,
           rec_s: int, frame_no: int, clock_str: str, conn_id: str,
           depth_rect: Optional[Tuple[int, int, int, int]] = None,
           proc_fps: float = 0.0, view_fps: float = 0.0) -> HudState:
    debug_info = debug_info or {}
    stair_demo = debug_info.get("stair_demo", {}) or {}
    robot = stair_demo.get("robot", {}) if isinstance(stair_demo, dict) else {}
    robot = robot if isinstance(robot, dict) else {}
    loco = stair_demo.get("locomotion", {}) if isinstance(stair_demo, dict) else {}
    loco = loco if isinstance(loco, dict) else {}

    cx_int = debug_info.get("center_x")
    if cx_int is None:
        cx_int = debug_info.get("bbox_center_x")
    present = cx_int is not None
    locked = bool(debug_info.get("matched_visual_lock", False))

    lidar_dec = decode_lidar_profile(debug_info.get("lidar_profile"))
    lidar_ok = lidar_dec is not None and int(lidar_dec.get("ray_count", 0)) > 0
    lidar_bearing_deg = debug_info.get("lidar_bearing_deg")

    swing: List[str] = []
    if frame_meta:
        swing = [str(x).upper() for x in frame_meta.get("swing_legs", [])]

    return HudState(
        W=W, H=H, node=node, present=present, locked=locked, reacquire=reacquire,
        fell=bool(robot.get("fell", False)), fall_type=str(robot.get("fall_type", "unknown")),
        dist=debug_info.get("depth_distance_m"), bearing=debug_info.get("rotation_error_deg"),
        stairs_det=bool(debug_info.get("stairs_detected", False)),
        stairs_conf=float(debug_info.get("stairs_conf") or 0.0),
        stairs_depth=debug_info.get("stairs_depth_m"),
        cam_ok=frame_meta is None or bool(frame_meta.get("success", True)),
        lidar_ok=lidar_ok, imu_ok=bool(robot),
        roll=robot.get("roll_deg"), pitch=robot.get("pitch_deg"), height=robot.get("height_m"),
        fusion_src=str(debug_info.get("distance_source", "N/A")).replace("_", " ").upper(),
        loco_mode=str(loco.get("mode", "N/A")).upper(),
        loco_gait=str(loco.get("gait_pattern", "N/A")).upper().replace("_", " "),
        loco_clear=loco.get("foot_clearance_m"), loco_speed=loco.get("commanded_speed_mps"),
        loco_policy=str(loco.get("policy", "N/A")),
        swing_legs=swing, cmd_v=float(trans_x_cmd or 0.0), cmd_w=float(rotation_cmd or 0.0),
        lidar_dec=lidar_dec,
        person_bearing_rad=math.radians(float(lidar_bearing_deg)) if lidar_bearing_deg is not None else None,
        disagreement=bool(debug_info.get("distance_disagreement", False)),
        flash_kind=flash_kind, flash_amt=flash_amt,
        conn_id=conn_id, rec_s=rec_s, frame_no=frame_no, clock_str=clock_str,
        camera_mode=str(camera_mode).upper(), depth_rect=depth_rect,
        proc_fps=float(proc_fps or 0.0), view_fps=float(view_fps or 0.0),
    )


# -------------------------------------------------------------- small utils
def _fmt_m(v, p=2) -> str:
    try:
        return "N/A" if v is None else f"{float(v):.{p}f}M"
    except Exception:
        return "N/A"


def _fill(ov, x0, y0, x1, y1, color: Color) -> None:
    ov.vector.rounded_rect_fill(x0, y0, x1, y1, 0.0, color)


# --------------------------------------------------------- tactical widgets
def _panel(ov, x0, y0, x1, y1, *, s, reveal=1.0, accent: Color = RED,
           title: Optional[str] = None, t: float = 0.0, badge: Optional[Tuple[str, Color]] = None):
    """Chamfered bracket panel — the signature frame element.

    Frame is a closed polyline with the top-right and bottom-left corners cut;
    the two square corners (TL, BR) get accent L-ticks.  Optional header: an
    accent notch + title + hairline underline (+ an optional right-aligned
    badge).  Returns the content-top y."""
    c = _CUT * s
    pts = [(x0, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1),
           (x0 + c, y1), (x0, y1 - c), (x0, y0)]
    ov.vector.polyline(pts, _fa(FRAME, 0.95), 1.3 * s, reveal=reveal)
    tick = 13 * s * min(1.0, reveal * 1.4)
    if tick > 1.0:
        ov.vector.line((x0, y0), (x0 + tick, y0), accent, 2.0 * s)
        ov.vector.line((x0, y0), (x0, y0 + tick), accent, 2.0 * s)
        ov.vector.line((x1, y1), (x1 - tick, y1), accent, 2.0 * s)
        ov.vector.line((x1, y1), (x1, y1 - tick), accent, 2.0 * s)
    cy = y0 + 9 * s
    if title and reveal > 0.5:
        hh = 10.5 * s
        _fill(ov, x0 + 6 * s, y0 + 7 * s, x0 + 9.5 * s, y0 + 7 * s + hh, accent)
        ov.text.text(title, x0 + 14 * s, y0 + 6 * s, hh, accent, align="left")
        if badge is not None:
            ov.text.text(badge[0], x1 - 8 * s, y0 + 7 * s, 9.0 * s, badge[1], align="right")
        uy = y0 + 7 * s + hh + 6 * s
        ov.vector.line((x0 + 6 * s, uy), (x1 - 6 * s, uy), _fa(FRAME, 0.8), 1.0 * s, reveal=reveal)
        cy = uy + 9 * s
    return cy


def _led(ov, x, y, r, state: str, s, t) -> None:
    """Status pip: filled accent=active, blinking hot=fault, grey ring=offline."""
    if state == "ok":
        ov.vector.disc(x, y, r, RED)
        ov.vector.ring(x, y, r + 2.0 * s, _fa(RED, 0.5), 1.0 * s)
    elif state == "fault":
        blink = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 8.0))
        ov.vector.disc(x, y, r, _fa(RED_HOT, blink))
    else:
        ov.vector.ring(x, y, r, _fa(GREY_DIM, 0.9), 1.2 * s)


def _row(ov, x0, x1, y, label, value, vcol: Color, s, *, led: Optional[str] = None,
         t: float = 0.0, vsize=11.5, reveal=1.0) -> None:
    """One instrument row: label left (grey), value right (vcol).  Optional LED."""
    if reveal <= 0.0:
        return
    lx = x0
    if led is not None:
        _led(ov, x0 + 3.5 * s, y + 5.0 * s, 3.0 * s, led, s, t)
        lx = x0 + 11 * s
    ov.text.text(label, lx, y + 1.0 * s, 9.0 * s, GREY_DIM, align="left")
    ov.text.text(value, x1, y - 1.0 * s, vsize * s, vcol, align="right",
                 mode="typeon", t=t, progress=reveal)


def _bar(ov, x0, y, w, frac: float, color: Color, s, *, segs=14, h=4.0) -> None:
    """Segmented horizontal fill bar (0..1)."""
    frac = max(0.0, min(1.0, frac))
    gap = 1.6 * s
    seg_w = (w - gap * (segs - 1)) / segs
    lit = int(round(frac * segs))
    for i in range(segs):
        sx = x0 + i * (seg_w + gap)
        col = color if i < lit else _fa(FRAME, 0.55)
        _fill(ov, sx, y, sx + seg_w, y + h * s, col)


def _bipolar(ov, cx, y, half_w, value: float, scale: float, color: Color, s, *, h=4.0) -> None:
    """Centre-anchored bipolar bar (signed command)."""
    ov.vector.line((cx, y - 2 * s), (cx, y + h * s + 2 * s), _fa(GREY, 0.7), 1.0 * s)
    bw = max(-half_w, min(half_w, value * scale))
    if bw >= 0:
        _fill(ov, cx, y, cx + bw, y + h * s, color)
    else:
        _fill(ov, cx + bw, y, cx, y + h * s, color)


# ------------------------------------------------------------------ top bar
def _topbar(ov, W, H, s, t, seq, st: HudState) -> None:
    p = seq.at(0.0, 0.35, linear)
    y = 0.030 * H
    # identity plate: accent spine + device id
    x0 = 0.020 * W
    _fill(ov, x0, y - 6 * s, x0 + 4 * s, y + 16 * s, st.accent)
    ov.text.text(f"{st.conn_id}", x0 + 10 * s, y - 8 * s, 14 * s, WHITE, align="left")
    ov.text.text(f"OPTICAL FEED // {st.camera_mode}   D435", x0 + 10 * s, y + 9 * s, 9 * s, GREY,
                 align="left")
    # center mission strip
    if seq.at(0.06, 0.4) > 0:
        ov.text.text("FOLLOW + STAIR ASSIST", 0.5 * W, y - 8 * s, 11 * s, st.accent, align="center")
        ov.text.text("O2-THERAPY ESCORT", 0.5 * W, y + 9 * s, 9 * s, GREY, align="center")
    # right: REC + frame + clock + link
    if seq.at(0.1, 0.4) > 0:
        rx = 0.980 * W
        blink = (t % 1.0) < 0.5
        ov.vector.disc(rx - ov.text.measure(f"REC {st.rec_s:04d}", 13 * s) - 10 * s, y + 1 * s,
                       3.6 * s, st.accent if blink else _fa(st.accent, 0.25))
        ov.text.text(f"REC {st.rec_s:04d}", rx, y - 8 * s, 13 * s, WHITE, align="right")
        fps = f"{st.proc_fps:.1f}FPS" if st.proc_fps > 0 else ""
        ov.text.text(f"FRM {st.frame_no % 1000000:06d}  {st.clock_str}  {fps}", rx, y + 9 * s,
                     9 * s, GREY, align="right")
    # ruler divider
    dy = 0.058 * H
    ov.vector.line((0.020 * W, dy), (0.980 * W, dy), _fa(RED_DIM, 0.9), 1.0 * s, reveal=p)
    if p >= 0.99:
        for i in range(21):
            tx = 0.020 * W + (0.960 * W) * (i / 20.0)
            th = 4 * s if i % 5 else 7 * s
            ov.vector.line((tx, dy), (tx, dy - th), _fa(FRAME, 0.8), 1.0 * s)


# -------------------------------------------------------- centre lock reticle
def _reticle(ov, W, H, s, t, seq, st: HudState) -> None:
    cx, cy, hw, hh = st.node
    accent = st.accent
    p = seq.at(0.18, 0.42, out_cubic)
    if p <= 0:
        return

    if st.present:
        # cut-corner lock brackets on the bbox
        pulse = 1.0 + 0.05 * math.sin(t * 4.0) if not st.locked else 1.0
        hw2, hh2 = hw * pulse, hh * pulse
        leg = max(12 * s, min(hw2, hh2) * 0.42)
        for ox, oy, sx, sy in ((-hw2, -hh2, 1, 1), (hw2, -hh2, -1, 1),
                               (-hw2, hh2, 1, -1), (hw2, hh2, -1, -1)):
            bx, by = cx + ox, cy + oy
            ov.vector.polyline([(bx + sx * leg, by), (bx, by), (bx, by + sy * leg)],
                               accent, 2.0 * s, reveal=p)
        if st.locked:
            # convergence ticks pulling inward = LOCK
            for ox, oy, sx, sy in ((-hw2, -hh2, 1, 1), (hw2, -hh2, -1, 1),
                                   (-hw2, hh2, 1, -1), (hw2, hh2, -1, -1)):
                ov.vector.line((cx + ox, cy + oy),
                               (cx + ox - sx * 9 * s, cy + oy - sy * 9 * s), _fa(accent, 0.5), 1.2 * s)
        # centre crosshair
        g = 7 * s
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ov.vector.line((cx + dx * g, cy + dy * g), (cx + dx * g * 2.6, cy + dy * g * 2.6),
                           _fa(accent, 0.85), 1.3 * s)
        ov.vector.disc(cx, cy, 2.0 * s, accent)
    else:
        # searching: rotating scan reticle
        r = max(0.05 * W, min(hw, hh) * 0.9)
        ov.vector.ring(cx, cy, r, _fa(accent, 0.8), 1.6 * s, reveal=p)
        ov.vector.ring(cx, cy, r * 0.62, _fa(accent, 0.35), 1.1 * s, reveal=p)
        a = t * 1.8
        ov.vector.line((cx, cy), (cx + math.cos(a) * r, cy + math.sin(a) * r), _fa(accent, 0.7), 1.5 * s)
        for k in range(4):
            ka = k * math.pi / 2 + t * 0.6
            ov.vector.line((cx + math.cos(ka) * r * 1.06, cy + math.sin(ka) * r * 1.06),
                           (cx + math.cos(ka) * r * 1.22, cy + math.sin(ka) * r * 1.22),
                           _fa(accent, 0.6), 1.3 * s)

    # headline callout above the reticle — clamped clear of the top bar/flash,
    # with a faint leader down to the bbox when there's a gap.  (Bearing lives in
    # the TARGET panel; not repeated here, to keep the reticle uncluttered.)
    # Suppressed while fallen: the fall banner owns the top-centre alert zone.
    if seq.at(0.30, 0.4) > 0 and not st.fell:
        dtxt = _fmt_m(st.dist) if (st.present and st.dist is not None) else ""
        head = {"FALLEN": "PATIENT // ASSIST", "LOCKED": "PATIENT-01 // LOCK",
                "TRACKING": "PATIENT-01 // TRACK", "SEARCHING": "REACQUIRING",
                "NO SIGNAL": "SCANNING"}[st.track_word]
        top = cy - (hh if st.present else 0.055 * H)
        ly = max(0.100 * H, top - 0.030 * H)
        gap = ov.text.measure(head + "   ", 12 * s)
        tw = gap + (ov.text.measure(dtxt, 12 * s) if dtxt else -ov.text.measure("   ", 12 * s))
        lx = cx - tw * 0.5
        ov.vector.polyline([(lx - 9 * s, ly - 2 * s), (lx - 9 * s, ly + 14 * s)], _fa(accent, 0.9), 2.0 * s)
        ov.text.text(head, lx, ly, 12 * s, WHITE, align="left")
        if dtxt:
            ov.text.text(dtxt, lx + gap, ly, 12 * s, accent, align="left")
        if st.present and (top - (ly + 16 * s)) > 6 * s:
            ov.vector.line((cx, ly + 16 * s), (cx, top - 2 * s), _fa(accent, 0.4), 1.0 * s)


# --------------------------------------------------------------- left column
def _left(ov, W, H, s, t, seq, st: HudState) -> None:
    x0, x1 = 0.020 * W, 0.234 * W
    inx = x1 - 8 * s

    # SENSORS panel
    p = seq.stagger(0, 0.30, 0.10, 0.4, out_cubic)
    if p > 0:
        cy = _panel(ov, x0, 0.090 * H, x1, 0.250 * H, s=s, reveal=p, accent=st.accent, title="SENSORS")
        rh = 0.036 * H
        _row(ov, x0 + 8 * s, inx, cy, "CAM D435", "LINK OK" if st.cam_ok else "FAULT",
             WHITE if st.cam_ok else RED_HOT, s, led="ok" if st.cam_ok else "fault", t=t, reveal=p)
        lw = (f"{int(st.lidar_dec.get('hit_count', 0))}/{int(st.lidar_dec.get('ray_count', 0))} HIT"
              if st.lidar_ok else "OFFLINE")
        _row(ov, x0 + 8 * s, inx, cy + rh, "LIDAR XT16", lw, WHITE if st.lidar_ok else GREY, s,
             led="ok" if st.lidar_ok else "off", t=t, reveal=p)
        _row(ov, x0 + 8 * s, inx, cy + 2 * rh, "IMU LINK", "NOMINAL" if st.imu_ok else "NO TELEM",
             WHITE if st.imu_ok else GREY, s, led="ok" if st.imu_ok else "off", t=t, reveal=p)

    # ATTITUDE panel
    p = seq.stagger(1, 0.30, 0.10, 0.4, out_cubic)
    if p > 0:
        cy = _panel(ov, x0, 0.262 * H, x1, 0.446 * H, s=s, reveal=p, accent=st.accent, title="ATTITUDE")
        rh = 0.036 * H
        rp = (f"{st.roll:+.0f} / {st.pitch:+.0f}" if st.roll is not None and st.pitch is not None else "N/A")
        _row(ov, x0 + 8 * s, inx, cy, "ROLL / PITCH", rp, WHITE if st.roll is not None else GREY, s,
             t=t, reveal=p)
        # tilt bar (|roll| + |pitch| toward a danger ceiling)
        tilt = 0.0
        if st.roll is not None and st.pitch is not None:
            tilt = min(1.0, (abs(st.roll) + abs(st.pitch)) / 60.0)
        _bar(ov, x0 + 8 * s, cy + rh * 0.65, inx - (x0 + 8 * s), tilt,
             RED_HOT if tilt > 0.6 else st.accent, s)
        _row(ov, x0 + 8 * s, inx, cy + rh * 1.35, "BODY HEIGHT",
             _fmt_m(st.height) if st.height is not None else "N/A", WHITE, s, t=t, reveal=p)
        status = f"FALLEN {st.fall_type.upper()[:7]}" if st.fell else ("UPRIGHT" if st.imu_ok else "NO TELEM")
        _row(ov, x0 + 8 * s, inx, cy + rh * 2.05, "STANCE", status,
             RED_HOT if st.fell else (WHITE if st.imu_ok else GREY), s, t=t, reveal=p,
             led="fault" if st.fell else ("ok" if st.imu_ok else "off"))


# -------------------------------------------------------------- right column
def _right(ov, W, H, s, t, seq, st: HudState) -> None:
    x0, x1 = 0.766 * W, 0.980 * W
    inx = x1 - 8 * s
    lx = x0 + 8 * s

    # TARGET panel
    p = seq.stagger(0, 0.34, 0.10, 0.4, out_cubic)
    if p > 0:
        badge = ("!DISAGREE", RED_HOT) if st.disagreement else None
        cy = _panel(ov, x0, 0.090 * H, x1, 0.300 * H, s=s, reveal=p, accent=st.accent,
                    title="TARGET", badge=badge)
        # big track-state headline
        ov.text.text(st.track_word, lx, cy - 2 * s, 17 * s, st.accent, align="left",
                     mode="typeon", t=t, progress=p)
        cy += 0.036 * H
        rh = 0.034 * H
        _row(ov, lx, inx, cy, "CLASS", "O2 PATIENT", WHITE, s, t=t, reveal=p)
        _row(ov, lx, inx, cy + rh, "DISTANCE", _fmt_m(st.dist) if st.dist is not None else "N/A",
             WHITE, s, t=t, reveal=p, vsize=12.5)
        _row(ov, lx, inx, cy + 2 * rh, "BEARING",
             f"{st.bearing:+.1f}DEG" if st.bearing is not None else "N/A", WHITE, s, t=t, reveal=p)
        _row(ov, lx, inx, cy + 3 * rh, "FUSION",
             st.fusion_src if st.fusion_src not in ("N/A", "NONE") else "---",
             st.accent if st.fusion_src not in ("N/A", "NONE") else GREY, s, t=t, reveal=p, vsize=10)

    # MISSION / STAIRS panel
    p = seq.stagger(1, 0.34, 0.10, 0.4, out_cubic)
    if p > 0:
        det = st.stairs_det
        badge = (f"{st.stairs_conf * 100:.0f}%", st.accent) if det else ("SCAN", GREY)
        cy = _panel(ov, x0, 0.312 * H, x1, 0.470 * H, s=s, reveal=p, accent=st.accent,
                    title="STAIRS", badge=badge)
        rh = 0.034 * H
        _row(ov, lx, inx, cy, "DETECT", "YES" if det else "SEARCHING",
             st.accent if det else GREY, s, t=t, reveal=p, led="ok" if det else "off")
        _bar(ov, lx, cy + rh * 0.72, inx - lx, st.stairs_conf if det else 0.0,
             st.accent, s)
        _row(ov, lx, inx, cy + rh * 1.35, "RISE / GAP",
             _fmt_m(st.stairs_depth) if st.stairs_depth is not None else "N/A", WHITE, s, t=t, reveal=p)
        _row(ov, lx, inx, cy + rh * 2.05, "TASK", "ASCEND ESCORT" if det else "FOLLOW",
             WHITE, s, t=t, reveal=p, vsize=10)


# ---------------------------------------------------------- depth inset frame
def _depth_frame(ov, W, H, s, t, seq, st: HudState) -> None:
    if st.depth_rect is None:
        return
    p = seq.at(0.6, 0.4)
    if p <= 0:
        return
    x0, y0, x1, y1 = st.depth_rect
    badge = (f"STAIRS {st.stairs_conf * 100:.0f}%", st.accent) if st.stairs_det else ("SCAN", GREY)
    _panel(ov, x0 - 6 * s, y0 - 22 * s, x1 + 6 * s, y1 + 6 * s, s=s, reveal=p,
           accent=st.accent, title="DEPTH // D435", badge=badge)
    # scan sweep line across the inset while active
    if p >= 0.99:
        sweep = (math.sin(t * 1.3) * 0.5 + 0.5)
        sx = x0 + (x1 - x0) * sweep
        ov.vector.line((sx, y0), (sx, y1), _fa(st.accent, 0.25), 1.0 * s)


# --------------------------------------------------------- bottom loco strip
def _bottom(ov, W, H, s, t, seq, st: HudState) -> None:
    p = seq.at(0.55, 0.45, out_cubic)
    if p <= 0:
        return
    x0, x1 = 0.278 * W, 0.722 * W
    y0, y1 = 0.896 * H, 0.974 * H
    _panel(ov, x0, y0, x1, y1, s=s, reveal=p, accent=st.accent)
    span = x1 - x0
    ax = x0 + span * 0.22
    bx = x0 + span * 0.72
    ov.vector.line((ax, y0 + 8 * s), (ax, y1 - 8 * s), _fa(FRAME, 0.7), 1.0 * s)
    ov.vector.line((bx, y0 + 8 * s), (bx, y1 - 8 * s), _fa(FRAME, 0.7), 1.0 * s)

    # cell A: 4-leg gait diagram (top-view; front = top)
    gx = x0 + (ax - x0) * 0.5
    gcy = y0 + (y1 - y0) * 0.52
    ov.text.text("GAIT", x0 + 10 * s, y0 + 6 * s, 8.5 * s, GREY_DIM, align="left")
    bw, bh = 0.008 * W, 0.014 * H
    ov.vector.rect(gx - bw, gcy - bh, gx + bw, gcy + bh, _fa(st.accent, 0.7), 1.2 * s)
    for sx, sy, name in ((-1, -1, "FL"), (1, -1, "FR"), (-1, 1, "RL"), (1, 1, "RR")):
        if st.swing_legs:
            swing = name in st.swing_legs
        else:
            swing = math.sin(t * 2.4 + (0 if sx * sy < 0 else 1) * math.pi) <= 0
        px, py = gx + sx * (bw + 0.009 * W), gcy + sy * (bh + 0.004 * H)
        if swing:
            ov.vector.ring(px, py, 3.0 * s, _fa(GREY, 0.9), 1.3 * s)
        else:
            ov.vector.disc(px, py, 3.0 * s, st.accent)
    ov.text.text(st.loco_gait[:11], gx, y1 - 11 * s, 7.5 * s, GREY, align="center")

    # cell B: locomotion readouts
    lx, rx = ax + 14 * s, bx - 12 * s
    ov.text.text("MODE", lx, y0 + 6 * s, 8.5 * s, GREY_DIM, align="left")
    mode_col = RED_HOT if st.loco_mode == "CLIMB" else st.accent
    ov.text.text(st.loco_mode, lx, y0 + 16 * s, 16 * s, mode_col, align="left",
                 mode="typeon", t=t, progress=p)
    sp = f"{st.loco_speed:.2f} M/S" if st.loco_speed is not None else "N/A"
    ov.text.text("SPEED", rx, y0 + 6 * s, 8.5 * s, GREY_DIM, align="right")
    ov.text.text(sp, rx, y0 + 16 * s, 13 * s, WHITE, align="right")
    clr = f"FOOT CLR {st.loco_clear * 100:.0f}CM" if st.loco_clear is not None else "FOOT CLR --"
    ov.text.text(f"POLICY {st.loco_policy[:18]}", lx, y1 - 12 * s, 7.5 * s, GREY, align="left")
    ov.text.text(clr, rx, y1 - 12 * s, 7.5 * s, GREY, align="right")

    # cell C: command bipolar bars (V forward, W yaw)
    ccx = bx + (x1 - bx) * 0.54
    half = (x1 - bx) * 0.24
    ov.text.text("CMD", bx + 12 * s, y0 + 6 * s, 8.5 * s, GREY_DIM, align="left")
    vy = y0 + (y1 - y0) * 0.44
    wy = y0 + (y1 - y0) * 0.74
    ov.text.text("V", bx + 12 * s, vy - 5 * s, 8.5 * s, GREY, align="left")
    _bipolar(ov, ccx, vy, half, st.cmd_v, half / 0.6, st.accent, s)
    ov.text.text(f"{st.cmd_v:+.2f}", x1 - 8 * s, vy - 5 * s, 8.5 * s, WHITE, align="right")
    ov.text.text("W", bx + 12 * s, wy - 5 * s, 8.5 * s, GREY, align="left")
    _bipolar(ov, ccx, wy, half, st.cmd_w, half / 0.7, st.accent, s)
    ov.text.text(f"{st.cmd_w:+.2f}", x1 - 8 * s, wy - 5 * s, 8.5 * s, WHITE, align="right")


# -------------------------------------------------------------- sector radar
_SECTORS = [-math.pi / 2, -math.pi / 3, -math.pi / 6, 0.0, math.pi / 6, math.pi / 3, math.pi / 2]


def _radar(ov, W, H, s, t, seq, st: HudState) -> None:
    p = seq.at(0.62, 0.4)
    if p <= 0:
        return
    x0, x1 = 0.766 * W, 0.980 * W
    y0, y1 = 0.560 * H, 0.820 * H
    _panel(ov, x0, y0, x1, y1, s=s, reveal=p, accent=st.accent, title="LIDAR // FWD-180")
    cx, cy = 0.5 * (x0 + x1), y1 - 0.052 * H
    r = min((x1 - x0) * 0.34, (y1 - y0) * 0.52)

    if not st.lidar_ok:
        ov.vector.ring(cx, cy, r, _fa(GREY_DIM, 0.9), 1.2 * s, a0=-math.pi / 2, a1=math.pi / 2, reveal=p)
        ov.text.text("OFFLINE", cx, cy - 6 * s, 10 * s, GREY, align="center")
        return
    # forward half-rings
    for rr, aa in ((r, 1.3), (r * 0.66, 0.9), (r * 0.33, 0.6)):
        ov.vector.ring(cx, cy, rr, _fa(st.accent, 0.28 + 0.12 * (rr / r)), 1.1 * s,
                       a0=-math.pi / 2, a1=math.pi / 2, reveal=p)
    for ang in (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2):
        ov.vector.line((cx, cy), (cx + math.sin(ang) * r, cy - math.cos(ang) * r), _fa(FRAME, 0.6), 1.0 * s)
    # sweeping beam
    if p >= 0.99:
        bang = math.sin(t * 0.9) * (math.pi / 2)
        ov.vector.line((cx, cy), (cx + math.sin(bang) * r, cy - math.cos(bang) * r), _fa(st.accent, 0.5), 1.4 * s)

    ranges = st.lidar_dec.get("ranges_m", []) if st.lidar_dec else []
    n = len(ranges) if ranges is not None else 0
    view = max(0.5, float(st.lidar_dec.get("view_range_m", 6.0))) if st.lidar_dec else 6.0
    fwd = None
    for ang in _SECTORS:
        rr = None
        if n:
            idx = int(((ang / _TAU) % 1.0) * n) % n
            v = float(ranges[idx])
            rr = v if v > 0 else None
        frac = (min(rr, view) / view) if rr else 1.0
        px = cx + math.sin(ang) * r * frac
        py = cy - math.cos(ang) * r * frac
        if rr is not None:
            danger = rr < 0.8
            ov.vector.disc(px, py, 2.4 * s, RED_HOT if danger else st.accent)
            ov.vector.line((cx + math.sin(ang) * 3 * s, cy - math.cos(ang) * 3 * s), (px, py),
                           _fa(RED_HOT if danger else st.accent, 0.35), 1.0 * s)
        if ang == 0.0:
            fwd = rr
    if st.person_bearing_rad is not None:
        mp_x = cx + math.sin(st.person_bearing_rad) * r
        mp_y = cy - math.cos(st.person_bearing_rad) * r
        ov.vector.ring(mp_x, mp_y, 4.0 * s, _fa(WHITE, 0.95), 1.4 * s)
        ov.vector.disc(mp_x, mp_y, 1.6 * s, WHITE)
    ov.vector.disc(cx, cy, 2.2 * s, _fa(st.accent, 0.9))
    if fwd is None:
        ov.text.text("FWD  --", cx, y1 - 16 * s, 10 * s, GREY, align="center")
    else:
        lab = "DANGER" if fwd < 0.8 else ("CAUTION" if fwd < 1.5 else "CLEAR")
        col = RED_HOT if fwd < 0.8 else st.accent
        ov.text.text(f"FWD {fwd:.2f}M", cx - 4 * s, y1 - 16 * s, 10 * s, WHITE, align="right")
        ov.text.text(lab, cx + 8 * s, y1 - 16 * s, 10 * s, col, align="left")


# ------------------------------------------------------------------- alerts
def _flash_cue(ov, W, H, s, st: HudState) -> None:
    msg = {"acquire": "TARGET ACQUIRED", "lock": "TARGET LOCKED",
           "unlock": "LOCK DROPPED // TRACKING", "lost": "SIGNAL LOST // REACQUIRING"}.get(st.flash_kind, "")
    if not msg or st.flash_amt <= 0.02:
        return
    col = RED_HOT if st.flash_kind in ("lost", "unlock") else st.accent
    a = min(1.0, st.flash_amt + 0.3)
    fy = H * 0.070
    ov.text.text(msg, W * 0.5, fy, 13 * s, _fa(col, a), align="center")
    tw = ov.text.measure(msg, 13 * s)
    ov.vector.line((W * 0.5 - tw * 0.5 - 8 * s, fy + 16 * s),
                   (W * 0.5 + tw * 0.5 + 8 * s, fy + 16 * s), _fa(col, a * 0.8), 1.2 * s)


def _fall_banner(ov, W, H, s, t, st: HudState) -> None:
    if not st.fell:
        return
    fl = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 8.0))
    msg = f"[ ROBOT FALLEN // {st.fall_type.upper()} ]"
    ov.text.text(msg, W * 0.5, H * 0.145, 15 * s, _fa(RED_HOT, fl), align="center")


def _alert_border(ov, W, H, s, t, st: HudState) -> None:
    if not (st.fell or (not st.present and st.reacquire)):
        return
    fl = 0.30 + 0.35 * (0.5 + 0.5 * math.sin(t * 9.0))
    m = 0.010 * W
    c = 22 * s
    col = _fa(RED_HOT if st.fell else RED, fl)
    # corner brackets only (not a full box) — reads as an alert frame, less heavy
    for bx, by, sx, sy in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        ov.vector.polyline([(bx + sx * c, by), (bx, by), (bx, by + sy * c)], col, 2.2 * s)


# ---------------------------------------------------------------- entrypoint
def build(ov, W, H, t, st: HudState) -> None:
    s = H / 600.0
    seq = Sequencer(t)
    _topbar(ov, W, H, s, t, seq, st)
    _left(ov, W, H, s, t, seq, st)
    _right(ov, W, H, s, t, seq, st)
    _depth_frame(ov, W, H, s, t, seq, st)
    _radar(ov, W, H, s, t, seq, st)
    _bottom(ov, W, H, s, t, seq, st)
    _reticle(ov, W, H, s, t, seq, st)
    _flash_cue(ov, W, H, s, st)
    _fall_banner(ov, W, H, s, t, st)
    _alert_border(ov, W, H, s, t, st)


def build_paused(ov, W, H, t) -> None:
    s = H / 600.0
    cx, cy = W * 0.5, H * 0.5
    seq = Sequencer(t)
    p = seq.at(0.0, 0.5, out_cubic)
    # standby plate
    pw, ph = 0.30 * W, 0.11 * H
    _panel(ov, cx - pw * 0.5, cy - ph * 0.5, cx + pw * 0.5, cy + ph * 0.5, s=s, reveal=p,
           accent=RED, title="SYSTEM // STANDBY")
    blink = 0.5 + 0.5 * math.sin(t * 3.0)
    ov.text.text("PREPARATION MODE", cx, cy - 6 * s, 15 * s, _fa(WHITE, 0.7 + 0.3 * blink), align="center")
    ov.text.text("ROBOT HELD // PRESS  P  TO RESUME FOLLOW", cx, cy + 14 * s, 10 * s, GREY, align="center")


__all__ = ["HudState", "derive", "build", "build_paused"]
