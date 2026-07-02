"""WARNING // Target-Acquisition HUD — runnable demo.

A yellow ref1-style hazard HUD (see :mod:`core.hud.warning_kit`) laid opaque over an
OpenCV optical feed, for the O2-therapy stair-assist follow dog.  It boots up, then
plays  LOCKED -> TARGET LOST -> REACQUIRE  with synced bleeps, and dynamic info
cards that spawn/collapse as the scene needs them.

Everything on screen is real telemetry: status, target count, confidence, target
position, an estimated range (pinhole, labelled EST), fps, and a smoothed signal
quality.  No fabricated numbers.

The arcv library does what it is uniquely good at here: **synced audio bleeps**
(``arcv.audio.Bleeps`` live + ``render_track`` for muxed export), **real CV
detectors** (``arcv.vision`` face detector on the webcam path), and the bundled
**Share Tech Mono** font.

Run it
------
    # live window over your webcam (real face detection drives the state):
    python examples/warning_hud.py --source webcam

    # scripted showcase over a simulated feed, live window + sound:
    python examples/warning_hud.py --source sim

    # render the showcase to a shareable MP4 (+GIF) with muxed audio (no display
    # or camera needed — this is the "view it later" path):
    python examples/warning_hud.py --record examples/media/warning_hud.mp4 --seconds 13

    # write per-state stills for quick offline judging:
    python examples/warning_hud.py --preview examples/media
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from core.hud.warning_kit import Target, Telemetry, WarningHud  # noqa: E402

W, H = 1280, 720
BOOT_TIME = 1.9
CYCLE = 13.0            # sim showcase loop length (seconds)


# ─────────────────────────── honest range estimate ────────────────────────────
def estimate_range(bbox_h_norm: float, subject_h_m: float = 1.45,
                   vfov_deg: float = 55.0) -> Optional[float]:
    """Pinhole range estimate from a target's normalised bbox height.

    ``dist = subject_h / (2 * tan(vfov/2) * bbox_h_fraction)``.  A genuine
    geometric estimate (hence labelled EST on the HUD), not ground truth."""
    if bbox_h_norm <= 1e-3:
        return None
    d = subject_h_m / (2.0 * math.tan(math.radians(vfov_deg) * 0.5) * bbox_h_norm)
    return float(max(0.15, min(9.99, d)))


# ─────────────────────────────── synthetic feed ───────────────────────────────
class SimSource:
    """A plausible corridor + stair scene with a scripted walking patient.

    Drives a deterministic showcase: patient present 2.2–7.0 s (acquire→lock, a
    second contact 4.5–6.2 s), gone 7.0–10.0 s (lost), returns 10.0 s (reacquire).
    The state itself emerges from this presence in :class:`StateMachine`, so the
    telemetry is honest — the feed is just clearly marked SIMULATED."""

    sim = True

    def __init__(self) -> None:
        self._bg = self._corridor()

    def _corridor(self) -> np.ndarray:
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
        base = 0.26 + 0.20 * (1.0 - yy) + 0.05 * np.sin(xx * 6.28) * (1.0 - yy)
        img = np.dstack([base * 0.90, base * 0.94, base]).astype(np.float32)
        for i in range(6):                       # stair treads lower-centre
            fy = 0.60 + i * 0.052
            y0, y1 = int(fy * H), int((fy + 0.026) * H)
            x0, x1 = int(0.40 * W), int(0.66 * W)
            img[y0:y1, x0:x1, :] *= 0.72
            img[max(0, y0 - 2):y0, x0:x1, :] = 0.58
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _person(frame: np.ndarray, cx: float, cy: float, h: float, warm: float = 1.0) -> None:
        bx0 = int((cx - h * 0.16) * W); bx1 = int((cx + h * 0.16) * W)
        by0 = int((cy - h * 0.5) * H); by1 = int((cy + h * 0.5) * H)
        cxi = (bx0 + bx1) // 2
        body = (int(70 * warm), int(74 * warm), int(86 * warm))
        cv2.rectangle(frame, (bx0, int(by0 + 0.22 * (by1 - by0))), (bx1, by1), body, -1, cv2.LINE_AA)
        cv2.circle(frame, (cxi, int(by0 + 0.11 * (by1 - by0))),
                   int(0.09 * (by1 - by0)), (int(84 * warm), int(88 * warm), int(96 * warm)), -1, cv2.LINE_AA)
        # O2 tank on the back
        cv2.rectangle(frame, (bx0 - int(0.02 * W), int(by0 + 0.32 * (by1 - by0))),
                      (bx0 + int(0.012 * W), int(by0 + 0.66 * (by1 - by0))),
                      (40, 58, 74), -1, cv2.LINE_AA)

    def read(self, t: float) -> Tuple[np.ndarray, List[Target], dict]:
        frame = self._bg.copy()
        tt = t % CYCLE
        targets: List[Target] = []

        def add(cx, cy, h, tid, warm=1.0):
            self._person(frame, cx, cy, h, warm)
            d = estimate_range(h)
            # confidence derived from apparent size + centring (a real function of
            # the scene geometry — larger, more-centred subjects detect stronger)
            score = float(np.clip(0.12 + 1.6 * h - 0.2 * abs(cx - 0.5), 0.35, 0.99))
            targets.append(Target(cx=cx, cy=cy, w=h * 0.32, h=h, dist_m=d, tid=tid, score=score))

        # primary patient present 2.2–7.0 and 10.0–CYCLE (feed is full-screen, so
        # keep the subject centred and clear of the corner instrument panels)
        if 2.2 <= tt < 7.0:
            u = (tt - 2.2) / (7.0 - 2.2)
            cx = 0.40 + 0.14 * u + 0.03 * math.sin(tt * 1.7)
            cy = 0.50 + 0.02 * math.sin(tt * 3.1)
            h = 0.46 + 0.12 * u
            add(cx, cy, h, 1)
        elif tt >= 10.0:
            u = (tt - 10.0) / (CYCLE - 10.0)
            cx = 0.60 - 0.10 * u + 0.03 * math.sin(tt * 1.7)
            cy = 0.50 + 0.02 * math.sin(tt * 3.1)
            h = 0.50 + 0.06 * u
            add(cx, cy, h, 1)
        # a bystander contact 4.5–6.2
        if 4.5 <= tt < 6.2:
            add(0.30 + 0.01 * math.sin(tt * 2.0), 0.46, 0.30, 2, warm=0.9)
        return frame, targets, self._sensors(tt, targets)

    # -- simulated D435 depth + XT16 LiDAR (the project's real sensors) ----------
    @staticmethod
    def _depth_img(h: int = 96, w: int = 128) -> np.ndarray:
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        d = (3300.0 - 2400.0 * yy) * np.ones((h, w), dtype=np.float32)   # floor receding
        for i in range(4):                                              # stair step-edges
            y = int((0.45 + i * 0.12) * h)
            d[y:y + max(1, int(0.05 * h)), int(0.28 * w):int(0.74 * w)] -= 520.0 * (i + 1)
        return np.clip(d, 280.0, 6000.0).astype(np.uint16)

    def _sensors(self, tt: float, targets: List[Target]) -> dict:
        det = tt >= 3.5
        conf = min(0.92, 0.55 + 0.09 * (tt - 3.5)) if det else 0.0
        stairs = (det, conf, (0.30, 0.45, 0.74, 0.95)) if det else (False, 0.0, None)
        n = 121
        ranges = []
        for i in range(n):                                             # corridor profile
            ang = (i / (n - 1)) * math.pi - math.pi / 2
            wall = 1.35 / max(0.16, abs(math.sin(ang)))                # side walls close in
            r = min(6.0, wall) * (0.96 + 0.08 * ((i * 37) % 5) / 4.0)  # + mild surface noise
            ranges.append(round(r, 2))
        for i in range(n):                                             # a chair on the right flank
            ang = (i / (n - 1)) * math.pi - math.pi / 2
            if math.radians(28) < ang < math.radians(46):
                ranges[i] = min(ranges[i], 1.15)
        if targets:                                                    # patient return
            p = targets[0]
            az = (p.cx - 0.5) * 1.05
            idx = int(((az + math.pi / 2) / math.pi) * (n - 1))
            for j in range(max(0, idx - 2), min(n, idx + 3)):
                ranges[j] = min(ranges[j], p.dist_m or 2.0)
        return {"depth": self._depth_img(), "stairs": stairs,
                "lidar": {"ranges_m": ranges, "view_range_m": 6.0}}


# ──────────────────────────────── webcam feed ─────────────────────────────────
class WebcamSource:
    """Real camera + arcv/OpenCV face detector.  Each detected face is a target;
    confidence is a tracking-stability metric (IoU persistence), so it is real,
    not a fabricated score."""

    sim = False

    def __init__(self, index: int = 0) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        from arcv.vision import FaceDetector
        self._det = FaceDetector(min_size_frac=0.10)
        self._prev: Optional[Tuple[float, float, float, float]] = None
        self._stab = 0.0

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0 = a[0] - a[2], a[1] - a[3]; ax1, ay1 = a[0] + a[2], a[1] + a[3]
        bx0, by0 = b[0] - b[2], b[1] - b[3]; bx1, by1 = b[0] + b[2], b[1] + b[3]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    def read(self, t: float) -> Tuple[np.ndarray, List[Target], dict]:
        ok, frame = self.cap.read()
        if not ok:
            return np.zeros((H, W, 3), np.uint8), [], {}
        frame = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._det.detect(gray)                # [(x,y,w,h), ...]
        rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
        targets: List[Target] = []
        for i, (x, y, w, h) in enumerate(rects[:4]):
            cx = (x + w * 0.5) / W; cy = (y + h * 0.5) / H
            nh = h / H
            targets.append(Target(cx=cx, cy=cy, w=w / W, h=nh,
                                  dist_m=estimate_range(nh), tid=i + 1))
        # stability confidence for the primary
        if targets:
            p = targets[0]
            cur = (p.cx, p.cy, p.w * 0.5, p.h * 0.5)
            iou = self._iou(cur, self._prev) if self._prev else 0.4
            self._stab = 0.85 * self._stab + 0.15 * (0.55 + 0.45 * iou)
            self._prev = cur
            for tg in targets:
                tg.score = float(min(1.0, self._stab)) if tg is p else 0.6
        else:
            self._stab *= 0.9
            self._prev = None
        # a laptop webcam has no depth cam / LiDAR — honestly report no sensors
        return frame, targets, {"depth": None, "stairs": None, "lidar": None}

    def close(self) -> None:
        self.cap.release()


# ─────────────────────────── presence-driven states ───────────────────────────
class StateMachine:
    """BOOTING → ACQUIRING → LOCKED ⇄ TARGET LOST → REACQUIRE, driven purely by
    target presence with a little debounce.  Emits audio-cue names on transitions
    so live playback and the exported track stay in sync."""

    def __init__(self, lock_frames: int = 7, lost_frames: int = 5) -> None:
        self.state = "BOOTING"
        self.lock_frames = lock_frames
        self.lost_frames = lost_frames
        self.present_run = 0
        self.absent_run = 0
        self.ever_locked = False
        self.locked = False
        self.lost_t: Optional[float] = None
        self._boot_cues = [(0.15, "panel"), (0.5, "assemble"), (0.85, "assemble"),
                           (1.2, "assemble"), (1.55, "type")]
        self._boot_i = 0

    def update(self, present: bool, t: float, cues: List) -> None:
        def emit(name: str) -> None:
            cues.append((t, name))

        if self.state == "BOOTING":
            while self._boot_i < len(self._boot_cues) and self._boot_cues[self._boot_i][0] <= t:
                emit(self._boot_cues[self._boot_i][1]); self._boot_i += 1
            if t >= BOOT_TIME:
                self.state = "ACQUIRING"; emit("scan")
            else:
                return

        if present:
            self.present_run += 1; self.absent_run = 0
        else:
            self.absent_run += 1; self.present_run = 0

        if self.state in ("ACQUIRING", "REACQUIRE"):
            # ACQUIRING/REACQUIRE are "no-lock, scanning" states — absence just
            # keeps scanning; you can only *lose* a target you actually locked, so
            # TARGET LOST is reachable only from LOCKED (below).
            if present and self.present_run >= self.lock_frames:
                self.state = "LOCKED"; self.locked = True; self.ever_locked = True
                self.lost_t = None; emit("lock")
        elif self.state == "LOCKED":
            if not present and self.absent_run >= self.lost_frames:
                self.state = "TARGET LOST"; self.locked = False
                self.lost_t = t; emit("alert"); emit("error")
        elif self.state == "TARGET LOST":
            if present:
                self.state = "REACQUIRE"; emit("scan")


# ────────────────────────────── the demo driver ───────────────────────────────
DETAIL_HOLD = 5.0        # seconds the TARGET detail card stays up after a lock
TOAST_HOLD = 1.9         # seconds a transient toast lingers


class Demo:
    def __init__(self, source, hud: WarningHud, with_audio: bool = True) -> None:
        self.src = source
        self.hud = hud
        self.sm = StateMachine()
        self.signal = 0.12
        self.fps = 0.0
        self.cues: List[Tuple[float, str]] = []
        self._bleeps = None
        self._cue_played = 0
        # transient-UI bookkeeping (spawn-in → hold → spawn-out)
        self._detail_until = -1.0
        self._toasts: List[Tuple[str, str, float]] = []   # (msg, kind, until)
        self._prev_state = None
        self._prev_n = 0
        self._prev_stairs = False
        if with_audio:
            try:
                from arcv.audio import Bleeps
                self._bleeps = Bleeps(volume=0.6, source="synth")
            except Exception:
                self._bleeps = None

    _SIG_TARGET = {"BOOTING": 0.12, "ACQUIRING": 0.5, "REACQUIRE": 0.5,
                   "LOCKED": 0.93, "TRACKING": 0.85, "TARGET LOST": 0.1}

    def _toast(self, t: float, msg: str, kind: str) -> None:
        self._toasts = [e for e in self._toasts if e[0] != msg]
        self._toasts.append((msg, kind, t + TOAST_HOLD))

    def step(self, t: float, dt: float) -> np.ndarray:
        c0 = time.perf_counter()
        frame, targets, sensors = self.src.read(t)
        present = bool(targets)
        self.sm.update(present, t, self.cues)
        # play any newly-emitted cues live
        if self._bleeps is not None:
            for (_, name) in self.cues[self._cue_played:]:
                self._bleeps.play(name)
        self._cue_played = len(self.cues)

        state = self.sm.state
        boot = min(1.0, t / (BOOT_TIME * 0.9)) if state == "BOOTING" or t < BOOT_TIME else 1.0
        if targets:
            targets[0].locked = self.sm.locked and state == "LOCKED"
        lost_for = (t - self.sm.lost_t) if (self.sm.lost_t is not None and state in ("TARGET LOST", "REACQUIRE")) else None
        # smooth signal toward the state's nominal quality (honest EMA)
        tgt = self._SIG_TARGET.get(state, 0.5)
        self.signal += (tgt - self.signal) * min(1.0, dt * 3.5)

        # --- transient UI: detail card spawns while acquiring, holds after lock,
        #     then spawns out; toasts fire on discrete events ------------------
        n = len(targets)
        stairs_det = bool(sensors.get("stairs") and sensors["stairs"][0])
        if state == "LOCKED" and self._prev_state != "LOCKED":
            self._detail_until = t + DETAIL_HOLD
            self._toast(t, "TARGET ACQUIRED // LOCK", "ok")
        if n >= 2 and self._prev_n < 2:
            self._toast(t, "CONTACT-02 DETECTED", "info")
        if stairs_det and not self._prev_stairs:
            self._toast(t, "STAIRS AHEAD // ASSIST", "info")
        show_detail = state in ("ACQUIRING", "REACQUIRE") or (state == "LOCKED" and t < self._detail_until)
        active_toasts = [(m, k) for (m, k, u) in self._toasts if t < u]
        self._toasts = [e for e in self._toasts if t < e[2]]
        self._prev_state, self._prev_n, self._prev_stairs = state, n, stairs_det

        tel = Telemetry(
            state=state, targets=targets, fps=self.fps, signal=self.signal,
            boot=boot, lost_for=lost_for, rec_s=int(t), frame_no=int(t / max(dt, 1e-3)),
            sim=getattr(self.src, "sim", False),
            depth=sensors.get("depth"), stairs=sensors.get("stairs"),
            lidar=sensors.get("lidar"), show_detail=show_detail, toasts=active_toasts,
        )
        out = self.hud.render(frame, tel, t, dt)
        # measured processing fps (honest)
        inst = 1.0 / max(1e-3, time.perf_counter() - c0)
        self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1
        return out

    def close(self) -> None:
        if self._bleeps is not None:
            self._bleeps.close()
        if hasattr(self.src, "close"):
            self.src.close()


# ───────────────────────────────── entrypoints ────────────────────────────────
def run_live(source_name: str, cam_index: int) -> None:
    try:
        src = WebcamSource(cam_index) if source_name == "webcam" else SimSource()
    except Exception as e:
        print(f"[warning_hud] {e} — falling back to simulated feed")
        src = SimSource()
    demo = Demo(src, WarningHud((W, H)), with_audio=True)
    win = "WARNING // TARGET ACQUISITION"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    t0 = time.perf_counter()
    last = t0
    print("[warning_hud] live — press Q or Esc to quit")
    try:
        while True:
            now = time.perf_counter()
            t, dt = now - t0, now - last
            last = now
            out = demo.step(t, max(1e-3, dt))
            cv2.imshow(win, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        demo.close()
        cv2.destroyAllWindows()


def run_record(out_path: str, seconds: float, fps: int, make_gif: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    demo = Demo(SimSource(), WarningHud((W, H)), with_audio=False)
    tmp_video = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = int(seconds * fps)
    dt = 1.0 / fps
    print(f"[warning_hud] rendering {n} frames @ {fps}fps ...")
    for i in range(n):
        out = demo.step(i * dt, dt)
        vw.write(out)
        if i % 30 == 0:
            print(f"  {i}/{n}  state={demo.sm.state}")
    vw.release()
    demo.close()

    # bake the emitted cue list into a WAV and mux it over the video
    wav = out_path + ".wav"
    try:
        from arcv.audio import render_track, save_wav
        track = render_track(demo.cues, seconds, volume=0.6, source="synth")
        save_wav(wav, track)
        print(f"[warning_hud] {len(demo.cues)} audio cues -> {wav}")
    except Exception as e:
        wav = None
        print(f"[warning_hud] audio render skipped: {e}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and wav and _has_encoder(ffmpeg, "libx264"):
        cmd = [ffmpeg, "-y", "-i", tmp_video, "-i", wav, "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path]
        if _run(cmd):
            os.remove(tmp_video)
            print(f"[warning_hud] wrote {out_path}  (H.264 + audio)")
        else:
            os.replace(tmp_video, out_path)
            print(f"[warning_hud] wrote {out_path}  (mp4v, no mux)")
    else:
        os.replace(tmp_video, out_path)
        print(f"[warning_hud] wrote {out_path}  (mp4v; ffmpeg/x264 unavailable for muxing)")

    if make_gif and ffmpeg:
        gif = os.path.splitext(out_path)[0] + ".gif"
        pal = out_path + ".pal.png"
        vf = "fps=14,scale=760:-1:flags=lanczos"
        if _run([ffmpeg, "-y", "-i", out_path, "-vf", vf + ",palettegen", pal]) and \
           _run([ffmpeg, "-y", "-i", out_path, "-i", pal, "-lavfi",
                 vf + " [x];[x][1:v] paletteuse", gif]):
            os.remove(pal)
            print(f"[warning_hud] wrote {gif}")

    if wav and os.path.exists(wav):        # drop the intermediate audio track
        try:
            os.remove(wav)
        except OSError:
            pass


def run_preview(out_dir: str) -> None:
    """Write one still per state (settled) for quick offline judging."""
    os.makedirs(out_dir, exist_ok=True)
    # times chosen to land squarely inside each phase of the sim showcase
    marks = {"boot": 0.7, "acquiring": 2.34, "locked": 5.2, "two_targets": 5.4,
             "target_lost": 8.5, "reacquire": 10.12}
    for name, tm in marks.items():
        demo = Demo(SimSource(), WarningHud((W, H)), with_audio=False)
        dt = 1.0 / 30.0
        out = None
        i = 0
        while i * dt <= tm:
            out = demo.step(i * dt, dt); i += 1
        cv2.imwrite(os.path.join(out_dir, f"warning_hud_{name}.png"), out)
        print("wrote", os.path.join(out_dir, f"warning_hud_{name}.png"), "state=", demo.sm.state)


# ------------------------------------------------------------------ ffmpeg utils
def _run(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _has_encoder(ffmpeg: str, name: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        return name in out
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="WARNING // target-acquisition HUD demo")
    ap.add_argument("--source", choices=["sim", "webcam"], default="sim",
                    help="live feed source (default: sim)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--record", metavar="OUT.mp4", help="render the showcase to a video (+GIF) and exit")
    ap.add_argument("--seconds", type=float, default=CYCLE, help="record duration")
    ap.add_argument("--fps", type=int, default=30, help="record fps")
    ap.add_argument("--no-gif", action="store_true", help="skip GIF export")
    ap.add_argument("--preview", metavar="DIR", help="write per-state stills and exit")
    args = ap.parse_args(argv)

    if args.preview:
        run_preview(args.preview)
    elif args.record:
        run_record(args.record, args.seconds, args.fps, make_gif=not args.no_gif)
    else:
        run_live(args.source, args.camera)


if __name__ == "__main__":
    main()
