"""Standalone HUD preview renderer — dev tool, no Isaac / no GPU-sim required.

Renders the on-frame HUD (``core.hud.visualization.draw_frame_overlays`` — the
real runtime compositor: vignette + scanlines + raster depth blit + arcv GPU
overlay) over a *simulated* camera scene with rich synthetic-but-plausible
telemetry, for every tracking state, and writes PNGs.

This is the visual iteration loop for the HUD: run it, look at the PNGs.  It
exercises the exact runtime path, so what you see here is what the recorded
follow videos get.

    python -m core.hud.hud_preview [OUT_DIR]

Only pulls in a real GL context via :mod:`core.hud.gl_hud` (verified stand-alone
on a dev GPU).  Every value drawn is the same field the live loop feeds; this
module only *fabricates* those fields so the layout can be judged offline.
"""
from __future__ import annotations

import base64
import os
import sys
import zlib
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.hud.visualization import draw_frame_overlays  # noqa: E402


# ---------------------------------------------------------------- synthetic scene
def _sim_camera(W: int, H: int, person_norm, stairs: bool) -> np.ndarray:
    """A plausible mid-tone corridor: floor/wall gradient, a person silhouette at
    ``person_norm`` (or none), and stair treads lower-centre.  Kept mid-tone so
    HUD legibility judgements transfer to real footage."""
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    # wall (upper) brighter, floor (lower) darker; slight vertical seam
    base = 0.30 + 0.22 * (1.0 - yy) + 0.05 * np.sin(xx * 6.28) * (1.0 - yy)
    img = np.dstack([base * 0.92, base * 0.95, base]).astype(np.float32)  # cool grey

    if stairs:
        for i in range(5):
            fy = 0.62 + i * 0.055
            y0 = int(fy * H)
            y1 = int((fy + 0.028) * H)
            x0, x1 = int(0.40 * W), int(0.66 * W)
            img[y0:y1, x0:x1, :] *= 0.72          # tread shadow
            img[max(0, y0 - 2):y0, x0:x1, :] = 0.62   # nosing highlight

    if person_norm is not None:
        bx0, by0, bx1, by1 = (person_norm[0] * W, person_norm[1] * H,
                              person_norm[2] * W, person_norm[3] * H)
        cx = int(0.5 * (bx0 + bx1))
        # torso + head silhouette, slightly warm (person), plus an O2 backpack blob
        cv2.rectangle(img, (int(bx0), int(by0 + 0.18 * (by1 - by0))), (int(bx1), int(by1)),
                      (0.40, 0.44, 0.52), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, int(by0 + 0.10 * (by1 - by0))), int(0.05 * (by1 - by0) + 6),
                   (0.46, 0.50, 0.58), -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(bx0 - 0.03 * W), int(by0 + 0.30 * (by1 - by0))),
                      (int(bx0 + 0.02 * W), int(by0 + 0.62 * (by1 - by0))),
                      (0.22, 0.30, 0.42), -1, cv2.LINE_AA)   # tank
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _lidar_profile(front_clear_m: float = 2.4, n: int = 180) -> Dict:
    """Encode a plausible XT16 polar profile (forward opening, a near obstacle on
    one flank, the tracked person straight ahead)."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = np.full(n, 5.5, dtype=np.float32)
    # walls left/right, person ahead, a close chair on the right flank
    r[(ang > np.radians(20)) & (ang < np.radians(70))] = 1.1
    r[(ang > np.radians(300)) & (ang < np.radians(340))] = 2.0
    fwd = (ang < np.radians(12)) | (ang > np.radians(348))
    r[fwd] = front_clear_m
    ranges_mm = np.clip(r * 1000.0, 0, 65000).astype(np.uint16)
    blob = base64.b64encode(zlib.compress(ranges_mm.tobytes())).decode("ascii")
    return {"ranges_mm": blob, "n_azimuth": n, "azimuth_step_deg": 360.0 / n,
            "view_range_m": 6.0, "hit_count": int((r < 5.4).sum()), "ray_count": n}


def _depth_img(H: int = 240, W: int = 320, stairs: bool = True) -> np.ndarray:
    """A depth frame in mm: floor receding + stair step-edges lower-centre."""
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    d = (3200.0 - 2200.0 * yy) * np.ones((H, W), dtype=np.float32)   # near at bottom
    if stairs:
        for i in range(4):
            y = int((0.45 + i * 0.12) * H)
            d[y:y + int(0.06 * H), int(0.28 * W):int(0.72 * W)] -= 550.0 * (i + 1)
    return np.clip(d, 250.0, 6000.0).astype(np.uint16)


# ---------------------------------------------------------------- telemetry states
def _debug(state: str) -> Tuple[dict, dict, bool, dict]:
    """Return (debug_info, frame_meta, reacquire, kwargs) for a named state."""
    person = [0.44, 0.24, 0.57, 0.82]          # normalised bbox
    lidar = _lidar_profile()
    common = {
        "lidar_profile": lidar, "lidar_bearing_deg": 3.0,
        "depth_img": _depth_img(), "stairs_bbox": [512, 300, 768, 560],
        "distance_source": "lidar_depth_fused", "distance_disagreement": False,
    }
    robot = {"roll_deg": 2.1, "pitch_deg": 8.4, "height_m": 0.31,
             "fell": False, "fall_type": "none"}
    loco = {"policy": "pgtt_level17", "mode": "CLIMB", "gait_pattern": "diagonal_trot",
            "foot_clearance_m": 0.06, "commanded_speed_mps": 0.30, "leg_commands": {}}
    meta = {"success": True, "swing_legs": ["FL", "RR"]}
    kw = dict(camera_mode="single", trans_x_cmd=0.30, rotation_cmd=-0.12,
              proc_fps=6.7, view_fps=5.9)

    if state == "locked":
        d = {**common, "center_x": 646, "person_bbox_norm": person,
             "matched_visual_lock": True, "depth_distance_m": 0.52,
             "rotation_error_deg": -6.4, "stairs_detected": True, "stairs_conf": 0.88,
             "stairs_depth_m": 0.42, "stair_demo": {"robot": robot, "locomotion": loco}}
        return d, meta, False, kw
    if state == "tracking":
        loco2 = {**loco, "mode": "WALK", "gait_pattern": "diagonal_trot"}
        d = {**common, "center_x": 700, "person_bbox_norm": [0.50, 0.30, 0.62, 0.80],
             "matched_visual_lock": False, "depth_distance_m": 1.85,
             "rotation_error_deg": 9.1, "stairs_detected": True, "stairs_conf": 0.55,
             "stairs_depth_m": 1.4, "stair_demo": {"robot": robot, "locomotion": loco2}}
        return d, meta, False, kw
    if state == "searching":
        loco3 = {**loco, "mode": "WALK", "commanded_speed_mps": 0.0}
        d = {**common, "center_x": None, "person_bbox_norm": None,
             "matched_visual_lock": False, "depth_distance_m": None,
             "rotation_error_deg": None, "stairs_detected": True, "stairs_conf": 0.61,
             "stairs_depth_m": 0.9, "distance_source": "none",
             "stair_demo": {"robot": robot, "locomotion": loco3}}
        return d, {"success": True}, True, kw
    if state == "fallen":
        robot_f = {**robot, "fell": True, "fall_type": "collapsed_low",
                   "roll_deg": 41.0, "pitch_deg": -22.0, "height_m": 0.14}
        d = {**common, "center_x": 646, "person_bbox_norm": person,
             "matched_visual_lock": False, "depth_distance_m": 0.6,
             "rotation_error_deg": -2.0, "stairs_detected": True, "stairs_conf": 0.9,
             "stairs_depth_m": 0.3, "distance_disagreement": True,
             "stair_demo": {"robot": robot_f, "locomotion": loco}}
        return d, meta, True, kw
    raise ValueError(state)


def _assemble_time(seconds: float = 6.0) -> None:
    """Shift the HUD clock back so the boot-in stagger has fully assembled — a
    still preview should show the settled HUD, not a mid-assemble frame."""
    import time
    from core.hud import visualization as V
    V._T0 = time.time() - seconds


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    W, H = 1280, 720
    states = ["locked", "tracking", "searching", "fallen"]
    tiles = []
    for st in states:
        _assemble_time()
        debug, meta, reac, kw = _debug(st)
        person = debug.get("person_bbox_norm")
        frame = _sim_camera(W, H, person, bool(debug.get("stairs_detected")))
        draw_frame_overlays(frame, debug, preparation_mode=False, reacquire_active=reac,
                            is_stitched=False, frame_meta=meta, **kw)
        cv2.putText(frame, st.upper(), (18, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, f"hud_{st}.png")
        cv2.imwrite(path, frame)
        tiles.append(frame)
        print("wrote", path)
    # paused / preparation screen
    _assemble_time()
    frame = _sim_camera(W, H, [0.44, 0.24, 0.57, 0.82], True)
    draw_frame_overlays(frame, _debug("locked")[0], preparation_mode=True,
                        reacquire_active=False, camera_mode="single")
    cv2.imwrite(os.path.join(out_dir, "hud_paused.png"), frame)
    print("wrote", os.path.join(out_dir, "hud_paused.png"))
    # 2x2 montage for one-glance review
    row0 = np.hstack([tiles[0], tiles[1]])
    row1 = np.hstack([tiles[2], tiles[3]])
    montage = cv2.resize(np.vstack([row0, row1]), (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "hud_montage.png"), montage)
    print("wrote", os.path.join(out_dir, "hud_montage.png"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "hud_preview_out")
    main(out)
