"""Person masking for the parkour depth input (pure numpy -- no Isaac deps).

Lives in its own module so the masking logic is unit-testable on the host without
booting Isaac Sim (isaac_env.py imports `isaacsim` at module load). isaac_env.py
imports `mask_person_in_parkour_depth` from here and uses it unchanged.

The followed person must be removed from the front depth image before it reaches the
perceptive parkour policy, or the near body reads as terrain and the dog charges it
(the close-range surge). The original fix flat-filled the whole person box to "far"
(clear), which ALSO blanked the step the person stands on -- so at the stair base the
policy went blind to the first riser and tripped. The terrain-preserving fill here
keeps the visible riser while still removing the body.
"""

import math

import numpy as np

# --- Person-mask FOV mapping (RGB/YOLO cam -> parkour depth cam) -------------
# The YOLO person bbox comes from the front RGB stream (add_camera: focal 26,
# aperture 36 x 20.25 -> ~69 deg hFOV / ~42.6 deg vFOV). The parkour policy reads
# the depth cam (add_parkour_depth_camera: focal 18.97, aperture 36 x 36*60/106
# -> 87 deg hFOV / ~56.5 deg vFOV). Both are the SAME co-located D435 (identical
# FRONT_D435_MOUNT + aim), so a bbox maps from RGB-normalized coords to depth
# pixels by center-scaling each axis by tan(FOV/2)_rgb / tan(FOV/2)_depth (the
# depth FOV is wider, so the RGB frame fills the central ~73% of it). CONTRACT:
# if you change either camera's intrinsics, update these to match (and the UDP
# person_bbox datagram in sim/bot/sim_robot_controller.py + isaac_env.py together).
_RGB_TAN_HALF_H = 36.0 / (2.0 * 26.0)                     # ~0.6923
_RGB_TAN_HALF_V = 20.25 / (2.0 * 26.0)                    # ~0.3894
_PK_TAN_HALF_H = 36.0 / (2.0 * 18.97)                     # ~0.9489
_PK_TAN_HALF_V = (36.0 * 60.0 / 106.0) / (2.0 * 18.97)    # ~0.5371
_BBOX_TO_DEPTH_SCALE_H = _RGB_TAN_HALF_H / _PK_TAN_HALF_H  # ~0.7296
_BBOX_TO_DEPTH_SCALE_V = _RGB_TAN_HALF_V / _PK_TAN_HALF_V  # ~0.7250
# Value written into masked pixels by the legacy far-fill. The depth preprocessing
# clips to far_clip, so any value >= far_clip reads as "max range / clear".
_PARKOUR_DEPTH_FAR_FILL = 1.0e5
# Depth at/above this (metres) is sky / no-return / already-cleared, not real
# terrain (Isaac returns inf or 0 for no-hit; the far-fill writes 1e5).
_PARKOUR_DEPTH_SKY_M = 50.0
# A person pixel reads at least this many metres NEARER than the terrain behind it,
# so the body is separable from the step/floor visible around the legs.
_PARKOUR_MASK_BODY_MARGIN_M = 0.10


def _terrain_reference_depth(d, cx1, cy1, cx2, cy2):
    """Robust depth (m) of the terrain immediately below/beside a person box.

    Samples the band just below the box (the floor/step in front of the feet) plus
    thin left/right side strips (terrain beside the body), and returns the median of
    the valid (finite, in-range, non-sky) depths. Returns None when no real terrain
    is visible around the box -- e.g. the person fills the lower frame -- so the
    caller can fall back to the legacy far-fill for that genuinely-occluded case.
    """
    h, w = d.shape
    box_h = max(1, cy2 - cy1)
    box_w = max(1, cx2 - cx1)
    band = max(2, int(round(0.25 * box_h)))
    side = max(2, int(round(0.15 * box_w)))
    samples = []
    # Floor / steps directly in front of the feet (rows just below the box).
    by2 = min(h, cy2 + band)
    if by2 > cy2:
        samples.append(d[cy2:by2, cx1:cx2].reshape(-1))
    # Terrain to the left / right of the body within the box's vertical span.
    lx1 = max(0, cx1 - side)
    if cx1 > lx1:
        samples.append(d[cy1:cy2, lx1:cx1].reshape(-1))
    rx2 = min(w, cx2 + side)
    if rx2 > cx2:
        samples.append(d[cy1:cy2, cx2:rx2].reshape(-1))
    if not samples:
        return None
    vals = np.concatenate(samples)
    vals = vals[np.isfinite(vals) & (vals > 1e-4) & (vals < _PARKOUR_DEPTH_SKY_M)]
    if vals.size < 3:
        return None
    return float(np.median(vals))


def mask_person_in_parkour_depth(depth_hw, person_bbox_norm, dilate_frac: float = 0.06,
                                 fill_mode: str = "terrain"):
    """Remove the followed person from the parkour depth frame WITHOUT blinding the
    policy to the terrain the person stands on.

    person_bbox_norm = [x1, y1, x2, y2] in [0, 1] of the RGB (YOLO) frame, mapped to
    depth pixels via the co-located-D435 FOV center-scaling above.

    fill_mode:
      'terrain' (default) -- terrain-preserving inpaint. Estimate the depth of the
        terrain immediately below/beside the box, KEEP the real terrain pixels still
        visible around the limbs, and overwrite only the protruding body (pixels
        markedly nearer than that terrain, plus holes) with the terrain depth. The
        perceptive policy still sees the step/riser the person occludes instead of a
        false "clear floor", which is what stops the dog going blind to the first
        step at the stair base. Removes the close-range body surge just as well, since
        the body no longer reads as a near vertical surface.
      'far' -- legacy flat far-fill: push the whole box to max range ("clear"). Kept
        for A/B (it reproduces the stair-base fall). 'terrain' also falls back to this
        when no real terrain is visible around the box (person fills the lower frame).

    Returns (out_depth, (cx1,cy1,cx2,cy2), stats); returns (input, None, None) on any
    bad/empty box. Deployable: the real robot runs the same YOLO detector, so this is
    not a sim-only ground-truth cheat -- it uses only real depth.
    """
    try:
        d = np.asarray(depth_hw)
        if d.ndim != 2 or person_bbox_norm is None or len(person_bbox_norm) < 4:
            return depth_hw, None, None
        h, w = d.shape
        x1, y1, x2, y2 = (float(person_bbox_norm[0]), float(person_bbox_norm[1]),
                          float(person_bbox_norm[2]), float(person_bbox_norm[3]))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # Small dilation to catch limb/edge leakage outside the tight box.
        x1 -= dilate_frac
        x2 += dilate_frac
        y1 -= dilate_frac
        y2 += dilate_frac

        def _to_depth_px(u, v):
            ud = 0.5 + (u - 0.5) * _BBOX_TO_DEPTH_SCALE_H
            vd = 0.5 + (v - 0.5) * _BBOX_TO_DEPTH_SCALE_V
            return ud * w, vd * h

        px1, py1 = _to_depth_px(x1, y1)
        px2, py2 = _to_depth_px(x2, y2)
        cx1 = max(0, int(math.floor(min(px1, px2))))
        cx2 = min(w, int(math.ceil(max(px1, px2))))
        cy1 = max(0, int(math.floor(min(py1, py2))))
        cy2 = min(h, int(math.ceil(max(py1, py2))))
        if cx2 <= cx1 or cy2 <= cy1:
            return depth_hw, None, None
        out = d.copy()
        box = out[cy1:cy2, cx1:cx2]                       # view into out
        stats = {"fill_mode": fill_mode, "terrain_ref_m": None,
                 "preserved_terrain_px": 0, "body_px": int(box.size)}

        terrain_ref = None
        if fill_mode == "terrain":
            terrain_ref = _terrain_reference_depth(d, cx1, cy1, cx2, cy2)

        if terrain_ref is None:
            # fill_mode == 'far', or no usable terrain around the box: legacy flat
            # far-fill so the near body never leaks through as terrain (the surge).
            box[:] = float(_PARKOUR_DEPTH_FAR_FILL)
            stats["fill_mode"] = "far" if fill_mode == "far" else "far_fallback"
            stats["body_px"] = int(box.size)
            return out, (cx1, cy1, cx2, cy2), stats

        # Terrain-preserving inpaint: keep terrain pixels visible around the limbs,
        # overwrite only the protruding body (markedly nearer than the terrain) and
        # any holes with the terrain depth.
        valid = np.isfinite(box) & (box > 1e-4) & (box < _PARKOUR_DEPTH_SKY_M)
        body = (~valid) | (box < (terrain_ref - _PARKOUR_MASK_BODY_MARGIN_M))
        box[body] = float(terrain_ref)
        stats["terrain_ref_m"] = round(float(terrain_ref), 3)
        stats["preserved_terrain_px"] = int(np.count_nonzero(~body))
        stats["body_px"] = int(np.count_nonzero(body))
        return out, (cx1, cy1, cx2, cy2), stats
    except Exception:
        return depth_hw, None, None
