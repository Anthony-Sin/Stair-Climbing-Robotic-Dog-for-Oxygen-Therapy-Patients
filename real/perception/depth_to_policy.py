"""RealSense depth -> the 106x60 person-masked frame the stair detector consumes.

Pure numpy (no cv2 dependency): resizes the metric depth to the parkour 106x60 size
with nearest-neighbor (depth must not be interpolated across discontinuities) and
masks the person bbox out so a person standing in front is not miscounted as a stair.
The mask reuses the proven ``mask_person_in_parkour_depth`` if importable, else falls
back to a simple bbox blanking -- the resize/mask is the single depth-preprocess seam.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Reuse the proven person-mask from real/bot; fall back to plain bbox blank so behavior
# degrades gracefully rather than crashing. Use the full package path so this resolves
# correctly regardless of sys.path (bare import only worked when real/bot was on path).
try:  # pragma: no cover - import path depends on deployment
    from real.bot.parkour_depth_mask import mask_person_in_parkour_depth  # type: ignore
except Exception:  # pragma: no cover
    mask_person_in_parkour_depth = None  # type: ignore

POLICY_W, POLICY_H = 106, 60


def resize_nearest(depth: np.ndarray, out_hw: Tuple[int, int] = (POLICY_H, POLICY_W)) -> np.ndarray:
    """Nearest-neighbor resize of an HxW array to out_hw (no cv2)."""
    a = np.asarray(depth, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return np.zeros(out_hw, dtype=np.float32)
    oh, ow = out_hw
    ih, iw = a.shape
    ri = np.clip((np.arange(oh) * ih // oh), 0, ih - 1)
    ci = np.clip((np.arange(ow) * iw // ow), 0, iw - 1)
    return a[np.ix_(ri, ci)].astype(np.float32)


def preprocess(
    depth_m: np.ndarray,
    person_bbox: Optional[Sequence[float]] = None,
    *,
    out_hw: Tuple[int, int] = (POLICY_H, POLICY_W),
) -> np.ndarray:
    """Return the 106x60 metric depth with the person masked out.

    ``person_bbox`` is normalized [x1,y1,x2,y2] in [0,1] (the follow command's bbox);
    None leaves the frame unmasked.
    """
    d = resize_nearest(depth_m, out_hw)
    if person_bbox is None:
        return d
    if mask_person_in_parkour_depth is not None:
        try:
            masked, _, _ = mask_person_in_parkour_depth(d, person_bbox, fill_mode="terrain")
            return np.asarray(masked, dtype=np.float32)
        except Exception:
            pass
    # Fallback: blank the bbox region (set to a far value so it is not read as a riser).
    h, w = d.shape
    x1, y1, x2, y2 = (float(v) for v in person_bbox)
    c0, c1 = int(np.clip(x1, 0, 1) * w), int(np.clip(x2, 0, 1) * w)
    r0, r1 = int(np.clip(y1, 0, 1) * h), int(np.clip(y2, 0, 1) * h)
    out = d.copy()
    if c1 > c0 and r1 > r0:
        out[r0:r1, c0:c1] = np.nan
    return out
