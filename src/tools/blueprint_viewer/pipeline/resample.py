"""Segment auto-selection (real-mode) + linear resampling to a fixed frame rate,
retimed to start at t=0. Operates on the frame-dict schema shared by
``synthetic_motion.py`` and the (future) real ``robot_frames.jsonl`` reader, so the
same code path resamples both.

Frame dict keys used here: t, base_pos, base_quat_wxyz, dof_pos, patient (+ passthrough
of handoff_state/stair_phase/stairs_action_active from the nearest source frame, which
are categorical and not meaningfully interpolatable).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from quat_math import Quat, quat_slerp

TARGET_FPS = 30.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    return [_lerp(a[i], b[i], t) for i in range(len(a))]


def _find_bracket(frames: List[dict], t: float) -> Tuple[int, int, float]:
    """Return (i0, i1, frac) such that frames[i0].t <= t <= frames[i1].t and frac is
    the interpolation parameter in [0,1] (or i0==i1, frac=0 at the exact ends)."""
    times = [f["t"] for f in frames]
    if t <= times[0]:
        return 0, 0, 0.0
    if t >= times[-1]:
        n = len(frames) - 1
        return n, n, 0.0
    # Linear scan is fine here (frame counts are in the hundreds-to-low-thousands,
    # called once per output frame at 30 Hz for a <=25s clip -- not a hot loop).
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            frac = 0.0 if span <= 1e-9 else (t - times[i]) / span
            return i, i + 1, frac
    n = len(frames) - 1
    return n, n, 0.0


def _interp_patient(p0: Optional[dict], p1: Optional[dict], frac: float) -> Optional[dict]:
    if p0 is None and p1 is None:
        return None
    if p0 is None:
        return p1
    if p1 is None:
        return p0
    out: Dict[str, object] = {}
    for key in p0:
        if key not in p1:
            continue
        v0, v1 = p0[key], p1[key]
        if key == "yaw_rad":
            # Shortest-path angle interpolation (avoid a +-pi wraparound snap).
            import math
            d = (v1 - v0 + math.pi) % (2 * math.pi) - math.pi
            out[key] = v0 + d * frac
        elif isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)):
            out[key] = _lerp_vec(v0, v1, frac)
        else:
            out[key] = v1 if frac >= 0.5 else v0
    return out


def resample_to_fixed_fps(
    frames: List[dict], *, fps: float = TARGET_FPS, t0: Optional[float] = None, t1: Optional[float] = None,
) -> List[dict]:
    """Linear-resample ``frames`` (sorted by ``t``, arbitrary/irregular native rate) to
    a fixed ``fps``, retimed so the FIRST output frame is at t=0. If t0/t1 are given,
    only that source-timeline window is resampled (t0/t1 in the SOURCE frames' own
    time base, before retiming); otherwise the full frame list's [first.t, last.t]
    span is used.

    Positions: linear. Quaternions (base_quat_wxyz): slerp (shortest path) for
    correctness during resampling, THEN the caller/animation baker independently
    applies fix_quat_key_signs() to the final glTF-bound key sequence for the LINEAR
    sampler (a different, later concern -- glTF playback interpolation, not this
    resampling step). Categorical fields (handoff_state, stair_phase,
    stairs_action_active) are taken from the NEAREST source frame (no interpolation).
    """
    if not frames:
        return []
    src = sorted(frames, key=lambda f: f["t"])
    span_t0 = src[0]["t"] if t0 is None else t0
    span_t1 = src[-1]["t"] if t1 is None else t1
    if span_t1 <= span_t0:
        raise ValueError(f"resample window is empty or inverted: t0={span_t0} t1={span_t1}")

    duration = span_t1 - span_t0
    n_out = max(2, int(round(duration * fps)) + 1)
    out: List[dict] = []
    for k in range(n_out):
        t = span_t0 + k / fps
        if t > span_t1:
            break
        i0, i1, frac = _find_bracket(src, t)
        f0, f1 = src[i0], src[i1]

        base_pos = _lerp_vec(f0["base_pos"], f1["base_pos"], frac)
        q0: Quat = tuple(f0["base_quat_wxyz"])  # type: ignore[assignment]
        q1: Quat = tuple(f1["base_quat_wxyz"])  # type: ignore[assignment]
        base_quat = list(quat_slerp(q0, q1, frac))
        dof_pos = _lerp_vec(f0["dof_pos"], f1["dof_pos"], frac)
        nearest = f0 if frac < 0.5 else f1

        out.append({
            "type": "frame",
            "t": round(t - span_t0, 6),  # retimed to start at 0
            "step": k,
            "base_pos": base_pos,
            "base_quat_wxyz": base_quat,
            "dof_pos": dof_pos,
            "handoff_state": nearest.get("handoff_state"),
            "stair_phase": nearest.get("stair_phase"),
            "stairs_action_active": nearest.get("stairs_action_active"),
            "patient": _interp_patient(f0.get("patient"), f1.get("patient"), frac),
        })
    return out


def select_follow_window(
    frames: List[dict], *, min_duration_s: float = 8.0, max_duration_s: float = 20.0,
    min_speed_mps: float = 0.05,
) -> Tuple[float, float]:
    """Auto-select the LONGEST contiguous stretch with handoff_state=='walk' AND
    stair_phase=='flat_follow' AND base speed > min_speed_mps, per the contract's
    segment auto-selection rule. Returns (t0, t1) in the SOURCE frames' time base.
    Raises ValueError if no window of at least ``min_duration_s`` qualifies.
    """
    src = sorted(frames, key=lambda f: f["t"])
    speeds = _base_speeds(src)

    def qualifies(i: int) -> bool:
        f = src[i]
        return (
            f.get("handoff_state") == "walk"
            and f.get("stair_phase") == "flat_follow"
            and speeds[i] > min_speed_mps
        )

    best: Optional[Tuple[float, float]] = None
    i = 0
    n = len(src)
    while i < n:
        if not qualifies(i):
            i += 1
            continue
        j = i
        while j + 1 < n and qualifies(j + 1):
            j += 1
        t0, t1 = src[i]["t"], src[j]["t"]
        dur = t1 - t0
        if dur >= min_duration_s:
            if dur > max_duration_s:
                t1 = t0 + max_duration_s
                dur = max_duration_s
            if best is None or dur > (best[1] - best[0]):
                best = (t0, t1)
        i = j + 1

    if best is None:
        raise ValueError(
            f"no contiguous flat_follow window >= {min_duration_s}s found "
            f"(handoff_state=='walk' and stair_phase=='flat_follow' and speed > {min_speed_mps} m/s)"
        )
    return best


def select_climb_window(frames: List[dict], *, max_duration_s: float = 25.0) -> Tuple[float, float]:
    """Auto-select climb = first stair_phase=='stair_approach' frame -> last frame in
    {'staircase','top_landing'}. Per the contract. Raises ValueError if no such span exists."""
    src = sorted(frames, key=lambda f: f["t"])
    start_idx = next((i for i, f in enumerate(src) if f.get("stair_phase") == "stair_approach"), None)
    if start_idx is None:
        raise ValueError("no frame with stair_phase=='stair_approach' found")
    end_candidates = [i for i, f in enumerate(src) if f.get("stair_phase") in ("staircase", "top_landing")]
    end_candidates = [i for i in end_candidates if i >= start_idx]
    if not end_candidates:
        raise ValueError("no frame with stair_phase in {'staircase','top_landing'} at/after stair_approach")
    end_idx = end_candidates[-1]
    t0, t1 = src[start_idx]["t"], src[end_idx]["t"]
    if t1 - t0 > max_duration_s:
        t1 = t0 + max_duration_s
    return t0, t1


def _base_speeds(frames: List[dict]) -> List[float]:
    """Per-frame base speed (m/s), central-difference where possible."""
    n = len(frames)
    speeds = [0.0] * n
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        if lo == hi:
            continue
        dt = frames[hi]["t"] - frames[lo]["t"]
        if dt <= 1e-9:
            continue
        p0, p1 = frames[lo]["base_pos"], frames[hi]["base_pos"]
        dist = sum((p1[k] - p0[k]) ** 2 for k in range(3)) ** 0.5
        speeds[i] = dist / dt
    return speeds
