"""Mocap-clip playback: sample baked per-bone rotation tracks by gait phase.

This is the realistic-animation counterpart to the analytic ``gait`` module. Where
``gait`` synthesises 13 single-axis joint angles, a clip carries the FULL per-bone
rotation of a real (motion-captured / hand-keyed) walk or stair-climb, so the
playback looks like a person rather than a 13-DOF mechanism.

Design split (so this stays host-testable, no Isaac/USD import):

  * ``ClipTracks`` -- a plain container of one clip's rotation samples, already
    mapped onto the target skeleton's joint order. The USD-side extractor in
    ``rig`` builds these (it reads the ``SkelAnimation`` prim); this module only
    consumes the plain numbers.
  * ``ClipPlayer`` -- holds one clip per ``AnimStyle`` and samples the active one at
    a gait ``phase`` in [0, 1). The phase is the SAME distance-synced phase the
    analytic gait uses (one cycle per stride travelled), so footfall cadence tracks
    real travel and the feet do not skate -- the clip is just a richer pose source
    for that phase.

Rotations are quaternions as plain ``(w, x, y, z)`` tuples. Interpolation between
adjacent samples is sign-corrected normalised-lerp (nlerp): cheap, robust, and
visually indistinguishable from slerp for the dense sampling mocap clips carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .types import AnimStyle

Quat = Tuple[float, float, float, float]  # (w, x, y, z)


def _normalize(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def nlerp(a: Quat, b: Quat, t: float) -> Quat:
    """Sign-corrected normalised lerp from ``a`` to ``b`` by ``t`` in [0, 1]."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    # Take the shorter arc: a quaternion and its negation are the same rotation.
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
    w = a[0] + (b[0] - a[0]) * t
    x = a[1] + (b[1] - a[1]) * t
    y = a[2] + (b[2] - a[2]) * t
    z = a[3] + (b[3] - a[3]) * t
    return _normalize((w, x, y, z))


@dataclass
class ClipTracks:
    """One baked animation clip, resampled onto a fixed skeleton joint order.

    ``frames[k]`` is the full per-joint local rotation at phase ``phases[k]``; every
    frame has exactly ``joint_count`` quaternions, in the SAME order as the target
    skeleton's joints (the USD extractor fills missing joints with their rest/standing
    rotation, so a frame is always complete). ``phases`` is sorted, in [0, 1), and the
    clip is treated as seamlessly looping (phase wraps modulo 1).
    """

    joint_count: int
    phases: List[float]
    frames: List[List[Quat]]
    name: str = ""

    @property
    def valid(self) -> bool:
        return (
            self.joint_count > 0
            and len(self.phases) >= 2
            and len(self.frames) == len(self.phases)
            and all(len(f) == self.joint_count for f in self.frames)
        )

    def sample(self, phase: float) -> List[Quat]:
        """Full per-joint rotation at ``phase`` (looped), nlerp between neighbours."""
        p = phase % 1.0
        ph = self.phases
        n = len(ph)
        # Locate the segment [ph[i], ph[i+1]) containing p; wrap the last->first.
        if p < ph[0] or p >= ph[-1]:
            # Wrap segment from the last sample back to the first (across the seam).
            lo, hi = n - 1, 0
            span = (1.0 - ph[-1]) + ph[0]
            local = ((p - ph[-1]) % 1.0) / span if span > 1e-9 else 0.0
        else:
            # Binary-ish linear scan (clips are short; phases are sorted).
            lo = 0
            for i in range(n - 1):
                if ph[i] <= p < ph[i + 1]:
                    lo = i
                    break
            hi = lo + 1
            span = ph[hi] - ph[lo]
            local = (p - ph[lo]) / span if span > 1e-9 else 0.0
        fa, fb = self.frames[lo], self.frames[hi]
        return [nlerp(fa[j], fb[j], local) for j in range(self.joint_count)]


class ClipPlayer:
    """Holds one ``ClipTracks`` per gait style and samples the requested one.

    The controller asks for a style + phase; the player returns that clip's full
    per-joint rotation, or ``None`` if no clip is registered for the style (the
    controller then falls back to the analytic gait for that style). Keeping the
    fallback per-style is what lets a realistic flat-walk clip coexist with the
    analytic stair gait until a stair-climb clip is also supplied.
    """

    def __init__(self, clips: Optional[Dict[AnimStyle, ClipTracks]] = None) -> None:
        self._clips: Dict[AnimStyle, ClipTracks] = {}
        for style, clip in (clips or {}).items():
            self.register(style, clip)

    def register(self, style: AnimStyle, clip: Optional[ClipTracks]) -> bool:
        if clip is not None and clip.valid:
            self._clips[style] = clip
            return True
        return False

    def has(self, style: AnimStyle) -> bool:
        return style in self._clips

    @property
    def styles(self) -> List[AnimStyle]:
        return list(self._clips.keys())

    def sample(self, style: AnimStyle, phase: float) -> Optional[List[Quat]]:
        clip = self._clips.get(style)
        if clip is None:
            return None
        return clip.sample(phase)


def build_clip_tracks(
    joint_count: int,
    raw_times: Sequence[float],
    raw_frames: Sequence[Sequence[Quat]],
    *,
    window_start: Optional[float] = None,
    window_len: Optional[float] = None,
    name: str = "",
) -> Optional[ClipTracks]:
    """Resample raw (time, full-pose) samples into a phase-normalised looping clip.

    ``raw_times``/``raw_frames`` are the clip's native time samples and full per-joint
    rotations (already in skeleton-joint order). A sub-window ``[window_start,
    window_start + window_len)`` -- ideally ONE L/R gait cycle, from a stable mid-clip
    region -- is extracted and normalised to phase [0, 1). When the window is omitted
    the whole clip is used. Returns ``None`` if there is not enough data.
    """
    if joint_count <= 0 or not raw_times or len(raw_times) != len(raw_frames):
        return None

    pairs = sorted(zip([float(t) for t in raw_times], raw_frames), key=lambda kv: kv[0])
    t0 = pairs[0][0]
    t1 = pairs[-1][0]
    if window_start is not None and window_len and window_len > 0.0:
        lo = float(window_start)
        hi = lo + float(window_len)
        win = [(t, f) for (t, f) in pairs if lo <= t < hi]
        if len(win) >= 2:
            pairs = win
            t0, t1 = lo, hi

    span = t1 - t0
    if span <= 1e-9 or len(pairs) < 2:
        return None

    phases: List[float] = []
    frames: List[List[Quat]] = []
    seen = set()
    for t, f in pairs:
        p = (t - t0) / span
        if p >= 1.0:
            continue  # keep phases strictly in [0, 1); the seam loops back to 0
        key = round(p, 6)
        if key in seen:
            continue
        seen.add(key)
        phases.append(p)
        frames.append([_normalize(tuple(float(c) for c in q)) for q in f])

    clip = ClipTracks(joint_count=joint_count, phases=phases, frames=frames, name=name)
    return clip if clip.valid else None
