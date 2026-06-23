"""Host (no-Isaac) tests for the patient mocap-clip playback core.

These prove the pure sampling/looping/registration logic of ``biped_anim.clip_player``
WITHOUT booting Isaac or USD: quaternion nlerp, phase-wrapped interpolation across the
loop seam, per-style registration + fallback, and the raw-sample -> phase-normalised
window builder.

Run: python tests/test_clip_player.py  (or via pytest)
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))

from biped_anim.clip_player import (  # noqa: E402
    ClipPlayer,
    ClipTracks,
    build_clip_tracks,
    nlerp,
)
from biped_anim.types import AnimStyle  # noqa: E402

IDENT = (1.0, 0.0, 0.0, 0.0)


def _quat_z(angle: float):
    """Rotation by ``angle`` about +Z, as (w, x, y, z)."""
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _is_unit(q) -> bool:
    return abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-6


def test_nlerp_endpoints_and_unit():
    a = _quat_z(0.0)
    b = _quat_z(1.0)
    assert _is_unit(nlerp(a, b, 0.0))
    assert _is_unit(nlerp(a, b, 0.5))
    assert _is_unit(nlerp(a, b, 1.0))
    # Endpoints are returned (up to sign / normalisation).
    s = nlerp(a, b, 0.0)
    assert abs(s[0] - a[0]) < 1e-6 and abs(s[3] - a[3]) < 1e-6


def test_nlerp_takes_shorter_arc():
    # b is the negation of a's neighbour: nlerp must flip sign and not collapse to ~0.
    a = _quat_z(0.1)
    b = tuple(-c for c in _quat_z(0.2))
    mid = nlerp(a, b, 0.5)
    assert _is_unit(mid)


def _two_joint_clip():
    # joint 0 sweeps 0 -> +0.4 rad about Z over the cycle, joint 1 stays identity.
    phases = [0.0, 0.25, 0.5, 0.75]
    frames = [
        [_quat_z(0.0), IDENT],
        [_quat_z(0.1), IDENT],
        [_quat_z(0.2), IDENT],
        [_quat_z(0.3), IDENT],
    ]
    return ClipTracks(joint_count=2, phases=phases, frames=frames, name="walk")


def test_clip_sample_interpolates_within_segment():
    clip = _two_joint_clip()
    out = clip.sample(0.125)  # halfway between phase 0.0 and 0.25
    assert len(out) == 2
    # joint 0 angle ~ halfway between 0.0 and 0.1 rad -> ~0.05 rad about Z.
    ang = 2.0 * math.atan2(out[0][3], out[0][0])
    assert abs(ang - 0.05) < 1e-3, ang
    # joint 1 untouched.
    assert abs(out[1][0] - 1.0) < 1e-6


def test_clip_sample_wraps_across_seam():
    clip = _two_joint_clip()
    # phase 0.875 is between the last sample (0.75) and the seam back to 0.0.
    out = clip.sample(0.875)
    assert _is_unit(out[0])
    # phase wraps: sample(1.2) == sample(0.2)
    a = clip.sample(0.2)
    b = clip.sample(1.2)
    for qa, qb in zip(a, b):
        assert all(abs(x - y) < 1e-9 for x, y in zip(qa, qb))


def test_player_registration_and_fallback():
    clip = _two_joint_clip()
    player = ClipPlayer({AnimStyle.FLAT_WALK: clip})
    assert player.has(AnimStyle.FLAT_WALK)
    assert not player.has(AnimStyle.STAIR_CLIMB)
    assert player.sample(AnimStyle.FLAT_WALK, 0.1) is not None
    # No clip for stairs -> None, so the controller falls back to the analytic gait.
    assert player.sample(AnimStyle.STAIR_CLIMB, 0.1) is None


def test_player_rejects_degenerate_clip():
    player = ClipPlayer()
    # Too few frames / mismatched joint count -> invalid, not registered.
    bad = ClipTracks(joint_count=2, phases=[0.0], frames=[[IDENT, IDENT]])
    assert not bad.valid
    assert not player.register(AnimStyle.FLAT_WALK, bad)
    assert not player.has(AnimStyle.FLAT_WALK)


def test_build_clip_tracks_window_normalises_to_phase():
    # 11 raw samples at t = 0..10; take the mid window [2, 6) and normalise to [0,1).
    raw_times = list(range(11))
    raw_frames = [[_quat_z(0.05 * t), IDENT] for t in raw_times]
    clip = build_clip_tracks(
        2, raw_times, raw_frames, window_start=2.0, window_len=4.0, name="win"
    )
    assert clip is not None and clip.valid
    assert clip.phases[0] == 0.0
    assert all(0.0 <= p < 1.0 for p in clip.phases)
    # 4 samples expected in [2,6): t=2,3,4,5 -> phases 0, .25, .5, .75
    assert len(clip.phases) == 4
    assert abs(clip.phases[1] - 0.25) < 1e-9


def test_build_clip_tracks_rejects_flat_window():
    raw_times = [0.0, 1.0]
    raw_frames = [[IDENT], [IDENT]]
    # window_len 0 -> degenerate span -> None
    assert build_clip_tracks(1, raw_times, raw_frames, window_start=0.0, window_len=0.0) is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} clip-player tests passed")


if __name__ == "__main__":
    _run_all()
