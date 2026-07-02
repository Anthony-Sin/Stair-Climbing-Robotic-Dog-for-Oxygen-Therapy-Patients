"""Host tests for the CMU stair-climb mocap clip pipeline (biped_anim.mocap_clip).

Pure-Python (binary-FBX parse -> JointPose), so no Isaac/USD is needed.
"""
import math
import os

import pytest

from biped_anim import mocap_clip as mc
from biped_anim.types import JointPose


FBX = os.path.join(os.path.dirname(mc.__file__), "..", "assets", "mocap", "83_27.fbx")


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db else 0.0


def test_resample_linear():
    # Two keys at t=0 and t=100 -> midpoint interpolates halfway.
    out = mc._resample([0, 100], [0.0, 10.0], [0, 25, 50, 100, 200])
    assert out[0] == 0.0
    assert abs(out[1] - 2.5) < 1e-6
    assert abs(out[2] - 5.0) < 1e-6
    assert out[3] == 10.0
    assert out[4] == 10.0  # clamps past the last key


def test_detect_step_cycle_counts_peaks():
    # 4 synthetic knee-flexion peaks over 80 frames -> ~20-frame cycle.
    knee = [50 + 50 * math.sin(2 * math.pi * 4 * i / 80) for i in range(80)]
    start, length = mc._detect_step_cycle({"lShin": {"d|X": knee}}, 80)
    assert 12 <= length <= 30
    assert 0 <= start <= 80 - length


@pytest.mark.skipif(not os.path.exists(FBX), reason="CMU stair FBX not present")
def test_build_stair_clip_is_real_stair_motion():
    clip = mc.build_stair_clip(FBX)
    assert clip is not None
    # One L/R cycle (two stairs), not the whole multi-step clip.
    assert 25 <= clip.win_len <= 90
    assert clip.frames and isinstance(clip.frames[0], JointPose)

    kl, kr = [], []
    for i in range(24):
        jp = clip.sample(i / 24.0)
        kl.append(math.degrees(jp.knee_l))
        kr.append(math.degrees(jp.knee_r))
    # Real stair climbing: a substantial knee bend, and the legs alternate.
    assert (max(kl) - min(kl)) >= 30
    assert _corr(kl, kr) < 0.0


@pytest.mark.skipif(not os.path.exists(FBX), reason="CMU stair FBX not present")
def test_clip_loops_seamlessly():
    clip = mc.build_stair_clip(FBX)
    a = clip.sample(0.0)
    b = clip.sample(1.0)  # phase wraps -> same frame
    assert abs(a.knee_l - b.knee_l) < 1e-6
    assert abs(a.hip_l - b.hip_l) < 1e-6
