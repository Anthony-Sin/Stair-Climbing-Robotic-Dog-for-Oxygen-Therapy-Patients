"""Host tests for the real-side observability + safety additions (no ROS 2 / no robot).

Covers the pure, host-importable pieces of tasks A/C/E/F:
  * RealTelemetry.record_frame_timing -> debug/frame_timing.jsonl (task A)
  * preflight CRC self-consistency golden + lidar-extrinsics WARN check (tasks C, F)
  * depth_to_policy.preprocess_parkour returns [58,87] (task E)

MUST NOT import rclpy / unitree_go / real.ros2.low_level_control_node (not importable
on this host). The control node is py_compile-checked only, never imported here.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.logging.real_telemetry import RealTelemetry
from real.perception.depth_to_policy import preprocess_parkour, PARKOUR_H, PARKOUR_W
from real.verification.preflight import (
    check_crc_roundtrip, check_lidar_extrinsics, _CRC_SELFCONSISTENCY_GOLDEN,
)


# ---------------------------------------------------------------- task A: frame_timing
def test_record_frame_timing_writes_jsonl():
    run_dir = tempfile.mkdtemp(prefix="run_frame_timing_test_")
    try:
        tel = RealTelemetry(run_dir)
        tel.record_frame_timing(
            stage_ms={"preprocess": 1.2, "policy": 3.4, "publish": 0.5, "tick_total": 5.6},
            tick_dt_ms=20.1,
        )
        path = os.path.join(run_dir, "debug", "frame_timing.jsonl")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 1
        ev = json.loads(lines[0])
        # Mirrors the sim's frame_timing shape: event name + data.stage_ms.
        assert ev["event"] == "frame_timing"
        sm = ev["data"]["stage_ms"]
        assert sm["preprocess"] == 1.2 and sm["policy"] == 3.4
        assert sm["publish"] == 0.5 and sm["tick_total"] == 5.6
        assert ev["data"]["tick_dt_ms"] == 20.1
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_record_frame_timing_is_exception_safe():
    # A bad stage_ms value must not raise (control loop must never crash on telemetry).
    run_dir = tempfile.mkdtemp(prefix="run_frame_timing_bad_")
    try:
        tel = RealTelemetry(run_dir)
        tel.record_frame_timing(stage_ms={"policy": "not-a-number"})  # type: ignore[arg-type]
        tel.record_frame_timing(stage_ms={"policy": 1.0}, fps=10.0)   # a good one still works
        path = os.path.join(run_dir, "debug", "frame_timing.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        # The bad call was swallowed; the good call wrote one line.
        assert len(lines) == 1
        assert json.loads(lines[0])["data"]["fps"] == 10.0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ------------------------------------------------------------------- task C: CRC golden
def test_crc_selfconsistency_golden_passes():
    r = check_crc_roundtrip()
    assert r.ok, r.detail
    # The pinned self-consistency golden must be what the encoder currently produces.
    assert f"{_CRC_SELFCONSISTENCY_GOLDEN:#010x}" in r.detail


def test_crc_golden_matches_recomputed_value():
    # Recompute the CRC exactly as the check/serialization does and compare to the pin.
    from real.control.lowcmd_builder import build_low_cmd_fields, crc32_core, N_CMD_SLOTS
    f = build_low_cmd_fields([0.1 * i for i in range(12)], 40.0, 0.5)
    words = []
    for i in range(N_CMD_SLOTS):
        words.append(int(f.mode[i]) & 0xFFFFFFFF)
        for v in (f.q[i], f.dq[i], f.kp[i], f.kd[i], f.tau[i]):
            words.append(int(np.float32(v).view(np.uint32)))
    assert crc32_core(words) == _CRC_SELFCONSISTENCY_GOLDEN


# --------------------------------------------------------------- task E: parkour resize
def test_preprocess_parkour_shape():
    depth = np.full((240, 424), 2.0, dtype=np.float32)
    out = preprocess_parkour(depth)
    assert out.shape == (58, 87)
    assert (PARKOUR_H, PARKOUR_W) == (58, 87)


def test_preprocess_parkour_masks_person():
    # A near vertical band (the "person") on a far background must be masked out, so the
    # masked frame differs from the unmasked one (terrain-fill or NaN, either is fine).
    depth = np.full((120, 160), 5.0, dtype=np.float32)
    depth[:, 60:100] = 0.4  # near band == person region
    bbox = [0.375, 0.0, 0.625, 1.0]
    masked = preprocess_parkour(depth, person_bbox=bbox)
    unmasked = preprocess_parkour(depth, person_bbox=None)
    assert masked.shape == (58, 87) and unmasked.shape == (58, 87)
    # NaN-safe comparison: the mask must have changed the frame somewhere.
    a = np.nan_to_num(masked, nan=-1.0)
    b = np.nan_to_num(unmasked, nan=-1.0)
    assert not np.allclose(a, b)


# ----------------------------------------------------- task F: lidar extrinsics WARN
def test_lidar_extrinsics_flags_placeholder_as_warn():
    r = check_lidar_extrinsics()
    # The shipped extrinsics are still the placeholder -> check passes (does NOT fail
    # preflight) but is flagged as a WARN so it can't silently ship.
    assert r.ok
    assert r.warn
    assert "PLACEHOLDER" in r.detail or "MEASURE" in r.detail


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
