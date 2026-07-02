"""Host tests for real-hardware telemetry + preflight (no ROS 2 / no robot).

The key test writes a real run via RealTelemetry and feeds it to the ACTUAL
perf_tracker.extract_metrics + classify_run -- proving a real run is ingested through
the sim's parser unchanged (read-only: it does NOT call record_run, so the real
archive is untouched).

Run: python tests/test_real_telemetry.py  (or via pytest)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.logging.fall_diag_schema import fall_diag_event
from real.logging.real_telemetry import RealTelemetry
from real.verification.preflight import check_crc_roundtrip, check_joint_limits

try:
    from perf_tracker.update_table import extract_metrics, classify_run, CATEGORY_REAL
    _HAVE_PERF = True
except Exception:
    _HAVE_PERF = False


def test_fall_diag_event_shape():
    ev = fall_diag_event(x=1.2, h=0.32, pitch_deg=3.0, roll_deg=-1.0,
                         policy_cmd=[0.4, 0.0, 0.1], action_norm=2.1)
    assert ev["event"]["action"] == "fall_diag"
    sim = ev["sim"]
    assert sim["x"] == 1.2 and sim["h"] == 0.32
    assert sim["pitch"] == 3.0 and sim["roll"] == -1.0 and sim["action_norm"] == 2.1
    # x/h may be None on real hardware
    assert fall_diag_event(x=None, h=None, pitch_deg=0, roll_deg=0,
                           policy_cmd=[0, 0, 0], action_norm=None)["sim"]["x"] is None


def test_real_run_roundtrips_to_perf_tracker():
    if not _HAVE_PERF:
        return  # perf_tracker (or matplotlib) unavailable here -> skip the integration leg
    run_dir = tempfile.mkdtemp(prefix="run_real_test_")
    try:
        tel = RealTelemetry(run_dir)
        tel.start(timestamp="2026-06-21T12:00:00", command="real/main.py --ros2 --follow")
        for i in range(60):                       # >= MIN_REAL_FALL_DIAG_STEPS (50)
            tel.record_fall_diag(pitch_deg=2.0 + 0.01 * i, roll_deg=-1.0,
                                 policy_cmd=[0.4, 0.0, 0.0], x=0.01 * i, h=0.31,
                                 action_norm=1.5)
        tel.finish(exit_reason="completed", motion_elapsed_sec=1.2, final_x_m=0.59)

        row = extract_metrics(Path(run_dir))
        assert row["fall_diag_steps"] == 60
        assert abs(float(row["final_x_m"]) - 0.59) < 1e-6
        assert row["final_pitch_deg"] is not None
        assert classify_run(row) == CATEGORY_REAL   # 60 steps + completed -> real
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_preflight_pure_checks_pass():
    assert check_crc_roundtrip().ok
    assert check_joint_limits().ok


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
