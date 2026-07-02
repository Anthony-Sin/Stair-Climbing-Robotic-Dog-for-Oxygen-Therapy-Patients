"""Unit tests for SimCameraCapture's opt-in sense->act latency delay buffer.

Verifies frames are held until their release time (FIFO), that the size-1 queue
keeps the latest frame, and that the default (latency disabled) path delivers
immediately. __init__ is bypassed so no UDP receiver socket/thread is opened, and
release times are forced rather than slept on, so the test is deterministic.
Run directly (python tests/test_sim_latency.py) or via pytest.
"""
import os
import queue
import sys
from collections import deque

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in (os.path.join("sim", "bot"), "core"):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_camera_capture as scc


def _cap(latency_enabled):
    cap = object.__new__(scc.SimCameraCapture)  # bypass __init__ (no socket/thread)
    cap._frame_queue = queue.Queue(maxsize=1)
    cap._seq_dropped = 0
    cap._latency_enabled = latency_enabled
    cap._latency_sec = 0.05
    cap._latency_jitter_sec = 0.0
    cap._delay_buf = deque()
    return cap


def test_latency_disabled_delivers_immediately():
    cap = _cap(False)
    cap._enqueue_frame("rgb", "depth")
    assert cap._frame_queue.qsize() == 1
    assert cap._frame_queue.get_nowait() == ("rgb", "depth")


def test_latency_holds_then_releases():
    cap = _cap(True)
    cap._enqueue_frame("rgb", "depth")
    # Held: not yet on the delivery queue.
    assert cap._frame_queue.qsize() == 0 and len(cap._delay_buf) == 1
    # Force the release time into the past, then flush.
    rel, bgr, depth = cap._delay_buf[0]
    cap._delay_buf[0] = (rel - 10.0, bgr, depth)
    cap._flush_delay_buf()
    assert cap._frame_queue.qsize() == 1 and len(cap._delay_buf) == 0


def test_latency_not_released_before_due():
    cap = _cap(True)
    cap._enqueue_frame("rgb", "depth")
    cap._flush_delay_buf()  # release time is ~50 ms out -> nothing released yet
    assert cap._frame_queue.qsize() == 0 and len(cap._delay_buf) == 1


def test_size1_queue_keeps_latest():
    cap = _cap(False)
    cap._enqueue_frame("a", "a")
    cap._enqueue_frame("b", "b")
    assert cap._frame_queue.qsize() == 1
    assert cap._frame_queue.get_nowait() == ("b", "b")
    assert cap._seq_dropped == 1


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for _fn in _tests:
        _fn()
        print("PASS", _fn.__name__)
    print(f"ALL {len(_tests)} TESTS PASSED")
