"""Host-side (no Isaac) tests for the async recorder + record-cadence math.

These cover the pure, host-testable logic added for the sim-side improvements:

  * AsyncRecordingWriter: writes are handed to a background worker, the queue is a
    bounded drop-oldest buffer (never blocks the caller), and release() DRAINS the
    queue + JOINS the worker BEFORE finalising the underlying writer (the moov-atom
    guarantee -- every frame accepted before release is encoded).
  * The --record-every-n-steps write-stride subsampling arithmetic (10-15 fps target).

isaac_env.py is deliberately NOT imported here: importing it by name re-executes the
module and boots a second SimulationApp. recording_writer.py is pure Python (cv2/numpy
imports are deferred), so it is safe to import. The _StepProfiler lives in isaac_env
and is verified by reading + py_compile, not by import.

Run: python -m pytest tests/test_sim_profiler.py
"""

import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))

from recording_writer import AsyncRecordingWriter  # noqa: E402


class _FakeWriter:
    """Stand-in for RecordingWriter: records every frame it is asked to encode.

    Mirrors the public surface AsyncRecordingWriter relies on (started / frames /
    backend / role / write / release). An optional per-write sleep simulates a slow
    encoder so the drop-oldest path can be exercised deterministically.
    """

    def __init__(self, role="test", write_sleep=0.0):
        self.role = role
        self.backend = "fake"
        self.started = False
        self.frames = 0
        self.released = False
        self._write_sleep = float(write_sleep)
        self._lock = threading.Lock()
        self.seen = []

    def write(self, bgr):
        if self._write_sleep:
            time.sleep(self._write_sleep)
        with self._lock:
            self.started = True
            self.frames += 1
            self.seen.append(bgr)

    def release(self):
        self.released = True
        return self.started and self.frames > 0


def test_async_writer_conserves_every_submitted_frame():
    """Frame accounting is exact: encoded + dropped == submitted (nothing vanishes),
    and release() drains + joins so no queued frame is stranded."""
    fake = _FakeWriter()
    aw = AsyncRecordingWriter(fake, maxsize=8)
    for i in range(20):
        aw.write(i)
    ok = aw.release()
    assert ok is True
    assert fake.released is True
    # Drop-oldest may drop some frames of a fast burst into a small queue, but every
    # submitted frame is accounted for (encoded or explicitly dropped) -- none lost.
    assert fake.frames + aw.dropped == 20
    # Whatever was encoded is in submission order (drop-oldest preserves ordering).
    assert fake.seen == sorted(fake.seen)


def test_async_writer_no_loss_when_worker_keeps_up():
    """At the real ~14 fps record cadence (frames spaced out), nothing is dropped."""
    fake = _FakeWriter()
    aw = AsyncRecordingWriter(fake, maxsize=8)
    for i in range(12):
        aw.write(i)
        time.sleep(0.01)  # spaced submits: the fast worker drains between them
    aw.release()
    assert fake.frames == 12
    assert aw.dropped == 0
    assert fake.seen == list(range(12))


def test_async_writer_release_is_idempotent():
    fake = _FakeWriter()
    aw = AsyncRecordingWriter(fake, maxsize=4)
    aw.write(1)
    assert aw.release() is True
    # Second release must not raise or double-finalise.
    assert aw.release() is True
    assert fake.frames == 1


def test_async_writer_release_joins_worker_thread():
    """The worker thread is joined by release() -- no lingering daemon."""
    fake = _FakeWriter()
    aw = AsyncRecordingWriter(fake, maxsize=4)
    aw.write(1)
    aw.release()
    assert aw._thread is None


def test_async_writer_drops_oldest_when_encoder_is_slow():
    """A slow encoder + a full queue drops the OLDEST queued frame; never blocks."""
    fake = _FakeWriter(write_sleep=0.05)  # deliberately slow encode
    aw = AsyncRecordingWriter(fake, maxsize=2)
    t0 = time.perf_counter()
    for i in range(200):
        aw.write(i)  # must return promptly even though the encoder is slow
    submit_elapsed = time.perf_counter() - t0
    # 200 submits must not block for anywhere near 200 * 0.05 s = 10 s.
    assert submit_elapsed < 2.0
    aw.release()
    # Some frames were dropped (encoder could not keep up), so fewer than 200 encoded.
    assert fake.frames < 200
    assert aw.dropped > 0
    # But no more frames were encoded than submitted, and the count is consistent.
    assert fake.frames + aw.dropped == 200


def test_async_writer_write_after_release_is_noop():
    fake = _FakeWriter()
    aw = AsyncRecordingWriter(fake, maxsize=4)
    aw.write(1)
    aw.release()
    aw.write(2)  # ignored: already released
    assert fake.frames == 1


def test_async_writer_pass_through_properties():
    fake = _FakeWriter(role="topdown")
    aw = AsyncRecordingWriter(fake, maxsize=4)
    assert aw.role == "topdown"
    assert aw.started is False
    assert aw.frames == 0
    aw.write(1)
    aw.release()
    assert aw.backend == "fake"
    assert aw.started is True
    assert aw.frames == 1


# ---------------------------------------------------------------------------
# Record write-stride cadence math (--record-every-n-steps).
# ---------------------------------------------------------------------------
def _record_write_fps(physics_hz, render_every, stride):
    """Mirror the isaac_env computation: record_fps = (physics/render_every)/stride."""
    render_rate = physics_hz / max(1, render_every)
    return render_rate / max(1, stride)


def test_record_write_stride_default_lands_in_10_15_fps():
    # Live contract: physics 200 Hz, render_every 7 -> ~28.6 fps render rate.
    # Default stride 2 -> ~14.3 fps write rate (inside the 10-15 fps target).
    fps = _record_write_fps(200, 7, 2)
    assert 10.0 <= fps <= 15.0


def test_record_write_stride_one_is_full_render_rate():
    # Stride 1 = old behaviour: write every render tick (~28.6 fps).
    fps = _record_write_fps(200, 7, 1)
    assert abs(fps - (200 / 7)) < 1e-6


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
