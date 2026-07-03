import json
import os
import threading
import time
from typing import Any, Dict, Optional


# Bump when the envelope shape changes so downstream readers can branch on it.
TRACE_SCHEMA_VERSION = 1


def _json_trace_default(obj: Any) -> str:
    """Fallback for json.dumps so the diagnostic trace never crashes the control
    loop on a non-JSON value (e.g. debug_info["depth_img"] is a SimDepthFrame /
    numpy array, not a scalar). Array-like objects are summarized by shape+dtype
    instead of dumping their data; everything else is stringified."""
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return f"<{type(obj).__name__} shape={tuple(shape)} dtype={getattr(obj, 'dtype', '?')}>"
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__}>"


class DebugTraceLogger:
    """Structured JSONL trace logger (one event per line).

    Used by the controller and the ROS2 sidecars for stall/latency
    diagnostics. Writing is gated on a non-empty ``trace_dir``.

    Every record is wrapped in a VERSIONED ENVELOPE so a trace file is
    self-describing and correlatable across processes/runs::

        {"schema": 1, "run_id": <str>, "frame": <int|None>, "ts": <wall>,
         "ts_mono": <perf_counter>, "source": <str>, "event": <str>,
         "seq": <int>, "data": {...}}

    ``run_id`` is minted once from the ``FOLLOW_RUN_ID`` env var (or generated and
    logged if unset) so every process in a run shares one id.
    """

    def __init__(
        self,
        trace_dir: str,
        filename: str,
        source: str,
        run_id: Optional[str] = None,
        warn_every: int = 500,
    ) -> None:
        self.source = str(source)
        self.enabled = bool(trace_dir)
        self.run_id = str(run_id) if run_id is not None else _resolve_run_id()
        self._lock = threading.Lock()
        self._seq = 0
        self._handle = None
        self.path = None

        # Counted-warning state for write failures (a full disk must not silently
        # black out the trace, nor spam the log every frame).
        self._warn_every = max(1, int(warn_every))
        self._write_errors = 0
        self._first_error_reported = False

        if not self.enabled:
            return

        os.makedirs(trace_dir, exist_ok=True)
        self.path = os.path.join(trace_dir, filename)
        self._handle = open(self.path, "a", encoding="utf-8", buffering=1)
        self.log("trace_logger_started", trace_path=self.path, run_id=self.run_id)

    def log(self, event: str, *, frame: Optional[int] = None, **fields: Any) -> None:
        if not self.enabled or self._handle is None:
            return

        payload: Dict[str, Any] = {
            "schema": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "frame": None if frame is None else int(frame),
            "ts": time.time(),           # wall clock (correlate across machines)
            "ts_mono": time.perf_counter(),  # monotonic (durations, no clock steps)
            "source": self.source,
            "event": str(event),
            "seq": self._seq,
            "data": fields,
        }
        self._seq += 1

        # The trace must never take down the controller: serialize with a fallback
        # for non-JSON values, and account (never swallow silently) any write error.
        try:
            line = json.dumps(
                payload, separators=(",", ":"), sort_keys=True,
                default=_json_trace_default,
            )
            with self._lock:
                self._handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 -- must not propagate into the loop
            self._note_write_error(exc)

    def _note_write_error(self, exc: BaseException) -> None:
        """Counted-warning for trace write/serialize failures.

        Reports the FIRST failure and then every Nth, keeping a running dropped
        count, so a failing disk is VISIBLE without spamming the log every frame.
        Never raises (a telemetry failure must not reach the control loop).
        """
        self._write_errors += 1
        should_warn = (not self._first_error_reported) or (self._write_errors % self._warn_every == 0)
        if not should_warn:
            return
        self._first_error_reported = True
        try:
            # Best-effort stderr notice; deliberately NOT the structured logger to
            # avoid re-entrancy / an error path that itself writes to disk.
            import sys
            print(
                f"[DebugTraceLogger] trace write failed "
                f"(dropped={self._write_errors}, source={self.source}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        except Exception:
            pass

    def close(self) -> None:
        if not self.enabled or self._handle is None:
            return
        self.log("trace_logger_stopped", write_errors=int(self._write_errors))
        with self._lock:
            self._handle.close()
            self._handle = None


def _resolve_run_id() -> str:
    """Read the shared run id from ``FOLLOW_RUN_ID`` or mint one.

    All processes in a run should share the id; the controller sets/propagates it.
    A generated id is time+pid based so it is unique and sortable.
    """
    rid = os.environ.get("FOLLOW_RUN_ID", "").strip()
    if rid:
        return rid
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
