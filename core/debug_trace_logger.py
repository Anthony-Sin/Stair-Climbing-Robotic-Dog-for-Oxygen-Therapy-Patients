import json
import os
import threading
import time
from typing import Any, Dict


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
    # DEBUG-TRACE REMOVE-ME: Temporary structured JSONL trace logger for stall debugging.
    def __init__(self, trace_dir: str, filename: str, source: str) -> None:
        self.source = str(source)
        self.enabled = bool(trace_dir)
        self._lock = threading.Lock()
        self._seq = 0
        self._handle = None
        self.path = None

        if not self.enabled:
            return

        os.makedirs(trace_dir, exist_ok=True)
        self.path = os.path.join(trace_dir, filename)
        self._handle = open(self.path, "a", encoding="utf-8", buffering=1)
        self.log("trace_logger_started", trace_path=self.path)

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._handle is None:
            return

        payload: Dict[str, Any] = {
            "ts_unix": time.time(),
            "ts_monotonic": time.monotonic(),
            "source": self.source,
            "event": str(event),
            "seq": self._seq,
            "data": fields,
        }
        self._seq += 1

        # The trace must never take down the controller: serialize with a fallback
        # for non-JSON values, and swallow any remaining write/serialize error.
        try:
            line = json.dumps(
                payload, separators=(",", ":"), sort_keys=True,
                default=_json_trace_default,
            )
            with self._lock:
                self._handle.write(line + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if not self.enabled or self._handle is None:
            return
        self.log("trace_logger_stopped")
        with self._lock:
            self._handle.close()
            self._handle = None
