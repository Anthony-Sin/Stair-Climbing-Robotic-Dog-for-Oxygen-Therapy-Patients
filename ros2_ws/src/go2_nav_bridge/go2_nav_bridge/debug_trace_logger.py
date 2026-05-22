import json
import os
import threading
import time
from typing import Any, Dict


class DebugTraceLogger:
    # DEBUG-TRACE REMOVE-ME: Temporary structured JSONL trace logger for bridge diagnostics.
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

        line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._handle.write(line + "\n")

    def close(self) -> None:
        if not self.enabled or self._handle is None:
            return
        self.log("trace_logger_stopped")
        with self._lock:
            self._handle.close()
            self._handle = None
