import csv
import json
import os
import threading
import time
from typing import Any, Dict, Iterable


class DebugTraceLogger:
    # DEBUG-TRACE REMOVE-ME: Temporary structured JSONL trace logger for sidecar diagnostics.
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


class CsvTraceLogger:
    # DEBUG-TRACE REMOVE-ME: Temporary CSV trace logger for quick matplotlib/pandas plotting.
    def __init__(self, trace_dir: str, filename: str, fieldnames: Iterable[str]) -> None:
        self.enabled = bool(trace_dir)
        self._lock = threading.Lock()
        self._handle = None
        self._writer = None
        self.path = None
        self._fieldnames = [str(name) for name in fieldnames]

        if not self.enabled:
            return

        os.makedirs(trace_dir, exist_ok=True)
        self.path = os.path.join(trace_dir, filename)
        file_exists = os.path.exists(self.path)
        self._handle = open(self.path, "a", encoding="utf-8", newline="", buffering=1)
        self._writer = csv.writer(self._handle)
        if (not file_exists) or os.path.getsize(self.path) == 0:
            self._writer.writerow(self._fieldnames)

    def log_row(self, **fields: Any) -> None:
        if not self.enabled or self._writer is None:
            return

        row = [self._format_value(fields.get(name)) for name in self._fieldnames]
        with self._lock:
            self._writer.writerow(row)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "NaN"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value) if value == value and abs(value) != float("inf") else "NaN"
        return str(value)

    def close(self) -> None:
        if not self.enabled or self._handle is None:
            return
        with self._lock:
            self._handle.close()
            self._handle = None
            self._writer = None
