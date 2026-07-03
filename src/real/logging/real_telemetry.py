"""Write a real run into the perf_tracker run-dir layout (current-run vs past-run).

One ``RealTelemetry`` per run: ``start()`` stamps the launch, ``record_fall_diag()``
appends the per-tick physics sample (throttled by the caller), ``finish()`` closes the
run with an outcome. The resulting folder is byte-compatible with
``perf_tracker.update_table.extract_metrics`` so ``extract_real_run`` ingests it through
the same public API the sim uses -- no sim-specific parsing.

Separation of concerns: each run gets its OWN folder (current run); the archive +
leaderboard (past runs) live in perf_tracker/data and are only touched by record_run.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

from real.logging.fall_diag_schema import (
    fall_diag_event, status_setup_event, status_docker_event,
    status_summary_event, stair_demo_report,
)


class RealTelemetry:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self._logs = os.path.join(run_dir, "logs")
        self._debug = os.path.join(run_dir, "debug")
        self._reports = os.path.join(run_dir, "reports")
        for d in (self._logs, self._debug, self._reports):
            os.makedirs(d, exist_ok=True)
        self._status_path = os.path.join(self._logs, "status.jsonl")
        self._isaac_path = os.path.join(self._debug, "isaac_env.jsonl")
        self._frame_timing_path = os.path.join(self._debug, "frame_timing.jsonl")
        self._n_samples = 0
        # Count of fall-diag writes dropped by an IOError/OSError (e.g. full eMMC). The flight
        # recorder must never kill the flight: a write failure is caught + counted, not raised.
        self._write_errors = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self, *, timestamp: str, command: str, locomotion_mode: str = "pgtt") -> None:
        self._append(self._status_path, status_setup_event(timestamp, locomotion_mode=locomotion_mode))
        self._append(self._status_path, status_docker_event(command))

    def record_fall_diag(
        self,
        *,
        pitch_deg: float,
        roll_deg: float,
        policy_cmd: Sequence[float],
        x: Optional[float] = None,
        h: Optional[float] = None,
        action_norm: Optional[float] = None,
    ) -> None:
        # A full eMMC (or any disk/serialization error) must NOT propagate to kill the 50 Hz
        # control node -- the flight recorder must not kill the flight. Catch + count; the
        # first failure is worth surfacing (the caller sees the count via ``write_errors``).
        try:
            self._append(self._isaac_path, fall_diag_event(
                x=x, h=h, pitch_deg=pitch_deg, roll_deg=roll_deg,
                policy_cmd=policy_cmd, action_norm=action_norm,
            ))
            self._n_samples += 1
        except (IOError, OSError, ValueError, TypeError):
            self._write_errors += 1

    @property
    def write_errors(self) -> int:
        """Count of dropped fall-diag/timing writes (e.g. disk full). 0 in the normal case."""
        return self._write_errors

    def record_frame_timing(
        self,
        *,
        stage_ms: Dict[str, float],
        fps: Optional[float] = None,
        tick_dt_ms: Optional[float] = None,
    ) -> None:
        """Append one per-tick timing sample to ``debug/frame_timing.jsonl``.

        Mirrors the sim's ``frame_timing`` trace event (event name + ``data.stage_ms``)
        so ``perf_tracker.extract_metrics`` and the analysis tooling ingest real runs
        with the SAME per-stage-ms evidence format the sim emits. The caller throttles
        this to ~10 Hz; the write is the only I/O here. Cheap + exception-safe: a bad
        value or a full disk must never take down the 50 Hz control loop.
        """
        try:
            data: Dict[str, Any] = {"stage_ms": {k: float(v) for k, v in dict(stage_ms).items()}}
            if fps is not None:
                data["fps"] = float(fps)
            if tick_dt_ms is not None:
                data["tick_dt_ms"] = float(tick_dt_ms)
            self._append(self._frame_timing_path, {"event": "frame_timing", "data": data})
        except (IOError, OSError, ValueError, TypeError):
            # Same policy as record_fall_diag: a full disk / bad value must never take down the
            # 50 Hz loop, but silently swallowing means the black box goes dark on a full eMMC.
            # Count it (surfaced via ``write_errors``) instead of a blanket pass.
            self._write_errors += 1

    def finish(self, *, exit_reason: str = "completed", motion_elapsed_sec: float = 0.0,
               final_x_m: Optional[float] = None) -> None:
        self._write_json(
            os.path.join(self._reports, "stair_demo_report.json"),
            stair_demo_report(exit_reason=exit_reason, motion_elapsed_sec=motion_elapsed_sec,
                              final_x_m=final_x_m),
        )
        self._append(self._status_path, status_summary_event())

    @property
    def n_samples(self) -> int:
        return self._n_samples

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _append(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    @staticmethod
    def _write_json(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
