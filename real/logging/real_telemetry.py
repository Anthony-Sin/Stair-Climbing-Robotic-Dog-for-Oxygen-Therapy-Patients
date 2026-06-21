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
        self._n_samples = 0

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
        self._append(self._isaac_path, fall_diag_event(
            x=x, h=h, pitch_deg=pitch_deg, roll_deg=roll_deg,
            policy_cmd=policy_cmd, action_norm=action_norm,
        ))
        self._n_samples += 1

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
