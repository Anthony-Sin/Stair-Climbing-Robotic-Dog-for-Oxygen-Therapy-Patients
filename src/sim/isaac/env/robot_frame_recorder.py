"""Opt-in per-physics-step robot-state recorder (SIM_ROBOT_FRAMES=1).

Writes ``robot_frames.jsonl`` next to ``debug/isaac_env.jsonl`` for the run: one
JSON header line (schema + dof_names + the active StairSpec), then one JSON
"frame" line per physics step at which the recorder is fed. A downstream GLB
baker parses this file, so the field names/shapes are a fixed contract -- see
the ``write_header``/``write_frame`` docstrings.

Stdlib-only (json/os), no Isaac/omni imports, so this module has zero cost to
import when the feature is off. Every public method no-ops immediately when
``enabled`` is False, and any per-step failure disables the recorder for the
rest of the run (logged once) rather than raising into the sim loop.
"""
import json
import logging
import os
from typing import Any, Dict, Optional


class _RobotFrameRecorder:
    """Buffers and flushes one JSONL line per physics step to <debug_dir>/robot_frames.jsonl.

    ``enabled`` is decided ONCE by the caller (from ``SIM_ROBOT_FRAMES``) and passed
    in; this class does not read the environment itself, so its behavior is fully
    determined by the constructor argument (easy to unit-test / reason about).
    """

    FLUSH_EVERY = 120

    def __init__(self, enabled: bool, debug_dir: str, *, logger: Optional[logging.Logger] = None,
                 log_event_fn=None) -> None:
        self.enabled = bool(enabled)
        self._logger = logger
        self._log_event = log_event_fn
        self._path = os.path.join(debug_dir, "robot_frames.jsonl") if debug_dir else ""
        self._fh = None
        self._lines_since_flush = 0
        self._disabled_reason: Optional[str] = None
        if self.enabled and not self._path:
            # No debug dir resolved (e.g. --log-dir not set this run) -- nothing to
            # write to; disable quietly rather than raising later on first write.
            self.enabled = False

    def note_failure(self, action: str, message: str, **fields: Any) -> None:
        """Disable the recorder and log ONCE. Public so a caller assembling frame data
        OUTSIDE write_frame (e.g. reading go2.get_world_pose() before calling it) can
        route its own failures through the same disable-and-log-once path."""
        if self._disabled_reason is not None:
            return
        self._disabled_reason = action
        self.enabled = False
        if self._log_event is not None:
            try:
                self._log_event(self._logger, logging.WARNING, action, message, **fields)
            except Exception:
                pass

    def write_header(self, *, dof_names, stair_spec: Dict[str, Any]) -> None:
        """Open the file and write the schema/dof_names/stair_spec header line.

        Call once, after ``dof_names`` and the active ``StairSpec`` are both
        resolved for this episode, and before any ``write_frame`` call.
        """
        if not self.enabled:
            return
        try:
            self._fh = open(self._path, "w", encoding="utf-8")
            header = {
                "type": "header",
                "schema": 1,
                "dof_names": [str(n) for n in dof_names],
                "stair_spec": stair_spec,
                "units": "m",
                "up_axis": "Z",
                "quat_order": "wxyz",
            }
            self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")
            self._fh.flush()
        except Exception as exc:
            self.note_failure(
                "robot_frame_recorder_header_failed",
                "SIM_ROBOT_FRAMES recorder could not open/write robot_frames.jsonl; disabling for this run",
                path=self._path, error=str(exc),
            )

    def write_frame(
        self,
        *,
        t: float,
        step: int,
        base_pos,
        base_quat_wxyz,
        dof_pos,
        handoff_state: Optional[str],
        stair_phase: Optional[str],
        stairs_action_active: bool,
        patient: Optional[Dict[str, Any]],
    ) -> None:
        """Append one "frame" line. No-op (cheaply) once disabled/off."""
        if not self.enabled or self._fh is None:
            return
        try:
            frame = {
                "type": "frame",
                "t": round(float(t), 5),
                "step": int(step),
                "base_pos": [round(float(v), 5) for v in base_pos],
                "base_quat_wxyz": [round(float(v), 5) for v in base_quat_wxyz],
                "dof_pos": [round(float(v), 5) for v in dof_pos],
                "handoff_state": handoff_state,
                "stair_phase": stair_phase,
                "stairs_action_active": bool(stairs_action_active),
                "patient": patient,
            }
            self._fh.write(json.dumps(frame, separators=(",", ":")) + "\n")
            self._lines_since_flush += 1
            if self._lines_since_flush >= self.FLUSH_EVERY:
                self._fh.flush()
                self._lines_since_flush = 0
        except Exception as exc:
            self.note_failure(
                "robot_frame_recorder_write_failed",
                "SIM_ROBOT_FRAMES recorder write raised; disabling for the rest of this run",
                path=self._path, error=str(exc),
            )

    def close(self) -> None:
        """Flush and close the file handle. Safe to call multiple times / when off."""
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        finally:
            self._fh = None
