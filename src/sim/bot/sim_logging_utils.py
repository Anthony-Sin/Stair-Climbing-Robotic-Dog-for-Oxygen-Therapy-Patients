import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Schema version stamped on every persisted stream this module emits (JSONL lines)
# and reused by the run reports / walk_log / frame sidecar so a consumer can key off it.
SCHEMA_VERSION = 1

# SINGLE SOURCE OF TRUTH for the "upright" body-tilt band (degrees). The live isaac_env
# waypoint climb-quality watchdog and the offline analyzer (sim/analysis/analyze_climb.py)
# both import this so they can never drift (the analyzer had hand-mirrored 18 deg vs the
# live 25 deg). Stdlib-only module, so the host-side analyzer imports it without Isaac.
UPRIGHT_TILT_DEG = 25.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _derive_run_id(log_dir: Path) -> Optional[str]:
    """Best-effort run identifier from a run folder path.

    Logs land under <log>/run_sim_<stamp>/debug/, so the run_sim_* (or warm_*)
    ancestor names the run. Returns None for ad-hoc log dirs.
    """
    for part in [log_dir, *log_dir.parents]:
        name = part.name
        if name.startswith("run_sim_") or name.startswith("warm"):
            return name
    return None


class _SimJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        labels: Dict[str, Any] = {
            "component": getattr(record, "sim_component", record.name),
        }
        run_id = getattr(record, "sim_run_id", None)
        if run_id:
            # Stamp every line with the run it belongs to so current-run output can
            # never be confused with a prior run's, even if files are concatenated.
            labels["run_id"] = run_id
        payload: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "@timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event": {
                "action": getattr(record, "sim_event", record.getMessage()),
            },
            "labels": labels,
            "process": {
                "pid": record.process,
                "name": record.processName,
            },
            "thread": {
                "name": record.threadName,
            },
        }
        fields = getattr(record, "sim_fields", None)
        if fields:
            payload["sim"] = _json_safe(fields)
        if record.exc_info:
            payload["error"] = {
                "stack_trace": self.formatException(record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class _SimConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        component = getattr(record, "sim_component", record.name)
        return f"{timestamp} {record.levelname:<7} [{component}] {record.getMessage()}"


# High-rate per-step diagnostic events that belong in the JSONL stream (for log-based
# verification) but FLOOD the terminal -- fall_diag alone prints ~every 15 steps. Drop these
# from the CONSOLE handler only; the file handler still records every one.
_CONSOLE_SUPPRESSED_EVENTS = {"fall_diag"}


class _ConsoleEventFilter(logging.Filter):
    """Console-only filter: drop the high-rate diagnostic events in _CONSOLE_SUPPRESSED_EVENTS
    so they do not spam the terminal. They are still written to the per-run JSONL file."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "sim_event", None) not in _CONSOLE_SUPPRESSED_EVENTS


def configure_sim_logger(
    component: str,
    *,
    log_dir: Optional[str] = None,
    reset: bool = True,
    console: bool = True,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """Create a per-run JSONL logger for simulation-only processes.

    `reset=True` (the default) opens the file in write mode, so each run starts
    from an empty file -- prior-run lines are never appended to. The resolved
    run_id is stamped onto every line as an extra separation guarantee.
    """
    configured_log_dir = log_dir or os.environ.get("SIM_LOG_DIR")
    resolved_log_dir = Path(configured_log_dir).expanduser() if configured_log_dir else _repo_root() / "log"
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{component}.jsonl"

    resolved_run_id = run_id or os.environ.get("SIM_RUN_ID") or _derive_run_id(resolved_log_dir)

    logger = logging.getLogger(f"sim.{component}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_level = logging.DEBUG if os.environ.get("SIM_LOG_DEBUG") == "1" else logging.INFO
    file_handler = logging.FileHandler(log_path, mode="w" if reset else "a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_SimJsonFormatter())
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_SimConsoleFormatter())
        # Keep the per-step fall_diag (and other high-rate diagnostics) OUT of the terminal --
        # they still go to the JSONL file handler above for log-based verification.
        console_handler.addFilter(_ConsoleEventFilter())
        logger.addHandler(console_handler)

    logger.sim_log_path = str(log_path)  # type: ignore[attr-defined]
    logger.sim_run_id = resolved_run_id  # type: ignore[attr-defined]
    return logger


def log_event(
    logger: Optional[logging.Logger],
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    if logger is None:
        return
    exc_info = fields.pop("exc_info", None)
    component = logger.name.split(".", 1)[-1]
    extra: Dict[str, Any] = {
        "sim_component": component,
        "sim_event": event,
        "sim_fields": fields,
    }
    run_id = getattr(logger, "sim_run_id", None)
    if run_id:
        extra["sim_run_id"] = run_id
    logger.log(level, message, extra=extra, exc_info=exc_info)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Recording visibility
#
# Recordings (mp4s) are ALWAYS captured to the run folder for later review. They
# are only *surfaced* -- live preview window, console call-outs -- when the
# operator opts in via SHOW_RECORDINGS. This keeps headless/batch runs quiet
# without ever losing the footage on disk.
# ---------------------------------------------------------------------------

def recordings_visible() -> bool:
    """True when recordings should be displayed/surfaced (SHOW_RECORDINGS truthy).

    Canonical host-side gate. The controller (core/main.py) runs in a separate
    Docker module tree and cannot import this module, so it mirrors the same
    SHOW_RECORDINGS check inline -- keep the two in sync if the contract changes.
    """
    return os.environ.get("SHOW_RECORDINGS", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Scene baseline
#
# A reference snapshot of the environment with NO person/character present,
# logged once per run BEFORE the patient actor is spawned. Every run then has an
# empty-scene reference to diff sensor/physics readings against.
# ---------------------------------------------------------------------------

def log_scene_baseline(
    logger: Optional[logging.Logger],
    *,
    terrain: Optional[str] = None,
    **scene: Any,
) -> None:
    """Log the empty-scene reference baseline (person absent).

    Call immediately BEFORE spawning the person. `person_present` is forced
    False so the baseline is unambiguous regardless of what the caller passes.
    """
    fields = dict(scene)
    fields["person_present"] = False
    if terrain is not None:
        fields["terrain"] = terrain
    log_event(
        logger,
        logging.INFO,
        "scene_baseline",
        "scene baseline captured (no person in scene)",
        **fields,
    )
