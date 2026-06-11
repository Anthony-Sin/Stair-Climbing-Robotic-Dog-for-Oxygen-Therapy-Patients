import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


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


class _SimJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event": {
                "action": getattr(record, "sim_event", record.getMessage()),
            },
            "labels": {
                "component": getattr(record, "sim_component", record.name),
            },
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


def configure_sim_logger(
    component: str,
    *,
    log_dir: Optional[str] = None,
    reset: bool = True,
    console: bool = True,
) -> logging.Logger:
    """Create a per-run JSONL logger for simulation-only processes."""
    configured_log_dir = log_dir or os.environ.get("SIM_LOG_DIR")
    resolved_log_dir = Path(configured_log_dir).expanduser() if configured_log_dir else _repo_root() / "log"
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{component}.jsonl"

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
        logger.addHandler(console_handler)

    logger.sim_log_path = str(log_path)  # type: ignore[attr-defined]
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    exc_info = fields.pop("exc_info", None)
    component = logger.name.split(".", 1)[-1]
    logger.log(
        level,
        message,
        extra={
            "sim_component": component,
            "sim_event": event,
            "sim_fields": fields,
        },
        exc_info=exc_info,
    )
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
