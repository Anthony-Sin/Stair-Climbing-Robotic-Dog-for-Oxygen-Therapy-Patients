import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import ecs_logging


_CONFIGURED_LOGGERS: Dict[Tuple[str, str, str, str, bool], Tuple[Optional[str], str]] = {}
_BASE_LOGGER_CONFIGURED = False
_ERROR_CONSOLE_CONFIGURED = False


class _StaticECSContextFilter(logging.Filter):
    def __init__(self, service_name: str, event_dataset: str, session_id: str) -> None:
        super().__init__()
        self._service_name = service_name
        self._event_dataset = event_dataset
        self._session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.setdefault("service.name", self._service_name)
        record.__dict__.setdefault("event.dataset", self._event_dataset)
        record.__dict__.setdefault("labels.session_id", self._session_id)
        record.__dict__.setdefault("labels.project", "cable-manipulation")
        return True


def _configure_base_logger() -> logging.Logger:
    global _BASE_LOGGER_CONFIGURED

    logger = logging.getLogger("cable")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not _BASE_LOGGER_CONFIGURED:
        logger.addHandler(logging.NullHandler())
        _BASE_LOGGER_CONFIGURED = True
    return logger


def _ensure_error_console_handler(logger: logging.Logger) -> None:
    global _ERROR_CONSOLE_CONFIGURED

    if _ERROR_CONSOLE_CONFIGURED:
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    _ERROR_CONSOLE_CONFIGURED = True


def setup_ecs_file_logging(
    service_name: str,
    event_dataset: str,
    log_dir: str = "logs",
    file_prefix: str = "ecs",
    enabled: bool = True,
) -> Tuple[Optional[str], str]:
    key = (service_name, event_dataset, os.path.abspath(log_dir), file_prefix, bool(enabled))
    if key in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[key]

    logger = _configure_base_logger()
    _ensure_error_console_handler(logger)

    session_id = uuid.uuid4().hex
    if not enabled:
        _CONFIGURED_LOGGERS[key] = (None, session_id)
        return _CONFIGURED_LOGGERS[key]

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join(log_dir, f"{file_prefix}_{timestamp}.jsonl"))

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ecs_logging.StdlibFormatter())
    handler.addFilter(_StaticECSContextFilter(service_name, event_dataset, session_id))
    logger.addHandler(handler)

    _CONFIGURED_LOGGERS[key] = (log_path, session_id)
    return log_path, session_id


def get_ecs_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"cable.{name}")


def build_ecs_extra(
    component: str,
    action: str,
    cable: Optional[Dict[str, Any]] = None,
    **ecs_fields: Any,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {
        "event.kind": "event",
        "event.category": ["application"],
        "event.action": action,
        "labels.component": component,
    }
    extra.update(ecs_fields)
    if cable is not None:
        extra["cable"] = cable
    return extra
