"""Centralized ``.env`` loading + typed env-var accessors.

This is the ".env so the run doesn't make mistakes" layer the project asked for:
every secret/config value enters the process through here, with one obvious file
(``fine_tuning/.env``) and clear errors when a required key is missing. Loading runs
BEFORE argparse builds its parser, so ``.env`` values flow into the argparse
``default=os.environ.get(...)`` fallbacks (matching ``core/args_parser.py``), and an
explicit CLI flag still overrides ``.env``.

python-dotenv is used when available; if it is not installed yet (e.g. before
``pip install -r requirements.txt``) we fall back to a tiny built-in parser so
preflight can still run and tell the user what to install.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = HERE / ".env"
EXAMPLE_ENV_PATH = HERE / ".env.example"

_LOADED_FROM: Optional[str] = None


class MissingEnvVar(RuntimeError):
    """Raised when a required environment variable is absent."""


def _minimal_dotenv(path: Path) -> int:
    """Fallback .env parser (KEY=VALUE, ignores blanks/`#` comments). Returns #loaded."""
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip an UNQUOTED inline comment, matching python-dotenv: a value ends at
        # the first " #" (whitespace + hash); a value that is only a comment (blank
        # value followed by a comment) becomes empty. Quoted values keep any '#'.
        # Without this, a line like `KEY=   # note` parsed to the comment text.
        if val[:1] not in ("'", '"'):
            hashpos = val.find(" #")
            if hashpos != -1:
                val = val[:hashpos].rstrip()
            if val.startswith("#"):
                val = ""
        val = val.strip('"').strip("'")
        if key and key not in os.environ:  # real env wins over .env, like dotenv
            os.environ[key] = val
            n += 1
    return n


def load_env(path: Optional[os.PathLike] = None, *, override: bool = False) -> Optional[str]:
    """Load ``fine_tuning/.env`` into ``os.environ`` once. Returns the path used, or None.

    A missing ``.env`` is NOT an error here -- many values have sane defaults and the
    cloud integrations are opt-in. Required-key enforcement happens in ``require()``.
    """
    global _LOADED_FROM
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.exists():
        _LOADED_FROM = None
        return None
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(env_path), override=override)
    except Exception:
        _minimal_dotenv(env_path)
    _LOADED_FROM = str(env_path)
    return _LOADED_FROM


def loaded_from() -> Optional[str]:
    return _LOADED_FROM


def get_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    return default if val is None or val == "" else val


def get_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def get_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def get_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def require(key: str) -> str:
    """Return a required env var, or raise a clear, actionable error."""
    val = os.environ.get(key)
    if val is None or val == "":
        raise MissingEnvVar(
            f"Required environment variable {key!r} is not set. "
            f"Add it to {DEFAULT_ENV_PATH} (see {EXAMPLE_ENV_PATH} for the template)."
        )
    return val
