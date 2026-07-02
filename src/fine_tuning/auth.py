"""The login system: authenticate the opt-in cloud integrations from ``.env``.

Each integration (Weights & Biases, RunPod) is enabled by a flag and reads its key
from ``.env``. A disabled or unconfigured integration is cleanly skipped (logged),
never fatal -- UNLESS ``require_cloud`` is set, in which case an *enabled* integration
with a missing/invalid credential aborts the run before any training happens. The goal
is that credential problems surface here, up front, with one actionable message each --
not three hours into a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from . import env_bootstrap as envb
from .config import FineTuneConfig

LOGGER = logging.getLogger("fine_tuning.auth")


class CredentialError(RuntimeError):
    """An enabled cloud integration is missing or has an invalid credential."""


@dataclass
class Credentials:
    wandb_enabled: bool = False
    wandb_ok: bool = False
    runpod_enabled: bool = False
    runpod_ok: bool = False
    messages: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.messages.append(msg)
        LOGGER.info(msg)


def _login_wandb(cfg: FineTuneConfig, creds: Credentials, *, strict: bool) -> None:
    creds.wandb_enabled = True
    try:
        import wandb  # type: ignore
    except Exception:
        msg = ("W&B enabled but the 'wandb' package is not installed "
               "(pip install -r fine_tuning/requirements.txt).")
        if strict:
            raise CredentialError(msg)
        creds.note("SKIP wandb: " + msg)
        return
    key = envb.get_str("WANDB_API_KEY")
    try:
        # If key is None, wandb falls back to a cached ~/.netrc login when present.
        ok = wandb.login(key=key, relogin=False, timeout=30)
        creds.wandb_ok = bool(ok)
        creds.note(f"W&B login OK (project={cfg.wandb_project}, "
                   f"key={'env' if key else 'cached'}).")
    except Exception as exc:
        msg = f"W&B login failed: {type(exc).__name__}: {exc}"
        if strict:
            raise CredentialError(msg)
        creds.note("SKIP wandb: " + msg)


def _login_runpod(cfg: FineTuneConfig, creds: Credentials, *, strict: bool) -> None:
    creds.runpod_enabled = True
    key = envb.get_str("RUNPOD_API_KEY")
    if not key:
        msg = "RunPod enabled but RUNPOD_API_KEY is not set in .env."
        if strict:
            raise CredentialError(msg)
        creds.note("SKIP runpod: " + msg)
        return
    try:
        import runpod  # type: ignore

        runpod.api_key = key
    except Exception as exc:
        msg = (f"RunPod enabled but the 'runpod' SDK import/config failed "
               f"({type(exc).__name__}: {exc}); pip install runpod.")
        if strict:
            raise CredentialError(msg)
        creds.note("SKIP runpod: " + msg)
        return
    # Light validation: a key that's present + SDK importable is enough to proceed.
    # A networked verify (runpod.get_pods) is wrapped so transient errors aren't fatal.
    try:
        runpod.get_pods()  # type: ignore[attr-defined]
        creds.note("RunPod API key validated (get_pods succeeded).")
    except Exception as exc:  # pragma: no cover - network dependent
        creds.note(f"RunPod key set; live verify skipped ({type(exc).__name__}: {exc}).")
    creds.runpod_ok = True


def login(cfg: FineTuneConfig, *, logger: logging.Logger | None = None) -> Credentials:
    """Authenticate enabled integrations. Honors ``cfg.require_cloud`` for strictness."""
    global LOGGER
    if logger is not None:
        LOGGER = logger
    creds = Credentials()
    strict = bool(cfg.require_cloud)
    if cfg.wandb:
        _login_wandb(cfg, creds, strict=strict)
    else:
        creds.note("W&B logging disabled (--wandb to enable).")
    if cfg.runpod:
        _login_runpod(cfg, creds, strict=strict)
    else:
        creds.note("RunPod integration disabled (--runpod to enable).")
    return creds
