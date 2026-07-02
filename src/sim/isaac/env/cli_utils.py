"""isaac_env.py extraction (Phase 2 split): cli_utils. Verbatim bodies; only env_state requalification added."""
import os
import sys

def _flag_passed(*names: str) -> bool:
    """True if any of these option strings were given on the command line.

    Lets the --sim2real-validation-cam preset supply a value WITHOUT overriding an
    explicit per-flag choice the user made.
    """
    return any(a == n or a.startswith(n + "=") for a in sys.argv[1:] for n in names)

def _log_bucket(log_dir: str, bucket: str) -> str:
    """Return <log_dir>/<bucket>, creating it. Each run folder is organised into
    videos/ (mp4s), reports/ (summaries, verification PNGs, JSON), and debug/
    (verbose JSONL/raw logs). Returns log_dir itself when log_dir is empty."""
    if not log_dir:
        return log_dir
    d = os.path.join(log_dir, bucket)
    os.makedirs(d, exist_ok=True)
    return d
