"""Log-components CLI validation (extracted from args_parser).

Owns the ``--log-components`` allowlist constant and the parser-level
validator used by :func:`core.args_parser.parse_args`.
"""

import argparse


_VALID_VISION_LOG_COMPONENTS = {"none", "all", "vision.main", "vision.exporter"}
 
 
def _normalize_log_components(parser: argparse.ArgumentParser, raw_value: str) -> str:
    parts = [part.strip() for part in raw_value.split(',') if part.strip()]
    if not parts:
        parser.error("--log-components requires at least one value")
 
    invalid = [part for part in parts if part not in _VALID_VISION_LOG_COMPONENTS]
    if invalid:
        parser.error(
            "--log-components only accepts: none, all, vision.main, vision.exporter"
        )
 
    unique_parts = list(dict.fromkeys(parts))
    if "none" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=none cannot be combined with other values")
    if "all" in unique_parts and len(unique_parts) > 1:
        parser.error("--log-components=all cannot be combined with other values")
 
    return ",".join(unique_parts)
