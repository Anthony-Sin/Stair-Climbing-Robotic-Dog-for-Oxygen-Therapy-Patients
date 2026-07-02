#!/usr/bin/env python3
"""Verify (and, once sources are filled in, fetch) the model weights from weights_manifest.json.

The runtime weights are gitignored (*.pt/*.trt/*.onnx/*.npz), so a fresh clone has none and
nothing records what they should be (review §9). This script reads weights_manifest.json and:

  * reports which weights are PRESENT / MISSING / CORRUPT (sha256 mismatch);
  * once each entry's "source" is filled in, downloads the missing/corrupt ones.

.trt/.engine files are marked non-portable in the manifest -- they are host-GPU specific and
must be rebuilt on the target (e.g. src/real/models/export_stairs_trt.py), not downloaded.

Usage:
    python scripts/fetch_weights.py            # verify local weights against the manifest
    python scripts/fetch_weights.py --fetch    # + download any missing (needs "source" URLs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST = os.path.join(_ROOT, "weights_manifest.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download missing/corrupt weights whose 'source' is a real URL")
    args = ap.parse_args()

    with open(_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    present = missing = corrupt = 0
    for e in manifest["entries"]:
        path = os.path.join(_ROOT, e["path"])
        want = e["sha256"]
        if os.path.exists(path):
            got = _sha256(path)
            if got == want:
                present += 1
                continue
            corrupt += 1
            print(f"CORRUPT  {e['path']}  (sha256 {got[:12]} != {want[:12]})")
        else:
            missing += 1
            print(f"MISSING  {e['path']}  -- {e['purpose']}")

        src = e.get("source", "")
        if not src.startswith(("http://", "https://")):
            print(f"         no fetchable source recorded (source={src!r})"
                  + ("  [rebuild on target -- not portable]" if not e.get("portable", True) else ""))
            continue
        if args.fetch:
            print(f"         downloading {src} ...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(src, path)
            got = _sha256(path)
            if got != want:
                print(f"         ERROR: downloaded sha256 {got[:12]} != manifest {want[:12]}")
                return 1

    print(f"\n{present} present, {missing} missing, {corrupt} corrupt "
          f"(of {len(manifest['entries'])})")
    return 1 if (corrupt or missing) and not args.fetch else 0


if __name__ == "__main__":
    sys.exit(main())
