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


def _fetch_atomic(src: str, dest: str, want_sha256: str) -> bool:
    """Download `src` to `dest` atomically: write to a sibling .part temp, verify the sha256,
    and only os.replace() into place on success. On any mismatch/error the partial file is
    removed and `dest` is left untouched -- a fresh clone never ends up with a half-written or
    corrupt weight (review §10).
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(src, tmp)
        got = _sha256(tmp)
        if got != want_sha256:
            print(f"         ERROR: downloaded sha256 {got[:12]} != manifest {want_sha256[:12]}"
                  f"  (kept nothing; check the 'source' / release tag)")
            return False
        os.replace(tmp, dest)  # atomic on same filesystem
        print(f"         OK: {dest} ({got[:12]})")
        return True
    except Exception as exc:  # network / IO / hash -- never leave a partial file behind
        print(f"         ERROR: fetch failed ({exc})")
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download missing/corrupt weights whose 'source' is a real URL")
    args = ap.parse_args()

    with open(_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    present = missing = corrupt = fetch_failed = 0
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
            if not _fetch_atomic(src, path, want):
                fetch_failed += 1

    print(f"\n{present} present, {missing} missing, {corrupt} corrupt "
          f"(of {len(manifest['entries'])})")
    if args.fetch:
        return 1 if fetch_failed else 0
    return 1 if (corrupt or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
