#!/usr/bin/env python
"""Headless-Chrome driver for js/main.js's `?shotseries=1` boot hook (round 2 of
the patient IK/gait overhaul VISUAL-VERIFY pass). Sibling of run_browser_trace.py
-- same rationale, same start-server/launch-chrome/poll/cleanup shape -- but
polls for a screenshot MANIFEST + the PNG files it lists, instead of one JSON
trace file.

WHY this exists (same KNOWN PITFALL as run_browser_trace.py, see that file's
own module docstring and the blueprint-viewer-capture-and-swap-pitfalls note):
the repo's preview_start/preview_* tooling serves the MAIN checkout's copy of
this tool, not THIS worktree's -- so a driver that wants to screenshot
worktree-local changes to js/main.js/PatientGait.js/PatientHuman.js must start
its OWN serve.py (this script does, from THIS file's own directory's parent)
and its own browser, never preview_*.

What it does, in order:
  1. Starts serve.py on a free, NON-default port (default 8972 here -- NOT
     8741, serve.py's own default, and NOT 8971, run_browser_trace.py's
     default, so both drivers can run without colliding).
  2. Launches headless Chrome pointed at
     http://127.0.0.1:<port>/?shotseries=1 -- js/main.js's `_runShotSeries()`
     boot hook (see that file's own comment above the hook) scrubs the unified
     timeline to a fixed list of (time, label, view) capture points, points
     the camera at the patient, and POSTs each frame to serve.py's POST /shot
     sink, finishing with a manifest POSTed to the EXISTING POST /diag sink
     (name=shotseries_manifest) listing every filename it expects to exist.
  3. Polls diag/shotseries_manifest.json for existence (same pattern as
     run_browser_trace.py's _wait_for_diag), then polls shots/ until every
     filename the manifest lists is actually present on disk (the manifest
     POST happens AFTER every shot POST resolves -- see the hook's own
     ordering comment -- but PNGs and the manifest still land via two
     separate requests, so a brief settle window after the manifest appears
     is worth a short extra poll, not assumed atomic).
  4. Sanity-checks every expected PNG: file size > 20 KB, and not a
     degenerate uniformly-flat frame (byte-variance heuristic over the raw
     file bytes -- Pillow may or may not be installed, so this does not
     depend on it; see _looks_nontrivial()).
  5. Kills Chrome + the server it started (never touches a server/browser it
     did not itself launch).

Tries TWO Chrome flag presets in order -- GPU-accelerated headless first, then
a SwiftShader (software GL) fallback -- for the same reason as
run_browser_trace.py: the whole app boot sequence constructs a real
THREE.WebGLRenderer, so headless Chrome needs A working WebGL context of SOME
kind or the page never reaches window.__viewer.ready, let alone runs the
shotseries hook.

Usage (from src/tools/blueprint_viewer, PowerShell or any shell):
    python audit/run_shot_series.py
    python audit/run_shot_series.py --port 8973 --timeout 180
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
_VIEWER_DIR = os.path.dirname(_DIR)  # src/tools/blueprint_viewer
_DIAG_DIR = os.path.join(_VIEWER_DIR, "diag")
_SHOT_DIR = os.path.join(_VIEWER_DIR, "shots")

_DEFAULT_PORT = 8972  # NOT 8741 (serve.py's own default) or 8971 (run_browser_trace.py's) -- see module docstring
_MANIFEST_NAME = "shotseries_manifest"
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

# Same two presets as run_browser_trace.py, verbatim -- see that file's own
# comment for why both exist and in this order.
_PRESET_A_GPU = [
    "--headless=new",
    "--window-size=1280,900",
    "--autoplay-policy=no-user-gesture-required",
    "--enable-logging=stderr",
    "--v=1",
]
_PRESET_B_SWIFTSHADER = [
    "--headless=new",
    "--window-size=1280,900",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-gpu",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--enable-logging=stderr",
    "--v=1",
]


def _find_chrome(explicit):
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise SystemExit(f"--chrome path does not exist: {explicit}")
    for cand in _CHROME_CANDIDATES:
        if cand and os.path.isfile(cand):
            return cand
    raise SystemExit(
        "chrome.exe not found at any standard Windows path. Tried:\n  "
        + "\n  ".join(c for c in _CHROME_CANDIDATES if c)
        + "\nPass --chrome <path-to-chrome.exe> explicitly."
    )


def _port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_server(port, timeout_s=15):
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.25)
    return False


def _start_server(port):
    if not _port_is_free(port):
        raise SystemExit(
            f"port {port} is already in use -- pick a different --port "
            f"(the default {_DEFAULT_PORT} is chosen to avoid serve.py's own "
            f"default 8741 and run_browser_trace.py's 8971, but this machine "
            f"may have something else on it)."
        )
    env = dict(os.environ)
    env["PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, "serve.py"],
        cwd=_VIEWER_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not _wait_for_server(port):
        proc.terminate()
        out = proc.stdout.read() if proc.stdout else ""
        raise SystemExit(f"serve.py never became reachable on port {port}. Output:\n{out}")
    return proc


def _run_chrome(chrome_path, url, extra_flags, log_path):
    args = [chrome_path, *extra_flags, url]
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_fh


def _wait_for_manifest(timeout_s, chrome_proc):
    """Poll diag/<_MANIFEST_NAME>.json for existence + valid JSON, same
    pattern as run_browser_trace.py's _wait_for_diag."""
    out_path = os.path.join(_DIAG_DIR, f"{_MANIFEST_NAME}.json")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.isfile(out_path):
            for _ in range(10):
                try:
                    with open(out_path, encoding="utf-8") as fh:
                        return json.load(fh)
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.2)
            return None  # never parsed cleanly -- let the caller's own check surface it
        if chrome_proc.poll() is not None:
            return None  # Chrome exited on its own -- almost certainly a crash/flag rejection
        time.sleep(0.5)
    return None


def _wait_for_shots(expected_names, timeout_s):
    """Poll shots/ until every filename in expected_names exists on disk.
    Returns (present, missing) lists."""
    deadline = time.time() + timeout_s
    missing = list(expected_names)
    present = []
    while time.time() < deadline and missing:
        still_missing = []
        for name in missing:
            if os.path.isfile(os.path.join(_SHOT_DIR, name)):
                present.append(name)
            else:
                still_missing.append(name)
        missing = still_missing
        if missing:
            time.sleep(0.3)
    return present, missing


def _looks_nontrivial(path, min_bytes=20_000):
    """Cheap, dependency-free sanity check on a captured PNG: big enough to
    plausibly be a real 3D render (a blank/degenerate frame PNG-compresses to
    a tiny file), and not byte-uniform (a solid black or solid white frame
    round-trips through zlib to a very low-variance byte stream even though
    its FILE size can still look plausible for a low-detail solid fill).
    Pure stdlib (no Pillow dependency) -- operates on the raw compressed PNG
    bytes as a heuristic, not on decoded pixels, per the task's own allowance
    ("a byte-variance heuristic on the raw file is acceptable if not [Pillow
    installed]"). Returns (ok: bool, reason: str)."""
    size = os.path.getsize(path)
    if size <= min_bytes:
        return False, f"only {size} bytes (<= {min_bytes})"
    with open(path, "rb") as fh:
        data = fh.read()
    # Skip the PNG signature + IHDR (33 bytes) so we don't measure header
    # constant bytes; sample the (compressed) IDAT-and-beyond tail.
    sample = data[33:]
    if len(sample) < 256:
        return False, "compressed payload implausibly small"
    # A real photo-like/3D render, even PNG-compressed, still has substantial
    # byte-to-byte variance. A solid-color frame compresses to long runs of
    # near-identical bytes -- stdev collapses toward 0.
    stride = max(1, len(sample) // 4000)  # cap sample count for speed on large files
    values = sample[::stride]
    if len(values) < 32:
        return False, "not enough sampled bytes"
    stdev = statistics.pstdev(values)
    if stdev < 3.0:
        return False, f"byte stdev {stdev:.2f} looks uniform/blank (< 3.0)"
    return True, f"{size} bytes, byte stdev {stdev:.2f}"


def _patch_head_commit_into_manifest():
    """Best-effort, same rationale as run_browser_trace.py's
    _patch_head_commit -- browser JS has no git access."""
    path = os.path.join(_DIAG_DIR, f"{_MANIFEST_NAME}.json")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_VIEWER_DIR, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover -- best-effort, never fatal
        print(f"[run_shot_series] WARNING: could not read git HEAD ({exc}); leaving manifest as-is")
        return
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["headCommit"] = commit
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or _DEFAULT_PORT))
    ap.add_argument("--chrome", default=None, help="explicit path to chrome.exe (auto-detected otherwise)")
    ap.add_argument("--timeout", type=int, default=150, help="seconds to wait for the manifest+shots per preset (default 150)")
    ap.add_argument("--min-bytes", type=int, default=20_000, help="minimum PNG size to count as non-trivial (default 20000)")
    args = ap.parse_args()

    chrome_path = _find_chrome(args.chrome)
    print(f"[run_shot_series] chrome: {chrome_path}")
    print(f"[run_shot_series] worktree viewer dir: {_VIEWER_DIR}")

    manifest_path = os.path.join(_DIAG_DIR, f"{_MANIFEST_NAME}.json")
    if os.path.isfile(manifest_path):
        os.remove(manifest_path)  # never read a stale manifest from a previous run as if it were fresh

    server_proc = _start_server(args.port)
    print(f"[run_shot_series] serve.py up on port {args.port}")

    url = f"http://127.0.0.1:{args.port}/?shotseries=1"
    presets = [("gpu", _PRESET_A_GPU), ("swiftshader", _PRESET_B_SWIFTSHADER)]

    manifest = None
    try:
        for label, flags in presets:
            log_path = os.path.join(_DIR, f"_chrome_shots_{label}.log")
            print(f"[run_shot_series] trying preset '{label}' -> {url}")
            print(f"[run_shot_series]   chrome log: {log_path}")
            chrome_proc, log_fh = _run_chrome(chrome_path, url, flags, log_path)
            try:
                manifest = _wait_for_manifest(args.timeout, chrome_proc)
            finally:
                if chrome_proc.poll() is None:
                    chrome_proc.terminate()
                    try:
                        chrome_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        chrome_proc.kill()
                log_fh.close()
            if manifest:
                print(f"[run_shot_series] preset '{label}' SUCCEEDED (manifest with {len(manifest.get('shots', []))} shots)")
                break
            print(f"[run_shot_series] preset '{label}' FAILED (no {manifest_path} within {args.timeout}s) -- see {log_path}")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[run_shot_series] serve.py stopped")

    if not manifest:
        raise SystemExit(
            "shotseries hook never produced "
            + manifest_path
            + " under EITHER Chrome preset (gpu, swiftshader). Check "
            "audit/_chrome_shots_gpu.log and audit/_chrome_shots_swiftshader.log "
            "for whatever Chrome itself printed (best-effort, not guaranteed to "
            "contain page console.log/error output)."
        )

    _patch_head_commit_into_manifest()

    expected_names = [s["name"] for s in manifest.get("shots", [])]
    print(f"[run_shot_series] manifest lists {len(expected_names)} shots (idle window: {manifest.get('idle')})")

    present, missing = _wait_for_shots(expected_names, timeout_s=30)
    if missing:
        raise SystemExit(
            f"manifest listed {len(expected_names)} shots but {len(missing)} never appeared in "
            f"{_SHOT_DIR} within 30s of the manifest landing: {missing}"
        )

    failures = []
    print(f"[run_shot_series] verifying {len(expected_names)} PNGs (>{args.min_bytes} bytes, non-uniform)...")
    for name in expected_names:
        path = os.path.join(_SHOT_DIR, name)
        ok, reason = _looks_nontrivial(path, min_bytes=args.min_bytes)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: {reason}")
        if not ok:
            failures.append((name, reason))

    if failures:
        raise SystemExit(
            f"{len(failures)}/{len(expected_names)} captured shots failed the non-trivial check: "
            + ", ".join(f"{n} ({r})" for n, r in failures)
        )

    print(f"[run_shot_series] DONE: all {len(expected_names)} shots present and non-trivial in {_SHOT_DIR}")


if __name__ == "__main__":
    main()
