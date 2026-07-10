#!/usr/bin/env python
"""Headless-Chrome driver for js/main.js's gaitTrace() diagnostic (round 2 of the
patient IK/gait overhaul -- see audit/TRACE_SCHEMA.md for the output schema).

WHY this exists (KNOWN PITFALL from prior sessions, see
blueprint-viewer-capture-and-swap-pitfalls in the agent's own memory): the
repo's preview_start/preview_* tooling serves the MAIN checkout's copy of this
tool, not THIS worktree's -- so a driver that wants to exercise worktree-local
changes to js/main.js/PatientGait.js/PatientHuman.js must start its OWN
serve.py (this script does, from THIS file's own directory's parent) and its
own browser, never preview_*.

What it does, in order:
  1. Starts serve.py (this directory's parent) on a free, NON-default port
     (default 8971 here -- NOT 8741, which is serve.py's own default and may
     already be in use by a main-checkout viewer instance running elsewhere on
     this machine; override with --port or the PORT env var).
  2. Launches headless Chrome (chrome.exe, standard Windows install path,
     override with --chrome) pointed at
     http://127.0.0.1:<port>/?gaittrace=1&dt=<dt>[&name=<name>] -- the
     js/main.js boot hook documented in audit/TRACE_SCHEMA.md's "Auto-run
     hook" section runs gaitTrace() once the viewer finishes loading and POSTs
     the result to serve.py's own POST /diag sink.
  3. Polls diag/<name>.json for existence (timeout + a clear error message
     that also surfaces whatever Chrome itself printed to stderr, best-effort,
     since headless Chrome does not reliably forward page console.log/error to
     its own process output).
  4. Patches meta.headCommit into the saved JSON (browser JS has no git
     access -- see js/main.js's own gaitTrace() comment) via `git rev-parse
     HEAD` run HERE, on the host, read-only.
  5. Kills Chrome + the server it started (never touches a server/browser it
     did not itself launch).

Tries TWO Chrome flag presets in order -- GPU-accelerated headless first, then
a SwiftShader (software GL) fallback -- since the app's entire boot sequence
(not just this diagnostic) constructs a real THREE.WebGLRenderer, so headless
Chrome needs A working WebGL context of SOME kind or the page never reaches
window.__viewer.ready at all, let alone runs gaitTrace().

Usage (from src/tools/blueprint_viewer, PowerShell or any shell):
    python audit/run_browser_trace.py
    python audit/run_browser_trace.py --dt 0.05 --name trace_smoke   # faster smoke run
    python audit/run_browser_trace.py --port 8972 --timeout 180
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
_VIEWER_DIR = os.path.dirname(_DIR)  # src/tools/blueprint_viewer
_DIAG_DIR = os.path.join(_VIEWER_DIR, "diag")

_DEFAULT_PORT = 8971  # NOT 8741 (serve.py's own default) -- see module docstring
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

# Two flag presets, tried in order. Both use --headless=new (Chrome 109+'s
# modern headless mode, with materially better WebGL/GPU support than the
# legacy --headless). Preset A lets Chrome pick its normal (often
# GPU-accelerated via ANGLE/D3D11 on Windows) GL backend; Preset B forces
# SwiftShader (CPU) rendering for environments where the real GPU path is
# blocked/unavailable to a headless process (common in CI/remote-desktop/RDP
# sessions -- exactly the kind of environment this might run in).
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
            f"default 8741, but this machine may have something else on it)."
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


def _run_chrome(chrome_path, url, extra_flags, log_path, timeout_s):
    """Launch chrome with extra_flags, wait up to timeout_s for the process to
    either exit on its own (it won't, normally -- headless Chrome pointed at a
    URL with no --dump-dom/--screenshot flag just sits there rendering) or for
    the caller's own polling (done by the CALLER, not here) to be satisfied.
    Returns the live Popen so the caller can poll diag/ in parallel and kill
    this when done. stdout+stderr are teed to log_path for post-mortem
    debugging (see module docstring: headless Chrome's OWN console-forwarding
    is unreliable, so this is best-effort, not guaranteed to contain page
    console.log/error output)."""
    args = [chrome_path, *extra_flags, url]
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_fh


def _wait_for_diag(name, timeout_s, chrome_proc):
    out_path = os.path.join(_DIAG_DIR, f"{name}.json")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.isfile(out_path):
            # File just appeared -- give the (single-threaded server) writer a
            # brief moment to finish flushing before we read it back, then
            # sanity-check it actually parses as JSON (a half-written file
            # would otherwise pass the isfile() check and fail confusingly
            # downstream in the caller's own sanity-check step).
            for _ in range(10):
                try:
                    with open(out_path, encoding="utf-8") as fh:
                        json.load(fh)
                    return out_path
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.2)
            return out_path  # last attempt already raised nothing fatal here; let the caller's own check surface it
        if chrome_proc.poll() is not None:
            # Chrome exited on its own -- almost certainly a crash/flag
            # rejection, not success (a normal run just sits there rendering
            # until we kill it).
            return None
        time.sleep(0.5)
    return None


def _patch_head_commit(path):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_VIEWER_DIR, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover -- best-effort, never fatal
        print(f"[run_browser_trace] WARNING: could not read git HEAD ({exc}); leaving headCommit=null")
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("meta", {})["headCommit"] = commit
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"[run_browser_trace] patched meta.headCommit = {commit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or _DEFAULT_PORT))
    ap.add_argument("--dt", type=float, default=1 / 60, help="sample step in SECONDS (default 1/60 s)")
    ap.add_argument("--name", default="trace_full", help="diag/<name>.json output name (default trace_full)")
    ap.add_argument("--chrome", default=None, help="explicit path to chrome.exe (auto-detected otherwise)")
    ap.add_argument("--timeout", type=int, default=150, help="seconds to wait for diag/<name>.json per preset (default 150)")
    args = ap.parse_args()

    chrome_path = _find_chrome(args.chrome)
    print(f"[run_browser_trace] chrome: {chrome_path}")
    print(f"[run_browser_trace] worktree viewer dir: {_VIEWER_DIR}")

    out_path = os.path.join(_DIAG_DIR, f"{args.name}.json")
    if os.path.isfile(out_path):
        os.remove(out_path)  # never read a stale file from a previous run as if it were fresh

    server_proc = _start_server(args.port)
    print(f"[run_browser_trace] serve.py up on port {args.port}")

    url = f"http://127.0.0.1:{args.port}/?gaittrace=1&dt={args.dt}&name={args.name}"
    presets = [("gpu", _PRESET_A_GPU), ("swiftshader", _PRESET_B_SWIFTSHADER)]

    result_path = None
    try:
        for label, flags in presets:
            log_path = os.path.join(_DIR, f"_chrome_{label}.log")
            print(f"[run_browser_trace] trying preset '{label}' -> {url}")
            print(f"[run_browser_trace]   chrome log: {log_path}")
            chrome_proc, log_fh = _run_chrome(chrome_path, url, flags, log_path, args.timeout)
            try:
                result_path = _wait_for_diag(args.name, args.timeout, chrome_proc)
            finally:
                if chrome_proc.poll() is None:
                    chrome_proc.terminate()
                    try:
                        chrome_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        chrome_proc.kill()
                log_fh.close()
            if result_path:
                print(f"[run_browser_trace] preset '{label}' SUCCEEDED: {result_path}")
                break
            print(f"[run_browser_trace] preset '{label}' FAILED (no {out_path} within {args.timeout}s) -- see {log_path}")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[run_browser_trace] serve.py stopped")

    if not result_path:
        raise SystemExit(
            "gaitTrace never produced diag/"
            + args.name
            + ".json under EITHER Chrome preset (gpu, swiftshader). "
            "Check audit/_chrome_gpu.log and audit/_chrome_swiftshader.log for "
            "whatever Chrome itself printed (headless Chrome's forwarding of "
            "page console.log/error to its own stderr is best-effort, not "
            "guaranteed -- if both logs are empty/unhelpful, the manual fallback "
            "in audit/run_browser_trace.md documents how to drive this by hand "
            "in a real (non-headless) browser instead)."
        )

    _patch_head_commit(result_path)

    with open(result_path, encoding="utf-8") as fh:
        data = json.load(fh)
    seg_summary = ", ".join(f"{s['name']}={len(s['samples'])} samples" for s in data.get("segments", []))
    print(f"[run_browser_trace] DONE: {result_path}")
    print(f"[run_browser_trace]   dt={data.get('meta', {}).get('dt')}  segments: {seg_summary}")


if __name__ == "__main__":
    main()
