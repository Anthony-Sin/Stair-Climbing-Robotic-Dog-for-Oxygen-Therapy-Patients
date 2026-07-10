# Browser-tier trace recorder (`gaitTrace`)

Full per-frame time-series capture of the REAL rendered patient rig — the
"round 2" numeric-analysis input for the patient IK/gait overhaul (see
`IK_OVERHAUL_SPEC.md` and `audit/TRACE_SCHEMA.md` for the exact output
schema). Unlike `audit/run_browser_audit.md`'s `patientDiag` (pass/fail bars),
this produces RAW data — trajectories, per-foot event timing, bone
positions/orientations, terrain heights — for a downstream analyzer to mine.

**KNOWN PITFALL (from prior sessions):** the repo's `preview_start`/`preview_*`
tooling serves the MAIN checkout of this tool, not a worktree's. If you're
working in an isolated git worktree (e.g. `.claude/worktrees/ik-overhaul`),
you MUST start `serve.py` yourself from THAT worktree's
`src/tools/blueprint_viewer` directory — `run_browser_trace.py` does this for
you automatically (see its own docstring).

## The 3-command flow

From `src/tools/blueprint_viewer` (PowerShell or any shell with `python` on
`PATH`):

```powershell
# 1) Run the full sweep (default dt=1/60s ~ 60Hz, both clips) via headless Chrome.
#    Starts its own serve.py on port 8971 (NOT 8741 -- avoids clashing with a
#    main-checkout viewer instance that might already be running there),
#    launches headless Chrome pointed at ?gaittrace=1, waits for
#    diag/trace_full.json to appear, patches meta.headCommit, then cleans up
#    both processes.
python audit/run_browser_trace.py

# 2) (optional) Faster smoke run at a coarser dt, custom output name.
python audit/run_browser_trace.py --dt 0.1 --name trace_smoke

# 3) Sanity-check the result numerically (NaN/null scan, expected sample
#    count, stance-foot-vs-terrain clearance, phaseC monotonicity, etc. --
#    write your own small script per audit/TRACE_SCHEMA.md's documented
#    fields, or reuse whatever check script accompanied the run that produced
#    this doc; see "Sanity-check checklist" below for what to verify).
python -c "import json; d=json.load(open('diag/trace_full.json')); print(d['meta']); print([(s['name'], len(s['samples'])) for s in d['segments']])"
```

## Flags (`run_browser_trace.py`)

| flag | default | meaning |
|---|---|---|
| `--port` | `8971` | serve.py port (NOT serve.py's own default `8741`) |
| `--dt` | `1/60` (~0.01667) | sample step in SECONDS (CLAUDE.md 8.6 — never frame counts) |
| `--name` | `trace_full` | output lands at `diag/<name>.json` |
| `--chrome` | auto-detected | explicit `chrome.exe` path if not at a standard Windows install location |
| `--timeout` | `150` | seconds to wait for the diag file to appear, PER Chrome preset tried (so a full run can take up to `2 x timeout` in the worst case where the first preset silently fails) |

## Why two Chrome presets

The app's entire boot sequence (not just this diagnostic) constructs a real
`THREE.WebGLRenderer` — if headless Chrome can't get a working WebGL context
at all, the page never reaches `window.__viewer.ready`, so `gaitTrace()` never
even runs. `run_browser_trace.py` tries a GPU-accelerated headless preset
first, falling back to a forced SwiftShader (software GL) preset if the first
one doesn't produce output within `--timeout` seconds. Both presets' Chrome
stdout/stderr are logged to `audit/_chrome_gpu.log` / `audit/_chrome_swiftshader.log`
(gitignored) for post-mortem debugging — note headless Chrome's forwarding of
in-page `console.log`/`console.error` to its own process output is
best-effort, not guaranteed, so an empty/unhelpful log does not by itself mean
nothing went wrong in the page.

## Manual fallback (if headless truly won't render)

Serve the worktree and drive it from a REAL (non-headless) browser tab:

```powershell
$env:PORT = 8971
python serve.py
# then open http://127.0.0.1:8971/?gaittrace=1 in a real Chrome/Edge window,
# or open the page normally and run in the DevTools console:
#   window.__viewer.gaitTrace({dt: 1/60, name: 'trace_full'})
#     .then(p => console.log('saved:', p));
# diag/trace_full.json appears once the promise resolves.
```

## Sanity-check checklist (numeric, not visual — per this tool's own recurring
"don't trust screenshots" lesson, AGENTS.md incidents #4-#15)

- Expected sample count per segment ≈ `segment.duration / dt` (+1 for the
  inclusive endpoint) — compare against `len(samples)`.
- Zero `NaN`/`null` in fields documented as always-numeric (`tGlobal`,
  `tLocal`, `pose.rootX/Y/Z/rootYaw/speed/phaseC/support`, every `bones.*.{x,y,z}`).
- Stance-foot toe height ≈ terrain height: for samples where
  `pose.leftFoot.planted` (or `rightFoot.planted`), `abs(bones.leftToeBase.z -
  terrain.underLeftToe)` should be small (structural bar per
  `IK_OVERHAUL_SPEC.md` M4/M8 is ≤1e-6 m / ≥-0.002 m — a coarser ~0.01 m bar on
  >90% of planted samples is a reasonable smoke bar for a first pass).
- Root speed consistency: recompute `|Δ(rootX,rootY)| / dt` between
  consecutive samples and compare to the recorded `pose.speed` (central-
  difference in the source, so expect noise near liftoff/landing edges —
  tolerance ~20% on smoothed values, per this task's own verification note).
- `pose.phaseC` monotone non-decreasing within a segment (never resets/jumps
  backward — the whole point of phaseC replacing the legacy staircase
  `gaitPhaseLegacy`, see `IK_OVERHAUL_SPEC.md` P1/M7).
