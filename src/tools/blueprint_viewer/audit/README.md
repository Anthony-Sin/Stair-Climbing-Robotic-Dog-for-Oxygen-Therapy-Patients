# Node-tier gait audit

Headless, numeric verification of `js/PatientGait.js` (spec: `IK_OVERHAUL_SPEC.md`
section 8, "Node tier") — replays the REAL scheduler/pose-query code against the
REAL recorded patient path, entirely in Node, no browser, no screenshots. This is
the "don't trust screenshots" layer: every metric is a computed number checked
against a documented bar (see `gait_audit.mjs`'s own header for the full list:
M1 step length, M2 cadence/alternation, M3 duty factor, M4 foot penetration, M5
double-support fraction, M6 max root travel while double-planted, M7 phaseC
continuity, M9 cane, M10 idle motion).

`js/PatientGait.js` is deliberately zero-import pure math (no THREE, no DOM — see
its own header), so it can be `import()`-ed directly in Node.

## Files

```
extract_tracks.py       Pulls patient_root's position/quaternion animation tracks out of
                         models/robot.glb (+ stair_spec/landing_far_x_m from
                         models/robot.meta.json) into out/tracks.json. Pure stdlib
                         (struct+json) GLB parser -- no pygltflib dependency, even though
                         pipeline/validate_glb.py uses that package and it happens to be
                         installed in this environment (see the script's own docstring for
                         why this was still worth writing from scratch, and how it was
                         cross-checked against pygltflib's own independent reader).
gait_audit.mjs           The actual audit. Zero npm dependencies (Node >=18 built-ins only).
                         Imports a PatientGait.js-shaped module, rebuilds terrain + one
                         schedule per "case", sweeps poseAt() at 60 Hz, and writes a JSON
                         report + prints a console summary table.
out/                     Generated (gitignored) -- tracks.json, the baseline module
                         snapshot, and report_*.json all live here.
```

## The 3-command flow

From `src/tools/blueprint_viewer` (PowerShell):

```powershell
# 1) Extract the real recorded patient_root tracks from the baked GLB.
python audit/extract_tracks.py
# -> prints per-clip duration/fps/x/y/z sanity numbers, writes audit/out/tracks.json.
# Re-run this whenever models/robot.glb is re-baked (pipeline/bake_gltf.py).

# 2) Audit the LIVE module (js/PatientGait.js as it stands right now).
node audit/gait_audit.mjs --label live --out audit/out/report_live.json

# 3) Audit a BASELINE snapshot for before/after comparison (e.g. the pre-overhaul
#    version at some git ref) and diff the two reports.
git show HEAD:src/tools/blueprint_viewer/js/PatientGait.js > audit/out/PatientGait_baseline.mjs
node audit/gait_audit.mjs --module audit/out/PatientGait_baseline.mjs --label baseline --out audit/out/report_baseline.json
```

(`git show HEAD:...` is read-only — it does not touch the working tree. Pick
whatever ref you want "before" to mean; HEAD is just the default "no local
changes yet" case. The `.mjs` extension on the snapshot matters: see "Node
quirk" below.)

Step 1 is optional for step 2/3 — `gait_audit.mjs` also runs three synthetic
fixtures (constant-velocity, stop-and-go, zig-zag) on flat terrain that need no
GLB/tracks.json at all, so the audit still produces a report (with "follow"/
"climb" cases skipped and a console warning) if you only ever run steps 2-3.

Each `node audit/gait_audit.mjs` run also prints a full metrics table to the
console (value + bar + PASS/FAIL/NA per metric per case) — the JSON report has
the same data for scripting/diffing.

## Reading a report

```jsonc
{
  "generatedAt": "...", "label": "live", "modulePath": "...", "overallPass": false,
  "cases": {
    "follow": { "duration": 23.6, "stepCount": 44, "pass": true, "features": { "hasPhaseC": false, ... },
      "metrics": {
        "M1_stepLengthMedian": { "value": 0.36, "bar": "[0.15, 0.45] m", "pass": true },
        "M7_phaseC": { "value": null, "bar": "...", "pass": "na(pending-gait-v2)" },
        ...
      } },
    "climb": { ... }, "constant": { ... }, "stopgo": { ... }, "zigzag": { ... }
  }
}
```

- `pass` is `true`, `false`, or a string starting with `"na"` — `"na(...)"` never
  counts against a case's own `pass` (see `gait_audit.mjs`'s `auditSchedule`).
  Two `na` flavors: `"na(no steps)"`-style (a structural non-applicability, e.g.
  a fixture with zero footfalls) and `"na(pending-gait-v2)"` (the metric needs a
  PatientGait.js v2 field — `phaseC`, `support`, or `cane` — that the loaded
  module doesn't produce; feature-detected via one probe `poseAt()` call, never
  a crash). Re-run once GAIT/RIG land v2 to get real numbers for those rows.
- `overallPass` (top level) is the AND of every case's `pass`.
- Per spec section 10 ("Definition of done"), every bar is checked identically
  on BOTH the `follow` and `climb` cases — there's a single implementation of
  each metric (`auditSchedule`), not per-clip copies.

## Known-real (not tool-bug) findings as of the initial baseline/live run

Both reports currently come back byte-identical (`js/PatientGait.js` hadn't been
touched yet in this worktree at extraction time) and `overallPass: false` on
both — NOT an audit-tooling bug. The v1 (pre-overhaul) module correctly fails
exactly the metrics spec section 4's "G1 REDUCE THE GLIDE" work item exists to
fix: on `climb` (and the synthetic `stopgo` fixture), M3 duty factor and M5
double-support fraction read too high and M6 (root travel while double-planted)
reads ~0.61 m against a 0.20 m bar — the documented P3 "glide" bug, now with
numbers. `follow` (faster, ~0.26 m/s) already passes M3/M5/M6 even pre-overhaul.
M7/M9 report `na(pending-gait-v2)` everywhere, as expected before GAIT lands
`phaseC`/`support`/cane events.

## Node quirk found while building this (worth knowing before debugging "why
didn't `node --check` catch that?")

`node --check somefile.js` is UNRELIABLE on a plain `.js` file that Node treats
as an ES module by its "detect module syntax" heuristic (any file with a
top-level `import`/`export` and no controlling `package.json` — which is exactly
`js/PatientGait.js`, `js/main.js`, etc. in this tool, since there's no
`package.json` anywhere in this repo): it can return exit 0 on genuinely broken
syntax (confirmed empirically — `const x = ;` behind a leading `import` line
passed `node --check`). The SAME file with a `.mjs` extension checks correctly.
Actually loading the module — `node --check file.mjs`, or a real `import()` /
`node file.mjs` — is reliable; `node --check file.js` on an import/export file
is not. `gait_audit.mjs` itself is `.mjs` for this reason, and its own
`import()` of whatever `--module` path you pass is itself a real syntax/load
check (stronger than `node --check` would be) — a broken `PatientGait.js` fails
loudly here (`[gait_audit] FATAL: ...`), it does not silently produce a clean
report.

## Extending

- New Node-tier metric: add it inside `auditSchedule()` in `gait_audit.mjs` —
  it automatically runs against all 5 cases (`follow`, `climb`, `constant`,
  `stopgo`, `zigzag`) for free, on any PatientGait.js version, old or new.
- New synthetic fixture: add a `buildXSamples()` generator (same `{t,x,y,zRoot,
  yaw,groundRef}` shape as `PatientGait.extractPathSamples`'s own output) and
  one more `cases.x = auditSchedule(...)` call in `main()`.
