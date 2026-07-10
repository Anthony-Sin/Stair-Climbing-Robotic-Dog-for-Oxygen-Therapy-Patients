# `gaitTrace` output schema (`diag/trace_full.json`)

Produced by `window.__viewer.gaitTrace({dt, name})` in `js/main.js` (round 2 of the
patient IK/gait overhaul — see `IK_OVERHAUL_SPEC.md`). This is a **raw, per-frame
time-series recorder**, not a pass/fail judge like `patientDiag`/`gaitReport`
(`audit/run_browser_audit.md` covers that tool) — it exists so a downstream
analyzer agent can compute its own metrics (trajectories, speeds, waypoint
timing, joint-angle smoothness) against the REAL rendered rig, at full frame
resolution, without re-deriving anything from screenshots.

## How it's produced (read this before trusting a number)

`gaitTrace` sweeps the SAME unified timeline / real code path the app itself
uses for scrubbing and autoplay: for each segment (`follow`, then `climb`, in
that order — mirrors `js/main.js`'s own `segments` array), for `t` from `0` to
the clip's `duration` (inclusive of the endpoint) in fixed `dt` steps, it does
exactly what `applyGlobalTime()`/`scrubToPercent()` do per frame — set
`action.time`, `mixer.update(0)`, `patientHuman.sync(segmentName, t)` — then
reads back real Three.js bone world transforms and the `PatientGait.poseAt()`
scheduler output for that same instant. It is a **read-only instrument**: no
gait/rig code is touched, and the viewer's prior phase/scrub position is
restored exactly (byte-for-byte, same technique as `patientDiag`) before the
call resolves — running this leaves no visible trace in the live app.

Default `dt` is `1/60` s (≈60 Hz) unless overridden via `opts.dt` or the
`?gaittrace=1&dt=<seconds>` boot query param (see "Auto-run hook" below).
Expected total sample count ≈ `(follow.duration + climb.duration) / dt` — at
the default dt and this deck's clip lengths (follow ≈23.6 s, climb ≈42.73 s)
that's roughly **3970-4000 samples** across both segments combined.

## Top-level shape

```jsonc
{
  "meta": {
    "generatedAt": "2026-07-10T12:34:56.789Z",  // wall-clock ISO stamp, for HUMANS reading the file only — not consumed downstream, not part of the determinism contract
    "dt": 0.016666666666666666,                  // seconds, the fixed sample step actually used
    "headCommit": "abcd1234" | null,              // patched in by audit/run_browser_trace.py AFTER saving (browser JS has no git access) — null means "not yet patched"
    "schemaVersion": 1,
    "columns": { /* short field-by-field docs, machine-readable mirror of this file's own tables below */ }
  },
  "segments": [
    { "name": "follow", "duration": 23.6, "samples": [ /* Sample, see below */ ] },
    { "name": "climb",  "duration": 42.73, "samples": [ /* Sample, see below */ ] }
  ]
}
```

Only segments that actually exist in the loaded model are present (mirrors
`js/main.js`'s own `segments` array construction) — a placeholder-model run
with no baked clips produces an empty `segments` array, not an error.

## `Sample` shape

```jsonc
{
  "tGlobal": 12.34,   // s, position on the UNIFIED follow+climb timeline (matches the scrubber/time-readout you'd see scrubbing manually) — computed from js/main.js's own `segments[i].start` + tLocal, so it's tied to the SAME mapping applyGlobalTime()/segmentAtGlobalTime() use, not independently recomputed
  "tLocal": 12.34,    // s, time within THIS segment's own clip (matches AnimationAction.time) — for "follow" tGlobal===tLocal (segment starts at 0); for "climb" tGlobal = follow.duration + tLocal
  "pose": { /* PatientGait.poseAt(schedule, terrain, tLocal) — see below */ },
  "bones": { /* P-frame bone world positions — see below */ },
  "pelvisOrientDeg": { "yawDeg": 0.0, "pitchDeg": 0.0, "rollDeg": 0.0 },
  "spine2YawDeg": 0.0,
  "cane": { "tip": {...} | null, "handleTarget": {...} | null, "handleEffective": {...} | null },
  "terrain": { "underRoot": 0.0, "underLeftToe": 0.0, "underRightToe": 0.0 },
  "lastSync": { /* shallow clone of patientHuman._lastSync right after this sample's sync() — see below */ }
}
```

### `pose` — `PatientGait.poseAt()`'s own contract, verbatim

Field names/meanings are **exactly** `IK_OVERHAUL_SPEC.md` section 3's contract
(re-derived here for convenience, not re-invented):

| field | meaning |
|---|---|
| `rootX`, `rootY`, `rootZ`, `rootYaw` | P-frame (isaac_world-local, Z-up meters/radians) root pose, linearly interpolated from the recorded path samples |
| `speed` | instantaneous root speed, m/s (central difference) |
| `groundSlope` | dz/dx of terrain under the root's current X (0 flat, `step_height_m/step_depth_m` on stairs) |
| `phaseC` | continuous gait phase (monotone, NOT the legacy staircase — see `gaitPhaseLegacy`) |
| `support` | lateral weight signal in `[-1,+1]`, +1 = fully on LEFT foot |
| `gaitPhaseLegacy` | the OLD staircase `gaitPhase` field (jumps 0.5 per liftoff, frozen between) — kept for back-compat cross-reference only, not for gait-quality judgments |
| `leftFoot`, `rightFoot` | `{x,y,z,yaw,planted,swingU,liftAt,landedAt,nextLiftAt,strideLen}`, P-frame. `liftAt`/`landedAt`/`nextLiftAt` are `null` when not applicable (e.g. `landedAt` before the first-ever event); `strideLen` is the horizontal `\|to-from\|` of the current/most-recent swing (0 before the first) |
| `cane` | `null` if `params.caneEnabled===false` or the loaded schedule predates the v2 cane machinery; else `{x,y,z,planted,swingU,liftAt,landedAt,nextLiftAt}` (P-frame TIP position — no `strideLen`/`yaw`, not part of the cane's own `poseAt` contract) |

### `bones` — P-frame (isaac_world LOCAL, Z-up) world positions, meters

Each value is `{x,y,z}`, obtained via `bone.getWorldPosition()` (true Three.js
scene-space, Y-up) then `isaacWorldNode.worldToLocal(...)` — **AGENTS.md
incident #5's diagnostic pitfall**: raw scene-space coordinates under
`isaac_world` are already rotated -90° about X (Z-up→Y-up), so comparing
scene-space `.z` directly against this pipeline's native Z-up convention
(terrain heights, `pose.*.z`, etc.) is apples-to-oranges — every position in
this trace has ALREADY been converted back, so it's directly comparable to
`pose.*`/`terrain.*` values without further conversion.

Bones recorded (Mixamo colon-free names per AGENTS.md incident #3; keys match
`PatientHuman.js`'s own `BONE_NAMES`/`_bones` map keys):
`hips`, `spine2`, `head`, `leftArm`, `rightArm`, `leftHand`, `rightHand`,
`leftFoot`, `rightFoot`, `leftToeBase`, `rightToeBase`.

### `pelvisOrientDeg` / `spine2YawDeg` — orientation, degrees, P-frame axes

Computed by decomposing the bone's WORLD quaternion into `isaac_world`'s own
local basis (`qP = isaacWorldNode.getWorldQuaternion()⁻¹ · bone.getWorldQuaternion()`),
then `THREE.Euler.setFromQuaternion(qP, 'ZYX')`:

- `yawDeg` = rotation about the P-frame **Z** (up) axis
- `pitchDeg` = rotation about the P-frame **Y** (lateral) axis
- `rollDeg` = rotation about the P-frame **X** (forward) axis

`spine2YawDeg` is the same computation's `yawDeg` component only, for
Spine2-vs-pelvis counter-rotation analysis (spec §6 item 5's "yaw
counter-rotation ≈ −0.6·pelvisYaw").

**CAVEAT (important, do not skip):** this is the bone's RAW achieved world
orientation, expressed in the P-frame basis — it is **NOT zero at rest**. Xbot
ships in the standard Mixamo/glTF convention (up=local Y, forward=local Z),
and `PatientHuman.js`'s `B_PLACEMENT` constant is a FIXED rotation reconciling
that convention with the P-frame's own (up=Z, forward=X, lateral=Y) — see
AGENTS.md incident #4. That reconciliation rotation is baked into every raw
`pelvisOrientDeg`/`spine2YawDeg` reading here; there was no attempt to strip it
out (doing so would mean reaching into `PatientHuman.js`'s private, unexported
`B_PLACEMENT` constant, which this recorder deliberately does not touch — see
"What this recorder does NOT do" below). **Compare relative values or ranges
across the trace (or against the trace's own first idle sample as a baseline),
not against an assumed zero.** For an already-bind-relative measure of pelvis/
spine dynamics, prefer `lastSync.pelvisListRad`/`pelvisYawRad`/`torsoPitch`/
`spineYawCounter` (see below) — those are the pipeline's own COMMANDED angles,
clean and bind-relative by construction, and pair naturally with this raw,
ACHIEVED cross-check (the same "commanded vs. achieved" split that caught the
AGENTS.md incident #15 IK bug: an internally-consistent re-derivation isn't as
trustworthy as the real rendered bone transform).

### `cane`

- `tip`: `pose.cane`'s own P-frame `{x,y,z}` (`null` if `pose.cane` is `null`)
- `handleTarget`: `patientHuman._lastSync.caneHandleTargetWorld` (P-frame,
  despite the "World" name — see `lastSync` note below) — the NOMINAL
  pre-reach-clamp target
- `handleEffective`: `patientHuman._lastSync.caneHandleEffectiveWorld` — the
  target actually handed to the right-arm IK this frame (differs from
  `handleTarget` only while `lastSync.caneReachClamped` is true)

Both `null` whenever `pose.cane` is `null` (no v2 cane schedule loaded).

### `terrain`

`terrain.heightAt(x)` (meters, P-frame Z) evaluated at three X positions:
`underRoot` (at `pose.rootX`), `underLeftToe`/`underRightToe` (at that sample's
own `bones.leftToeBase.x`/`bones.rightToeBase.x`). Directly comparable to
`bones.leftToeBase.z`/`bones.rightToeBase.z`/`pose.rootZ` (same P-frame Z-up
convention, no further conversion needed) — e.g. a stance-foot penetration
check is simply `terrain.underLeftToe - bones.leftToeBase.z > 0`.

### `lastSync`

A shallow clone of `patientHuman._lastSync` taken immediately after this
sample's `sync()` call — i.e. exactly what `patientDiag` itself reads. Field
names/meanings are **whatever `PatientHuman.js` happens to expose** (this
recorder does not invent or rename anything here); as of this writing that
includes (non-exhaustive — read `js/PatientHuman.js`'s own `_lastSync = {...}`
assignment for the authoritative, current list):

`leftAnkleTargetWorld`, `rightAnkleTargetWorld` (IK targets, P-frame despite
the "World" name — same convention as `caneHandleTargetWorld` above: these are
computed analytically IN the P-frame per I5, "World" here means "the rig's own
world-equivalent math frame", not Three.js scene space), `leftKneeBendDeg`,
`rightKneeBendDeg`, `leftPlanted`, `rightPlanted`, `speed`, `phaseC`,
`support`, `torsoPitch`, `pelvisListRad`, `pelvisYawRad`, `pelvisBobM`,
`pelvisShiftM`, `spineYawCounter`, `spineLateralLean`, `breathing`,
`leftFootRollPitchDeg`, `rightFootRollPitchDeg`, `leftToePitchDeg`,
`rightToePitchDeg`, `leftHeelStrikeActive`, `rightHeelStrikeActive`,
`leftFootContact`, `rightFootContact` (`{x,y,z,mode}`, P-frame — the
roll-model's fixed heel/flat/toe contact point, see AGENTS.md incident #14),
`leftToeOffActive`, `rightToeOffActive`, `armSwingLeftDeg`,
`armSwingRightDeg`, `armSwingRightAvailable`, `caneAvailable`,
`caneHandleTargetWorld`, `caneHandleEffectiveWorld`, `caneReachClamped`,
`caneLeanDeg`, `ikSelfCheckFailed`.

`lastSync` is `null` only if `patientHuman._lastSync` itself was falsy at
sample time (should not happen mid-sweep — `sync()` always sets it — but
guarded rather than assumed, per this codebase's "never guess, degrade
visibly" convention).

## What this recorder does NOT do

- No pass/fail judgment, no bars, no violations list — that's `patientDiag`'s
  job (`audit/run_browser_audit.md`). This is raw measurement only.
- No gait/rig code changes, no new bone lookups beyond what was already
  reachable via `patientHuman._bones`/`.anchor` — purely additive to
  `js/main.js`.
- Does not read/strip `PatientHuman.js`'s private `B_PLACEMENT` constant (see
  the orientation caveat above) — file-ownership boundary per
  `IK_OVERHAUL_SPEC.md` §2 (`js/main.js` is VERIFY-owned; `PatientHuman.js` is
  RIG-owned, read-only for this recorder).
- Does not call `renderFrame()` — bone world transforms are read via
  `Object3D.getWorldPosition()`/`getWorldQuaternion()`, which internally force
  an ancestor-chain matrix update (`updateWorldMatrix`) without needing a full
  scene render, exactly like `patientDiag` already relies on.

## Auto-run hook (headless driver entry point)

`js/main.js`'s boot sequence checks `?gaittrace=1[&dt=<seconds>][&name=<n>]` on
the page URL AFTER `window.__viewer.ready` resolves (the same "GLB + patient
fully loaded" moment `loadRealModel().then(...)`'s own `resolveReady()` call
marks), runs `gaitTrace({dt, name})`, and sets `document.title = 'TRACE_DONE'`
on success or `'TRACE_ERROR: <message>'` on failure. See
`audit/run_browser_trace.py`/`audit/run_browser_trace.md` for the headless
Chrome driver that uses this. The title is a convenience signal only — the
driver's real completion check is the `diag/<name>.json` file landing on disk.
