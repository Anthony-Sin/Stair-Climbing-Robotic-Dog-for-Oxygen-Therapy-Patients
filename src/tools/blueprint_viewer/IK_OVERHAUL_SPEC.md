# Patient IK/Gait Overhaul — Design Spec & API Contract

Branch: `worktree-ik-overhaul` (isolated git worktree — never touch `main`).
Orchestrator-owned document. Implementation agents: read this FIRST, then your own
section. The repo-root `CLAUDE.md` and this directory's `AGENTS.md` (incident ledger)
are BINDING — cite incident numbers when your change touches one.

## 0. Mission

Make the procedural human (elderly oxygen-therapy patient, Xbot rig) walk, hold a
cane, and swing arms REALISTICALLY in the three.js blueprint viewer. Current system
is mechanically correct (deterministic, scrub-safe, zero foot-skate by construction)
but visually robotic. Verified problems to fix:

- P1 `gaitPhase` is a STAIRCASE (jumps 0.5 at each liftoff, frozen between). It
  drives the canned Mixamo "walk" clip → the upper body/arms SNAP once per step and
  freeze between steps. (PatientGait.js `_fillPhaseTimeline`, PatientHuman.js sync()
  step 6/7.)
- P2 The anchor bob `bobAmplitude * sin(4π·gaitPhase)` is ALWAYS ZERO (phase only
  takes values k/2 → sin(2πk)=0). Dead feature — the pelvis never bobs.
- P3 Between steps BOTH feet stay planted while the root glides forward
  (at the follow clip's ~0.26 m/s, ~40-50% of walk time is a two-feet-glued glide)
  → the "skating/shuffling, feet placed wrong" complaint.
- P4 No heel-strike / foot-flat / heel-off / toe-off roll. Feet land and leave
  flat, with only a tiny generic sine pitch (±~7°) mid-swing.
- P5 No pelvis dynamics: no lateral weight shift, no pelvic list/yaw. Anchor yaw
  == root yaw exactly.
- P6 Upper body = Xbot's canned clip layered under overrides. Three separate
  incidents (#5, #7, #10) were caused by fighting that clip. Arms have no
  relationship to the cane; no cane in the rig layer at all.
- P7 When idle the patient freezes ROCK SOLID (mannequin). Needs subtle breathing
  / life that never moves the feet.

## 1. Non-negotiable invariants (from AGENTS.md/CLAUDE.md — verified properties)

- I1 DETERMINISM/SCRUB-SAFETY: `poseAt(schedule, terrain, t)` and everything
  downstream must remain a pure function of (schedule, terrain, t). No
  Math.random, no Date.now, no memo of last-t. Querying t=27.3, 5.0, 27.3 returns
  bit-identical results. `buildSchedule` remains the ONE stateful forward march.
- I2 PLANTED FEET NEVER MOVE: a planted foot's pose is a constant lookup
  (structural zero drift). The new foot-ROLL animates foot PITCH about a FIXED
  heel/toe contact point during landing/pre-lift windows — the CONTACT point must
  stay bit-identical; the ankle may move only via analytic rotation about it.
- I3 IDLE = NO STEPS, NO FOOT MOTION: idle gates in buildSchedule stay. Breathing
  and other idle life may only touch spine/chest/shoulders/head/arms — never feet,
  never the anchor XY/yaw. (`patientDiag` bars: idleFootMotionMax ≤ 0.002 m,
  plantedDriftMax ≤ 0.01 m must keep passing.)
- I4 PatientGait.js stays ZERO-IMPORT pure math (Node-testable, no THREE/DOM).
- I5 PatientHuman.js never reads matrixWorld/worldToLocal mid-sync — analytic
  quaternion math only (existing convention, see sync()).
- I6 SkinnedMesh rules: anchor stays a sibling of patient_root (AGENTS.md #1),
  frustumCulled=false (#2), colon-free bone names (#3).
- I7 Two-bone IK solver + B_PLACEMENT + foot flat-orientation math are VERIFIED —
  reuse, don't rewrite. Generalizing `_solveLegIK` to take a rest-direction/for
  arms is allowed (keep the leg path's behavior bit-identical where inputs are
  unchanged).
- I8 Xbot's canned "walk" AnimationClip is RETIRED by this overhaul. Delete the
  mixer/phase-offset machinery rather than layering on top of it (incidents
  #5/#7/#10). Hips position AND rotation still get explicitly set every sync()
  (we own them now).
- I9 Durations in SECONDS (CLAUDE.md 8.6). Angles in radians internally, degrees
  in diagnostics.
- I10 If a "safe disable" guard turns a feature off (e.g. no cane schedule →
  no cane), console.log ONCE at boot what is off and why (CLAUDE.md 8.8).

## 2. Architecture (files & ownership)

| File | Owner agent | Role |
|---|---|---|
| `js/PatientGait.js` | GAIT | scheduler + signals (pure math) |
| `js/PatientHuman.js` | RIG | bones: legs, feet+roll, pelvis, spine, arms, head, cane grip |
| `js/PatientCane.js` (NEW) | RIG | cane mesh build + tip/handle pose helper (pure THREE, no DOM) |
| `js/main.js`, `serve.py`, `audit/*` (NEW), `index.html`, `styles.css` | VERIFY | diag extension, reset UI/API, Node harness, endpoints |

Anyone may READ any file. Only the owner EDITS it. Integration wiring in main.js
(e.g. calling new PatientHuman APIs) belongs to VERIFY per contract below; the
orchestrator reconciles small mismatches.

Integration facts (from the code inventory — verified, don't re-derive):
- The ONLY usage site of PatientHuman/PatientGait is `main.js` (hero.js/topple.js
  hide the patient; deck.js/democlips.js only use `window.__viewer`).
- Scrub-driven: every pose update goes through `applyGlobalTime()` →
  `patientHuman.sync(seg.name, local)` (main.js:1591-1608). Autoplay uses the
  same zero-delta path. Unified timeline = follow (23.6 s) + climb (42.73 s).
- The CURRENT cane is a decorative world-space prop built in main.js:1229-1257
  (`patientCane` Group) and re-planted at a fixed hip offset by `updateCane()`
  (main.js:1259-1270). It is NOT in the hand and does not swing. This whole
  block is DELETED (VERIFY) and replaced by the rig-owned cane (§5): PatientHuman
  builds/updates the cane INTERNALLY (PatientCane.js helper), attaches it in
  `attachTo()` as a sibling of the anchor under isaacWorldNode, and poses it in
  `sync()` — main.js needs zero per-frame cane code afterwards.
- main.js DIRECTLY READS these PatientHuman fields — DO NOT rename/remove:
  `.ikSelfCheckFailed`, `._attached`, `._patientRootNode`, `._schedules`,
  `._terrain`, `._bones`, `._lastSync`, `._gaitParams`, `.anchor`, plus methods
  `load/attachTo/buildGait/sync`. Changes must be ADDITIVE.
- `patientDiag({dt})` lives at main.js:2294-2512 and already sweeps both clips
  via the real sync() path with state save/restore — extend it, keep its shape.
- `serve.py` (port 8741, launch.json runs it) already has POST /shot; follow
  that pattern for /diag. GETs serve the tool root with no-cache.
- No reset exists. Primitives: `applyGlobalTime(t)` / `jumpToSegment(name)` /
  `window.__viewer.scrub(pct)`. NOTE: `.viewer-controls { display:none }` in
  styles.css:891 — the deck hides manual controls, so the reset button must be
  separate minimal stage chrome (or shown via existing patterns), not buried in
  the hidden control bar.
- `PATIENT_HIP_HEIGHT_M = 0.92` is duplicated in main.js:1199, PatientGait.js:175,
  PatientHuman.js:55, pipeline/anim_bake.py:61 — do not change its value.

## 3. PatientGait.js v2 — API contract (GAIT implements, RIG+VERIFY consume)

`buildSchedule(samples, terrain, params)` — same signature. New DEFAULT_GAIT_PARAMS
keys (all overridable): see §4 tunables. Schedule gains `caneEvents` (array, same
event shape as foot events + `side: 'cane'`), built by the SAME march (see §5).

`poseAt(schedule, terrain, t)` returns (superset of today — existing keys keep
exact meaning; nothing existing is renamed):

```js
{
  rootX, rootY, rootZ, rootYaw, speed, groundSlope,   // unchanged
  gaitPhase,                                          // unchanged (legacy staircase; patientDiag uses it)
  phaseC,          // NEW: continuous phase. CORRECTED CONTRACT (2026-07-10): phaseC is its
                   // OWN clean counter, NOT tied to legacy gaitPhase values — measurement
                   // showed the legacy staircase is non-monotone (0.5, 0.0, 1.5, 1.0, ...
                   // when the right foot steps first), so boundary-equality is impossible
                   // for a monotone signal. Definition: with events merged in tLift order,
                   // phaseC(t) = 0.5*(events fully landed by t) + 0.5*(linear progress of
                   // the active swing, if any). Monotone, frozen when no swing is active,
                   // advances exactly 0.5 per event. Legacy gaitPhase stays EXACTLY as-is
                   // (zig-zag included) for back-compat — do not "fix" it.
  support,         // NEW: lateral weight signal in [-1,+1]. +1 = weight fully on LEFT
                   // foot, -1 = fully on RIGHT. During a left-foot swing → -1 (weight on
                   // right); right-foot swing → +1; between swings ease (smoothstep over
                   // supportEaseSec) from the last value toward 0 (centered stand).
                   // Function of the event lists + t only.
  leftFoot / rightFoot: {
    x, y, z, yaw, planted, swingU,                    // unchanged
    liftAt,     // t of current swing's liftoff (null when planted)
    landedAt,   // tLand of the event that planted this foot (null before first event)
    nextLiftAt, // tLift of this foot's NEXT event at or after t (null if none)
    strideLen,  // horizontal |to-from| of current/most-recent swing (0 before first)
  },
  cane: {          // null if params.caneEnabled === false
    x, y, z,       // TIP position, P-frame (z from terrain at plant; arc during swing)
    planted, swingU, liftAt, landedAt, nextLiftAt,
  },
}
```

Notes:
- phaseC MUST kill the staircase: a 60 Hz sweep during steady walking shows
  max per-sample |Δ walkSignal| bounded (no 0.5 jumps); frozen while idle.
- support freezes feet-agnostic: derived only from event windows (deterministic).
- Foot events never overlap L/R (existing invariant) → phaseC is well defined.
- Keep the walk-on tail working (cane gets walk-on steps too — simplest: cane
  event synthesized per walk-on left-foot step, same pattern as §5).

## 4. Gait realism upgrades in buildSchedule (GAIT)

Target look: slow, careful, but FLUID elderly walk. Recorded speeds: follow clip
~0.26 m/s, climb ~0.11 m/s + stairs.

- G1 REDUCE THE GLIDE (P3): trigger steps PREDICTIVELY. When computing `need`,
  add the drift the root will accumulate over the next `predictLeadSec` (default
  0.6·swingDur): i.e. evaluate drift of planted foot vs nominal at
  `t + predictLeadSec` (sample the array forward — never extrapolate). Tune with
  stepTrigger so steady-state double-support fraction lands in 0.25–0.45 at the
  follow clip's speed (Node metric M5) instead of today's ~0.5+, and the root
  never travels more than ~0.20 m with both feet planted while non-idle (M6).
  Keep idle gates exactly as-is (I3). Adjustment-step behavior on yaw must stay.
- G2 SPEED-ADAPTIVE SWING: swingDur scales mildly with need/speed —
  `swingDur_eff = clamp(swingDur * (refSpeed/max(speed, 0.05))^0.25, swingDur,
  swingDurSlowMax)` (defaults: refSpeed 0.4, swingDurSlowMax 0.55 flat / 0.7
  climb). Slower body → slower, more deliberate steps. Deterministic (speed read
  from samples at trigger time).
- G3 OUT-TOEING: plant yaw = sample yaw ± outToeRad (default 0.10 rad, toes-out,
  left −, right + ... sign such that toes point AWAY from midline; document the
  sign). Applied to nominal + touchdown yaw so feet stand slightly splayed.
- G4 STEP WIDTH: footLateral stays measured-from-rig, add stanceWidenM (default
  0.012 m) — elderly slightly wider stance.
- G5 CANE SCHEDULE (§5).
- G6 phaseC + support + per-foot timing fields (§3).
- G7 Keep/extend the header self-test notes; do NOT break the existing exported
  function signatures used by PatientHuman.buildGait (additive only).

## 5. Cane model

Physical: overall length so the handle sits near greater-trochanter height
(~0.92 m hip) — tip-to-handle caneLengthM default 0.90. Held in the RIGHT hand.
3-point pattern for a single-cane user: the cane advances WITH (slightly leading)
the CONTRALATERAL (LEFT) foot's swing, plants before/with left touchdown, bears
load during left stance.

GAIT side (caneEvents): for each LEFT-foot event E, synthesize one cane event:
`tLift = E.tLift − caneLeadSec` (default 0.08, clamped ≥ previous cane land +
0.05), `tLand = min(E.tLand − 0.02, tLift + caneSwingDur)` (default 0.30; the cane
must finish planting no later than the foot). Target tip XY = nominal at the
cane's own landing time: root(tLand) + yaw-rotated offset (forward caneForwardM
default 0.18, lateral caneLateralM default 0.28 on the RIGHT side), z =
terrain.heightAt with the same stair-snap margins as feet (treat tip footprint as
a point + 2 cm margin both sides). Swing z: same endpoint-blend + arc + clamp
machinery as feet (clearance caneClearanceM default 0.05; reuse the helpers —
refactor into a shared internal function rather than copy-paste). Before the
first left event / when idle: planted at its initial nominal. Walk-on: one cane
event per walk-on left step, same rule.

RIG side: cane = THREE.Group built in PatientCane.js (blueprint-style: shaft
cylinder ~11 mm radius, offset-T/derby handle ~0.11 m, small rubber tip ~2 cm,
theme-tinted material passed in like tintMaterial). Attached under the SAME
isaac_world parent as the anchor (a sibling — NOT parented to the hand; the HAND
is IK'd to the cane, so cane placement stays terrain-exact and the arm absorbs
error). Per-frame: tip at pose.cane position; shaft axis: tip→handle direction =
mostly +Z with forward lean `atan2(handleOffset)`: when planted, handle sits
caneHandForwardM (0.05) ahead of tip and follows the WRIST height model: handle z
= tip z + caneLengthM·cos(lean) etc. Keep it simple: compute handle = tip +
axis·caneLengthM where axis interpolates: planted → (0.06, 0, 1) normalized in
facing frame (slight forward lean); mid-swing → lean forward up to ~18°. RIGHT
hand: two-bone arm IK (Arm→ForeArm→Hand) targets the HANDLE grip point; wrist
(Hand bone) oriented so palm faces down the shaft (grip axis alignment —
document the axis mapping you measure from the bind pose, don't guess signs:
incident #4 discipline). Fingers: static curled grip pose on the right hand set
once at load (per-finger-bone fixed local rotations); left hand: relaxed slight
curl. If the arm IK can't reach (handle too far), log once and clamp — the cane
tilts toward the hand rather than the arm hyper-extending.

## 6. PatientHuman.js v2 (RIG) — bones written per sync(), all procedural

Order of application (single pass, no mixer):
1. Hips position = bind + bob + weight-shift lateral offset (see below); Hips
   quaternion = pelvisList∘pelvisYaw (small: list = supportSign·pelvisListRad
   about forward axis, yaw = pelvisYawRad·swingDirection about up axis; defaults
   list 0.05 rad, yaw 0.06 rad, both scaled by min(1, speed/0.3) and by phase
   signals so they freeze/settle at idle). LEG IK MUST COMPENSATE: hip pivots
   move with Hips — recompute pivot positions analytically under the new Hips
   transform and write UpLeg locals as HipsQ⁻¹·(anchor-desired) (children of
   Hips). Verify via the existing load-time FK self-check EXTENDED to run with a
   nonzero test pelvis rotation (bar stays ≤ 1 cm).
2. Legs: existing two-bone IK vs ankle targets (unchanged), with roll-modified
   ankle targets from §6b during landing/heel-off windows.
3. Feet: flat-at-yaw base (existing verified math) + ROLL PITCH profile §6b +
   out-toe already in yaw from GAIT.
4. ToeBase: articulate: heel-off window → toe extension (dorsiflex at MTP) up to
   toeOffToeRad 0.35 rad so toes stay flat while heel rises; swing → slight
   relax droop (−0.08 rad); else identity. (Replaces blanket identity — safe
   because WE own the full chain now; cite incident #10's cancellation trick if
   composing with knee angles.)
5. Spine chain (Spine, Spine1, Spine2): pitch = existing lean model (keep
   gains); ADD yaw counter-rotation ≈ −0.6·pelvisYaw distributed across
   Spine1/Spine2; ADD lateral lean ≈ 0.4·support·pelvisList toward stance side +
   caneLoadLean 0.02 rad toward the cane while it's planted and bearing (left
   stance); ADD breathing: +breathPitchRad·sin(2π·breathHz·t) distributed on
   Spine1/Spine2 (defaults 0.008 rad, 0.27 Hz) — ALWAYS on (walk + idle).
6. Shoulders/arms:
   - LEFT (free) arm: FK swing. Drive signal = CONTRALATERAL leg advance:
     `adv_R(t) = clamp(((rightFoot.x,y) − root)·facing_fwd / 0.35, −1, 1)`
     (planted or swinging — it's continuous). Shoulder (Arm bone) pitch =
     armSwingRad·adv_R (default 0.22 rad ≈ 12.5°, elderly-small), slight
     abduction armAbductRad 0.10 so the sleeve clears the sweater/hip; ForeArm:
     elbowBaseRad 0.35 + elbowSwingRad·max(0, adv_R)·0.3 (bends more swinging
     forward, straighter behind); Hand: relaxed. At idle adv→frozen ✓ plus
     breathing micro-sway only.
   - RIGHT (cane) arm: two-bone IK to the cane handle (§5). Shoulder bone gets a
     small depression/elevation with cane load (±0.02 rad, support-driven).
   - Arm rest pose must NOT clip the widened torso: verify hand/forearm-to-body
     clearance numerically (VERIFY M12) — tune abduction if needed.
7. Head/Neck: counter-pitch so the head stays level vs spine lean (≈ −0.7·total
   spine pitch), counter-yaw ≈ −0.5·(pelvisYaw+spineYaw) (gaze stays on path),
   plus gazeDownRad 0.10 (watching the ground ahead — elderly, careful). Add
   headBobDamp: head world vertical excursion should come out < pelvis bob
   (emergent from counter-pitch; measure, don't force).
8. Pelvis vertical BOB (fixes P2): z += bobAmplitude·(−cos(2π·(2·phaseC)))·0.5
   — i.e. lowest at each support transfer (phaseC ≈ k/2, where steps happen),
   highest mid-step; bobAmplitude default 0.018 m; multiplied by
   min(1, speed/0.25) so it fades at creep speeds and freezes at idle (phaseC
   frozen ⇒ bob frozen; the multiplier just kills residual offset at stand).
   Weight-shift lateral: hipShiftM·support (default 0.025 m) applied in the
   FACING frame (lateral axis), added to Hips position (NOT the anchor —
   feet must not move; verify plantedDrift stays 0).
9. Diagnostics: extend `_lastSync` with: phaseC, support, torsoPitch, pelvis
   list/yaw applied, arm swing angles L/R, cane handle target + achieved hand
   position error, per-foot roll pitch, toe-off/heel-strike state flags.

### 6b. Foot roll model (fixes P4) — stateless from event data

For each foot at time t (all windows clamped inside the planted interval;
missing data (nulls) ⇒ flat foot — never guess):
- HEEL-STRIKE→FOOT-FLAT: for t ∈ [landedAt, landedAt+rollDownSec (0.12)]:
  foot pitch from +heelStrikeRad (default 0.14 rad ≈ 8°, dorsiflexed, heel down
  toe up) easing to 0 (smoothstep). Ankle target pivots about the HEEL contact:
  heelPoint = plant pos − facing·heelBackM (measure heel offset from rig:
  ankle-to-heel horizontal ≈ toe measurement's backward analog; if not
  measurable, heelBackM default 0.06) — ankle = heelPoint + R(pitch)·(bind heel→ankle offset).
  Scale heelStrikeRad by min(1, strideLen/0.25) — tiny shuffle steps don't heel-strike.
- MID-STANCE: flat (existing behavior).
- HEEL-OFF→TOE-OFF: for t ∈ [nextLiftAt−heelOffSec (0.18), nextLiftAt]: pitch
  from 0 to −toeOffRad (default 0.30 rad ≈ 17°, plantarflex, heel up) smoothstep;
  ankle pivots about the TOE contact point (toe stays planted — this is where
  ToeBase extension §6.4 kicks in, keeping toes flat). Scale by
  min(1, strideLen_next... use the NEXT event's strideLen if accessible else
  speed factor min(1, speed/0.15)).
- SWING: keep the current profile but connect ends: start at −toeOffRad·0.6
  (continuing the push-off), mid dorsiflex to +0.10, land at +heelStrikeRad
  (matching the landing window start — NO pitch pop at tLand or tLift; assert
  continuity numerically in the harness M9).
- The CONTACT-POINT pivot means the ankle IK target changes during roll windows
  while the CONTACT stays fixed (I2). Toe-vs-ground clearance must stay ≥ −2 mm
  (M8 bar) — the previous per-frame analytic drop logic (incident #14) is
  superseded by this explicit contact-point model; keep its lesson: derive the
  full chain analytically at the CURRENT angles, never a bind-pose constant.

## 7. Reset + living-room/home scene (VERIFY; details from the exploration report)

- `window.__viewer.resetDemo()` (and a small UI button consistent with existing
  viewer chrome): rewind the unified demo timeline to t=0 & restart autoplay;
  also reset any per-scene patient state. Must be idempotent and work mid-scrub.
- If the home/living-room scene drives PatientHuman from a synthetic route, it
  uses the same buildSchedule machinery — confirm it picks up all new features
  (cane, arms) with zero scene-specific code beyond wiring.

## 8. Verification harness (VERIFY) — "don't trust screenshots"

### Node tier — `audit/gait_audit.mjs` (+ `audit/extract_tracks.py` if needed)
Replays PatientGait v2 headlessly (it's zero-import). Input: patient_root tracks
from robot.glb — either parse GLB minimally in Node (JSON chunk + accessor reads;
~80 lines, no deps) or a tiny Python extractor to JSON using the pipeline's
pygltflib tooling. Fixture: also a synthetic constant-speed path + stop-and-go
path + zig-zag (unit-style). Metrics (report JSON + pass/fail vs bars):
- M1 stepLength distribution (median ∈ [0.15, 0.45] m at follow speed)
- M2 cadence (steps/min) and L/R alternation ratio (≥ 0.9 strict alternation
  during steady segments)
- M3 duty factor per foot ∈ [0.55, 0.8] during steady walking
- M4 foot penetration below terrain: max ≤ 1e-6 m (structural)
- M5 double-support fraction during non-idle ∈ [0.2, 0.5]
- M6 max root travel while both feet planted & non-idle ≤ 0.20 m (kills P3)
- M7 phaseC: monotone; max |Δ| per 1/60 s sample ≤ 0.04 (no staircase);
  frozen (Δ=0) across idle windows; equals gaitPhase at event boundaries
- M9 cane: tip penetration ≤ 1e-6; planted fraction while root idle = 1.0;
  cane leads/coincides with left-foot swings (timing correlation), tip stays
  within 0.55 m HORIZONTAL (XY) range of the root (CORRECTED 2026-07-10: the
  original "tip within caneLengthM of the hip" was geometrically impossible —
  a ground-planted tip is always ≥ hip height ≈ 0.92 m from the hip in 3D)
- M10 idle: zero foot/cane motion across all idle windows (existing bar)
- Baseline mode: run the SAME metrics against `git show HEAD:...PatientGait.js`
  (write to audit/baseline/) so before/after is quantified in the final report.

### Browser tier — extend `main.js` `patientDiag` (+ `window.__viewer.gaitAudit()`)
Everything M-numbered that needs the RIG: sweep both clips at dt=0.05 like today:
- M8 toe/heel mesh vs ground/tread: min clearance ≥ −0.002 m, and during roll
  windows the CONTACT point (heel or toe) moves ≤ 0.005 m
- M9b foot pitch continuity: max per-sample |Δpitch| ≤ 0.12 rad at 20 Hz
  (no pops at lift/land)
- M11 arm swing: shoulder pitch amplitude during steady walk ∈ [0.1, 0.45] rad;
  contralateral phase: corr(shoulder_L, adv_R) ≥ +0.7 (sign convention: define
  and assert, don't hand-wave); at idle amplitude ≤ 0.01 rad (breathing only)
- M12 clearances: hand/forearm-to-thigh/torso min distance ≥ 0.02 m; cane
  hand-to-handle error ≤ 0.015 m every sampled frame; cane shaft never
  intersects the body/legs (segment-to-capsule distance ≥ 0.03 m vs shin/thigh)
- M13 pelvis: bob amplitude during steady walk ∈ [0.008, 0.035] m; ZERO at
  idle; head world vertical amplitude < pelvis amplitude (stabilization works)
- M14 fkErrorMax, plantedDriftMax, idleFootMotionMax, kneeBend stance median —
  existing bars keep passing
- keep the whole thing exposed as `window.__viewer.patientDiag({dt})` returning
  ONE JSON with a top-level `pass: boolean` + per-metric {value, bar, pass}
- serve.py: add `POST /diag` (like /shot) so a driver can save the JSON to disk;
  add `audit/run_browser_audit.md` documenting the 3-command flow (serve, open,
  save) for humans.

### Visual tier (orchestrator runs): saveShot series at fixed clip times +
per-phase; before/after comparison. Screenshots are for LOOK judgment only —
numbers decide correctness (AGENTS.md recurring lesson).

## 9. Tunables (single source of truth: DEFAULT_GAIT_PARAMS + a new
PATIENT_BODY_PARAMS in PatientHuman.js)

All new constants above get named params with the defaults given. Keep units in
names or comments (M/Sec/Rad suffixes). No magic numbers inline.

## 10. Definition of done

1. All Node + browser metrics pass on BOTH clips (follow, climb) — and on the
   home-scene route if applicable.
2. Existing acceptance bars keep passing (no regression).
3. Before/after metric table + screenshot set produced.
4. No console errors; 60 fps not measurably regressed (sync() stays
   allocation-light — reuse scratch objects; no new per-frame allocations in
   hot paths beyond a handful of vectors).
5. Reset works from any scrub position.
