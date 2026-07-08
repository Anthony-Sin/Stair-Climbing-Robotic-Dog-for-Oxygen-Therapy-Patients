# Blueprint viewer — local agent notes

Scoped to `src/tools/blueprint_viewer/`. See the global `CLAUDE.md` for repo-wide rules;
this file only covers gotchas specific to this pipeline/viewer.

## Incident Ledger

### 1 — glTF skinned-mesh nodes must sit at a CONSTANT-transform ancestor
- **TRIGGER:** Adding/editing a `THREE.SkinnedMesh` in this pipeline (currently
  `js/PatientHuman.js`'s imported human model; the pipeline's OWN hand-built skinned
  limbs this incident originally documented were removed 2026-07-07 in favor of that
  imported model — see incident #4 — but the constraint applies to any SkinnedMesh).
- **LESSON:** A glTF/three.js `SkinnedMesh`'s own NODE transform is captured ONCE at
  bind/load time (`bindMatrix`) and held fixed forever — three.js never re-derives it
  per frame. All visible motion must come from the skeleton's bone nodes, which is
  fine IF the mesh node's own transform (and every one of ITS ancestors) truly never
  animates. This is why `PatientHuman`'s anchor is a sibling of `patient_root` under
  `isaac_world` (manually re-positioned/re-oriented from `patient_root`'s CURRENT
  transform every `sync()` call) rather than an actual child of `patient_root` (which
  is itself animated every frame) — nesting a SkinnedMesh under an animated ancestor
  double-applies that ancestor's motion (once via the mesh node's own transform, again
  via the bones' `matrixWorld`, which already includes that same ancestor chain).
- **WHY:** Worked out by hand from the glTF skinning spec + three.js's actual
  `skinning_vertex.glsl` formula — getting this wrong silently double-transforms every
  skinned vertex, not an error that throws.

### 2 — SkinnedMesh + frustum culling: limbs vanish once the character walks far enough
- **TRIGGER:** Any `SkinnedMesh` in this scene whose own node sits at a FIXED location
  (see #1) while its skeleton moves the character far away from that location (e.g.
  partway up the staircase, metres from world origin).
- **LESSON:** three.js's default per-object frustum cull test uses `geometry.
  boundingSphere`, computed from the RAW (bind-pose) vertex data and transformed by
  the mesh node's OWN `matrixWorld` — NOT the GPU-skinned result. Since a mesh node
  sitting at a fixed anchor location (see #1) never moves, its cull test always checks
  a small sphere near that anchor, regardless of where the bones have actually moved
  the visible geometry. Once the camera is far enough from the anchor that the stale
  bounding sphere falls outside the frustum, the GPU still renders the correctly
  skinned limb — but three.js culls the whole object before it ever reaches the GPU, so
  it silently vanishes (confirmed live twice: once with this pipeline's own former
  skinned limbs, again with the imported Xbot model — both rendered fine near the
  "follow" clip's start-of-route position but disappeared entirely partway up the
  stairs). Fix: `node.frustumCulled = false` on every SkinnedMesh (see
  `PatientHuman.attachTo`'s traverse). Any future skinned mesh added to this viewer
  needs the same treatment (or an explicitly recomputed/expanded bounding volume)
  unless its mesh node itself is animated to track the character (which would
  reintroduce incident #1).
- **WHY:** No error, no console warning — the mesh is just quietly absent from frames
  far from the origin, and present in frames near it, which reads exactly like an
  animation-baking bug rather than a renderer-side culling setting.

### 3 — GLTFLoader strips colons from node names
- **TRIGGER:** Looking up a glTF node by name at runtime (`Object3D.getObjectByName`)
  when the source asset's node names contain a colon (Mixamo rigs universally name
  bones `mixamorig:Hips`, `mixamorig:LeftUpLeg`, etc).
- **LESSON:** three.js's `GLTFLoader` silently drops the colon when it builds each
  `Object3D.name` (`mixamorig:LeftUpLeg` becomes `mixamorigLeftUpLeg` at runtime) even
  though the `.glb`'s own JSON still has the colon in `nodes[i].name`. Confirmed by
  traversing a freshly loaded `Xbot.glb` in the browser console — `getObjectByName`
  with the colon-containing name returned `null` for every bone; the colon-free form
  worked. `js/PatientHuman.js`'s `BONE_NAMES` map uses the colon-free names for this
  reason (with a comment pointing here).
- **WHY:** No warning, no thrown error at load time — just a `null` from
  `getObjectByName` later, which reads like a wrong-bone-name typo rather than a
  loader-normalization quirk. If you inspect the raw `.glb` JSON (e.g. with
  `pipeline/validate_glb.py`'s pygltflib-based tooling) the names still have colons,
  which makes the mismatch even more confusing until you actually traverse the loaded
  three.js scene graph and print real `.name` values.

### 4 — Patient mannequin: imported rig + cross-convention angle retargeting (2026-07-07)
- **TRIGGER:** Touching `js/PatientHuman.js`, `pipeline/anim_bake.py`'s
  `patient_pose` output, or `models/patient_pose.json`'s schema.
- **LESSON:** The patient used to be a hand-built primitive/skinned mannequin baked
  entirely by this Python pipeline (see git history before 2026-07-07) — replaced
  with a real imported+rigged human (`models/vendor/Xbot.glb`, a Mixamo mannequin
  bundled in three.js's own examples repo, see `models/vendor/NOTICE.md`) because the
  primitive body looked bad and visibly clipped at the knee/elbow no matter how the
  geometry was tuned. `pipeline/anim_bake.py` still owns 100% of the leg/torso POSE
  MATH (same IK as before); it just emits plain scalars (`patient_pose.json`) instead
  of glTF quaternion node tracks, since nothing in `robot.glb` needs to be animated by
  them anymore. `js/PatientHuman.js` retargets those scalars onto Xbot's own bones
  every frame, but layers them UNDER Xbot's own canned "walk" `AnimationClip` (arm
  swing / spine sway / hip bob) rather than replacing it outright — i.e. play the
  canned clip first (`mixer.update(0)`), THEN overwrite `LeftUpLeg`/`LeftLeg`/
  `RightUpLeg`/`RightLeg` and add a `Spine` lean from this pipeline's own angles. Do
  NOT just play the canned clip through the climb unmodified — it's a generic
  flat-ground loop with no idea where OUR stairs' risers are, so it clips through/
  floats above the treads; the whole reason the pose stays data-driven is to guarantee
  the feet always land where this pipeline's own IK says the tread actually is.
- **Coordinate reconciliation**: Xbot ships in the common glTF/Mixamo convention
  (up=local Y, forward=local Z, lateral=local X); this pipeline's own world (under
  `isaac_world`) uses up=Z, forward=X, lateral=Y. `PatientHuman.js`'s `B_PLACEMENT` is
  the one fixed rotation reconciling the two — confirmed (not guessed) empirically in
  the browser: dumped Xbot's raw bind-pose bone rotations (all ≈identity — Mixamo's
  own T-pose bakes the character's shape into TRANSLATIONS, not rotations) and its
  "walk" clip's dominant `LeftUpLeg` rotation axis (local X, confirming the sagittal
  hip-flexion axis) before writing any retargeting code, then verified the final
  result with `getWorldPosition`/axis-transform checks plus front/back camera
  screenshots (character faces the same `+X` world direction the robot walks).
- **WHY:** Cross-rig retargeting has no compiler-checkable correctness — a sign or
  axis mistake doesn't throw, it just contorts the character in a way that's easy to
  misread from a single unlucky camera angle (an early debug screenshot looked like a
  bent-double crouch; the actual bone quaternions were all small, natural angles, and
  a front-on shot showed a normal standing pose — the "bug" was the camera angle, not
  the retargeting). Verify with numeric bone-transform dumps before trusting a
  screenshot's first impression.

### 5 — Retargeted feet clipped through ground/treads (user-reported, 2026-07-07): two real bugs behind it
- **TRIGGER:** Any future change to `PatientHuman.js`'s leg/hip placement, or trusting
  a `getWorldPosition()` reading during debugging.
- **LESSON, bug A (hip bob + uncontrolled ankle rotation):** Xbot's canned "walk" clip
  animates MORE than arm swing/spine sway — it also animates `Hips` TRANSLATION (a
  vertical bob dipping ~6cm below the bind-pose height `_hipsHeightM` was calibrated
  from) and, independently, `LeftFoot`/`RightFoot`/`*ToeBase` ROTATION (ankle/toe
  articulation tuned for the canned clip's OWN hip/knee angles). Overriding only
  `LeftUpLeg`/`LeftLeg` (as the first version of this module did) left Hips free to
  bob below the calibrated height AND left the ankle at a rotation totally mismatched
  with this pipeline's own hip/knee angles — a 20-30 degree mismatch at the end of a
  ~0.44m shin moves the toe tip by literally tens of centimeters. Fix: every `sync()`
  call now restores `Hips.position` to its captured bind-pose value (cancels the bob,
  keeps the canned clip's rotation sway) and forces `Foot`/`ToeBase` rotation to
  identity (bind pose) on both sides, so NOTHING but this pipeline's own hip/knee
  angles decides where the leg chain ends up.
- **LESSON, bug B (anatomical ankle-vs-ground mismatch):** Even with bug A fixed, the
  toe still sat ~6-8cm into the floor. Root cause: `anim_bake.py`'s leg IK targets the
  ANKLE at ground/tread level — correct for this pipeline's OLD hand-built rig, whose
  "foot" was a thin box centered right at the ankle joint (ankle at ground WAS sole at
  ground). Xbot is a real anatomical skeleton: `Foot` IS the ankle bone, which sits
  genuinely ~8-9cm above the ground in a natural standing pose, with the sole reached
  only via `ToeBase`'s own translation further down+forward. Landing the ankle exactly
  at ground (as the IK's calibration assumes) therefore buries the real toe mesh in
  the floor by that anatomical gap. Fixed by measuring `footWorld.y - toeWorld.y` once
  at load time (bind pose) and raising the WHOLE anchor by that amount, so the ankle
  sits above ground by exactly enough for the real foot geometry to put the TOE at
  ground/tread level instead.
- **Diagnostic pitfall hit while chasing this**: an early debug pass compared
  `bone.getWorldPosition(tmp); tmp.z` directly against Python's raw (pre-isaac_world-
  rotation) ground/tread height and found an apparently catastrophic, monotonically
  growing 2-metre deviation over the climb. That deviation was ENTIRELY a diagnostic
  bug, not a real one: `getWorldPosition()` returns true three.js scene-space
  coordinates, which for anything under `isaac_world` have ALREADY been rotated -90
  degrees about X (Z-up -> Y-up) — so scene-space `.z` is no longer "up" once you're
  below that node, and comparing it against a raw Z-up expectation is apples-to-
  oranges. The fix for DIAGNOSTICS (not for the app, which never reads world
  positions — it only manipulates local transforms, so this bug never touched what
  users actually see): convert back into `isaac_world`'s own local frame first, e.g.
  `isaacWorldNode.worldToLocal(worldPos)`, before comparing `.z` against anything
  computed in this pipeline's native (raw, pre-rotation) convention.
- **WHY:** All three of these are the same shape of failure: an assumption that held
  for the OLD simplified/stylized rig (rigid foot, ankle-at-ground, no canned
  animation to fight) silently stopped holding the moment a REAL, more detailed rigged
  asset (with its own anatomy and its own animation) was substituted in. Whenever
  swapping in a more "real" asset for a stylized placeholder, explicitly re-derive
  which of the placeholder's simplifying assumptions the new asset actually violates —
  don't assume "it has similar proportions" is the same as "it has the same contract."

### 6 — Patient gait "steps, not walking" / "jumps on stairs" (user-reported, 2026-07-08): a family of "recompute live" bugs
- **TRIGGER:** Any future change to `pipeline/synthetic_motion.py`'s
  `_patient_foot_target`/`_synthetic_patient_pose`, or to a stance/swing gait's
  GaitParams (stride_len, lift_h, cycle_period_s, swing_frac).
- **LESSON:** The patient's ORIGINAL synthetic foot trajectory was a placeholder that
  never lifted the feet at all (a flat sine shuffle glued to the ground, re-snapping
  to `terrain_height(x)` every frame with zero easing) — this produced BOTH reported
  symptoms at once: no vertical lift reads as "sliding/stepping, not walking," and the
  terrain re-snap is a literal instantaneous jump every time the foot's x crosses a
  tread boundary. Replacing it with a real 2-beat stance/swing gait (reusing the
  quadruped's own proven `GaitParams`/arc-height machinery, see `_patient_foot_target`)
  surfaced FOUR DISTINCT instances of the SAME underlying bug shape, found one at a
  time via direct frame-dumps (not screenshots — see incident #4's closing note, which
  applies doubly here since these are ANGLE-space bugs, not position/orientation):
  recomputing some value freshly from the CURRENT live state every frame, instead of
  remembering what was already decided/committed at the last discrete transition.
    1. Height, at liftoff: reading target height as `live_nominal.z + arc(t)` assumes
       "nominal" hasn't moved since the foot was last anchored — false while climbing,
       since nominal tracks the hip, which keeps rising during the OTHER foot's stance.
       Fix: blend from the actual liftoff Z (captured once) to a touchdown Z (also
       decided once, at that same liftoff instant), with the arc added on top --
       `ease(0)=0` and `ease(1)=1` exactly, so neither endpoint can introduce a pop.
    2. Height, mid-swing: even after fixing #1, re-deriving the ARRIVAL height every
       frame from the swinging foot's OWN advancing x (via the terrain clamp) let it
       cross a SECOND tread boundary before the swing finished, double-counting a
       riser. Fix: freeze the arrival height too, at liftoff, via `_next_tread_height`
       (scans forward from x0 for the first HIGHER terrain sample -- robust regardless
       of how far into the current tread x0 already is, unlike a fixed look-ahead
       distance, which over/undershoots depending on stride phase and was tried first).
    3. `_next_tread_height`'s own "current height" baseline: MUST be the raw terrain
       height, not a low-pass-filtered/smoothed reference (see #4 below) -- the
       filtered value LAGS during a transition, which made the scan sometimes conclude
       the CURRENT tread was already "next" and stop at zero distance, freezing the
       touchdown a full riser too low (later "corrected" by a DIFFERENT stance-time
       terrain clamp, producing the exact same class of pop at the opposite end of the
       swing instead).
    4. Horizontal (X), at liftoff: the fore-aft sweep (`_stride_fore_aft`) implicitly
       assumes stride_len exactly equals (forward speed) x (stance duration) -- true
       only by coincidence, and NOT true for `PATIENT_GAIT_FLAT`'s first tuning (0.36 m
       stride vs. ~0.29 m actually covered per stance), which snapped the foot
       backward at every liftoff even on perfectly FLAT ground (no terrain involved at
       all -- pure gait-parameter mistuning). Fixed the SAME way as height: blend the
       FULL 3D position from a liftoff snapshot to a touchdown decided once at that
       liftoff, rather than trusting stride_len to stay perfectly matched to speed.
  A SEPARATE, non-swing-related instance of the same bug shape: the patient's own
  BODY-HEIGHT reference (`ground_z = terrain_height(patient's lead x)`) is a raw step
  function too -- even with the swinging foot's own trajectory fully smoothed, a
  STANCE leg's IK still sees a sudden change if the HIP reference it's measured
  relative to jumps a full riser the instant the (invisible) lead-point crosses a
  tread boundary, while the (fixed, planted) foot hasn't moved. Fixed with a simple
  exponential low-pass filter (`_smooth_ground_z`, ~0.25s time constant) on that one
  shared scalar, used everywhere `pz_ground` feeds into hip placement.
- **WHY:** Every one of these reads as a locally-reasonable simplification in
  isolation ("just use the live/current value, it's simpler") and produces NO error --
  only a discontinuity that's easy to miss in a static screenshot review (see incident
  #4/#5's same warning) and only shows up as a specific complaint once someone watches
  the actual motion ("jumps", "doesn't really walk"). When a value should represent
  "what was decided at the last discrete event" (a footstep, a liftoff), don't
  recompute it fresh from whatever the live/continuous state happens to be this frame
  — snapshot it at the event and hold it fixed until the next event.

### 7 — Patient "hips sag" / erratic forward lean / intermittent floating (user-reported, 2026-07-09): Hips ROTATION was never reset, only position
- **TRIGGER:** Touching `js/PatientHuman.js`'s `sync()`, or anything that layers this
  pipeline's own bone overrides on top of Xbot's canned "walk" `AnimationClip`.
- **LESSON:** `sync()` already restored `Hips.position` to bind pose every frame (see
  incident #5) but left `Hips.quaternion` exactly as the canned clip's `mixer.update(0)`
  set it. `LeftUpLeg`/`RightUpLeg` (this pipeline's own data-driven leg bones) are
  CHILDREN of `Hips`, so the uncontrolled Hips rotation silently re-rotated the whole
  leg chain every frame by whatever the canned clip's own sway happened to be at that
  instant. The canned clip's loop period (~0.97s, `_walkAction.getClip().duration`) has
  no relationship to this pipeline's own physically-computed gait period (~1.1-1.2s,
  `PATIENT_GAIT_FLAT`/`PATIENT_GAIT_CLIMB`), so the two drift in and out of phase
  continuously -- confirmed by a numeric sweep (scrub both clips every 2%, measure
  toe-height-above-ground/tread and a spine world-up-vector lean angle): lean varied
  2.7-9.9 degrees with NO correlation to gait phase (exactly what an uncorrelated
  second rotation source riding the same bones looks like), and toe deviation above
  ground never reached 0 (never actually planted). A SECOND, compounding bug: `Spine`'s
  lean was applied via `quaternion.premultiply(torsoPitch)` -- stacking this pipeline's
  real, physically-meaningful lean ON TOP OF the canned clip's own uncorrelated spine
  sway, instead of replacing it. Fix: `Hips.quaternion.identity()` alongside the
  existing position reset, and `Spine.quaternion.setFromAxisAngle(...)` (replace, not
  premultiply). Re-verified: lean pinned at the correct value everywhere (204 samples,
  both clips), zero ground/tread clipping, worst "float" reading matches legitimate
  mid-swing arc height.
- **WHY:** Same shape as incident #5's bug A (uncontrolled canned-clip motion on a bone
  this pipeline claims to fully own) but on `Hips` instead of `Foot`/`ToeBase` --
  resetting ONE of {position, rotation} on a bone and assuming that's "the pose reset"
  is an easy partial fix to mistake for a complete one, especially since Hips
  POSITION was the one earlier incidents (correctly) flagged; rotation looked
  untouched-and-therefore-safe by comparison.

### 8 — Leg-IK "stance" reach too short: 2-link knee-bend angle is highly nonlinear near full extension
- **TRIGGER:** Tuning `PATIENT_STANCE_TARGET_Z` (synthetic_motion.py) or
  `PATIENT_MAX_REACH_M` (anim_bake.py) for the patient's leg IK.
- **LESSON:** `PATIENT_STANCE_TARGET_Z = -0.80` (original tuning) looked like a
  reasonable "slightly-short-of-full-extension" standing reach against the 0.88 m
  geometric leg length (0.44+0.44 m segments) -- only 9% short. It actually solved to
  a ~49 degree knee bend (a visible squat/sit posture at every stance instant, not a
  standing walk -- user-reported: "doesn't look natural at all", screenshots showing a
  "sitting in an invisible chair" pose), because the 2-link IK's knee-bend angle is a
  HIGHLY NONLINEAR function of reach near full extension: at reach=0.86 m (98% of max),
  the bend is still ~24 degrees, not the few degrees intuition suggests. Fixed by
  raising `PATIENT_STANCE_TARGET_Z` to -0.858 (closer to `PATIENT_MAX_REACH_M`'s 0.86 m
  cap, leaving ~2mm margin so it isn't sitting exactly at the IK's clamp boundary) --
  stance bend now ~24-30 degrees, matching a natural walking gait. Before tuning ANY
  reach/target-Z constant for this solver, compute the actual resulting knee-bend angle
  (law of cosines: `cos(knee_interior) = (2*L^2 - reach^2) / (2*L^2)`) rather than
  reasoning from "how close to max reach, as a percentage" -- percentage-of-max-reach
  is not a useful proxy for the resulting visual bend on this solver.
- **WHY:** The nonlinearity is severe enough that a "conservative-looking" 9%-short
  reach produces a dramatically bent-knee pose, and the bug survived an earlier full
  session (including the incident #4-#6 gait rewrite) because nobody computed the
  actual angle -- the code comment asserting "-0.80 leaves a natural standing knee
  bend" was never numerically checked against what angle -0.80 actually produces (see
  incident 8.7 in the repo-root CLAUDE.md: don't trust a comment's safety/correctness
  claim without citing the numbers it depends on -- this is the same failure shape,
  just in this pipeline's own docs instead of a safety-property comment).

### 9 — Real recorder never logs per-frame patient foot/hip/head data: the "no feet logged" fallback was a FROZEN static pose, not a walk
- **TRIGGER:** Baking `--frames <robot_frames.jsonl>` from a REAL Isaac run (as
  opposed to `--synthetic`), or touching `anim_bake._bake_patient_legs`'s no-
  logged-feet branch.
- **LESSON:** `anim_bake.py`'s docstrings already said the real recorder "does not"
  log `patient['l_foot']`/`['r_foot']` (or `hip`/`head`, so `torso_pitch` is always
  0.0 for real data) -- but the ORIGINAL no-feet-logged fallback was a single static
  IK target (both legs identically "straight down from the hip"), applied EVERY
  frame. That means a real-data bake's patient had its ROOT walk the correct real
  recorded path while its LEGS never moved at all -- a frozen statue sliding across
  the ground, only noticeable once you actually look at consecutive `hip_pitch_l` vs
  `hip_pitch_r` values (they were IDENTICAL at every single frame; a quick scrub in
  the browser doesn't obviously show "frozen legs" the way it obviously shows
  "clipping into the floor"). Fixed by driving the SAME stance/swing gait
  `synthetic_motion.py`'s own patient uses (`_patient_foot_target`,
  `PATIENT_GAIT_FLAT`/`PATIENT_GAIT_CLIMB`) from the REAL per-frame root
  position/yaw/time instead -- this is exactly the architecture the user asked for
  ("use the real patient's x trajectory, don't copy Isaac's own per-frame body pose"
  -- which was never even recorded to copy). Needed a fresh per-clip
  `_PatientGaitState` (stance_anchor/liftoff_pos/touchdown_pos/swinging/
  ground_smooth dicts) threaded through `bake_clip()`'s frame loop, plus a
  `stair_spec` parameter on `bake_clip()` so the real-data climb clip can pick
  `PATIENT_GAIT_CLIMB` and terrain-clamp swings correctly (a real "follow" window
  can ALSO briefly cross onto the first tread near its boundary -- pass `stair_spec`
  to BOTH clips, not just "climb").
- **WHY:** A per-frame-identical L/R angle pair is silent -- no clipping, no crash,
  passes every self-check (hip->ankle distance and ankle-vs-ground bounds are
  satisfied trivially by a static "legs straight down" pose) -- so this shipped
  and looked fine in isolated screenshots (see incident #4/#5/#6's identical warning
  about screenshots) until someone watched the actual real-data playback move.

### 10 — Twisted/"wired" foot on stairs specifically (user-reported, 2026-07-09): identity Foot rotation only looks right at SMALL knee_bend
- **TRIGGER:** Touching `js/PatientHuman.js`'s `sync()` Foot/ToeBase handling, or any
  future gait tuning that lets `knee_bend` grow large (a high-clearance swing, e.g.
  climbing a tall riser).
- **LESSON:** Incident #5 fixed `Foot`/`ToeBase` to `quaternion.identity()` (relative
  to their parent `Leg`/shin bone) as a "safe default" to stop the canned clip's own
  wrong ankle articulation -- verified against MODERATE bends at the time. This is a
  LOCAL rotation, so identity means "Foot's world rotation = Leg's (shin's) world
  rotation": fine when the shin stays close to vertical (small-to-moderate
  `knee_bend`, e.g. flat-ground walking's ~24-50 degrees), but the climb gait folds
  the knee up to ~80-110 degrees to clear a 0.145 m riser (a real, correct angle --
  see incident #9) -- at that bend the SHIN itself is rotated way back, and a foot
  rigidly copying that rotation points almost straight up/backward: a visibly
  twisted/broken-looking ankle on stairs specifically (flat ground has small
  knee_bend, so it looked fine there -- user confirmed "fixed the ground issue but
  not the stairs issue"). Fix: set Foot's LOCAL rotation to `+kneeBend` (Leg's own
  local rotation is `-kneeBend`, both about the SAME axis, `PITCH_AXIS`, so this is
  an EXACT angle cancellation, not an approximation) -- Foot's world rotation becomes
  `hip_pitch` alone, i.e. it tracks the THIGH's angle instead of the shin's,
  mimicking real ankle dorsiflexion that keeps the foot roughly aligned with the
  leg's overall swing direction as the knee folds. Verified numerically (not just
  visually): at the patient_pose.json's peak climb `knee_bend` frame (109 degrees,
  climb clip pct=3.5), `leftFoot`'s world-space local-Y-axis direction now matches
  `leftUpLeg`'s EXACTLY (`(0.824, 0.566, -0.015)` both), vs. the old identity
  approach which would have matched `leftLeg`'s (shin's) very different direction
  (`(-0.802, 0.597, 0.014)`).
- **WHY:** "Identity" reads as the maximally-safe, assumption-free choice, so it's
  easy to trust it stays correct as OTHER numbers (like knee_bend's range) change
  elsewhere in the system -- but identity is only "neutral" in the bone's OWN local
  frame; composed through a rotated parent, it inherits 100% of that parent's swing.
  A fix verified at the angle ranges available when it was written can silently stop
  applying once a LATER change (here: the incident #9 gait-driven real-data legs,
  which pushed knee_bend into a much higher range than the original flat-ground-only
  synthetic gait ever produced) pushes the system into a regime the original
  verification never covered.

### 12 — Real recorder's ground_z is a SMOOTH RAMP, not a staircase: a discrete "next tread" lookahead over-climbs every swing (user-reported, 2026-07-09, "the knees look way too off")
- **TRIGGER:** Touching `anim_bake._bake_patient_legs`'s gait-driven (no-logged-feet)
  fallback (incident #9), or `_patient_foot_target`'s `_next_tread_height` scan, for
  a REAL (`--frames`) bake specifically.
- **LESSON:** Fixing incident #10 (the twisted foot) unmasked a SEPARATE, larger-
  magnitude problem: even with the foot oriented correctly, the knee itself folded
  57-93+ degrees for MOST of the climb clip (median 58 degrees, worst observed 109),
  never fully straightening even during "stance". Root cause, found by hand-tracing
  `_patient_foot_target`'s inputs/outputs frame-by-frame (NOT by staring at angles --
  see the recurring warning in incidents #4-#9): the real recorder's logged
  `patient.pos[2]` (ground height under the patient) is a PERFECTLY SMOOTH RAMP as
  the patient climbs -- confirmed by dumping consecutive raw frames: a constant
  ~0.00155 m per 33 ms step (~0.062 m/s), with NO discrete per-riser jumps anywhere,
  consistent with a walking person's hip height rising gradually rather than
  teleporting up a full riser at each footfall. But this pipeline's gait fallback fed
  that smooth value into `_stair_terrain_height`'s DISCRETE per-tread step function
  (via `extra_height_at_x`) for `_next_tread_height`'s "find the next tread higher
  than current" scan -- which, given ANY current height, jumps to the next FLAT
  tread's value regardless of how little real forward progress justifies it. At this
  run's real recorded pace (~0.13 m/s forward), ONE FULL GAIT CYCLE (1.1 s) only
  covers ~0.143 m -- under HALF a tread_depth (0.305 m) -- so treating every single
  swing as "climb one whole riser" demanded the foot rise ~2 riser-heights in one
  swing while the (correctly real-data-anchored) hip barely rose, folding the knee
  to compensate. Fixed by replacing the discrete lookahead with
  `_real_data_ramp_terrain_fn`: a SMOOTH ramp (average slope = step_height/step_depth)
  anchored to pass exactly through the CURRENT real (smoothed) ground height, so
  every swing's rise stays proportional to its actual horizontal advance instead of
  snapping to a fixed tread boundary. Verified: full-clip knee_bend median dropped
  58.2 -> 26.5 degrees, max 109 -> 64.4 degrees, stance now correctly recovers to
  near-full leg extension (reach saturating at `PATIENT_MAX_REACH_M`) between swings
  -- re-confirmed visually (natural climbing stride, no more folded-knee "sitting"
  look) and numerically (0% of frames over 70 degrees, was ~20%).
- **WHY:** Incident #6 already established that this pipeline's climb terrain model
  is a clean discrete staircase (by design, for the SYNTHETIC generator, which
  authors its own matched pacing) -- it was reasonable to assume the REAL recorder
  would report the same discrete shape, since it's querying the SAME physical
  staircase. It doesn't: whatever real ground-height query the recorder uses returns
  something that behaves like a smooth interpolation of a walking person's hip
  trajectory, not a raycast against literal tread geometry. Reusing a shared
  discrete-terrain helper (`_stair_terrain_height`/`_next_tread_height`) against a
  DATA SOURCE with a fundamentally different (continuous, not stepped) character is
  the same failure shape as incident #9 (assuming a shared architecture generalizes
  to a new data source without checking whether that source actually behaves the
  way the architecture assumes) -- both were only caught by directly inspecting the
  real recorded VALUES, not by reasoning about the code in the abstract.

### 13 — "Stride too short relative to velocity, sliding/moonwalking" (user-reported, 2026-07-07): REFUTED against the Python gait math -- `stride_len` does not control swing amplitude at all
- **TRIGGER:** Before changing `PATIENT_GAIT_FLAT`/`PATIENT_GAIT_CLIMB`'s `stride_len`
  (synthetic_motion.py) to "fix" a reported short/sliding/moonwalking stride, or before
  assuming a gait-parameter-vs-real-clip-speed mismatch is the cause of that complaint.
- **INITIAL (WRONG) HYPOTHESIS:** `PATIENT_GAIT_FLAT.stride_len=0.36`/`PATIENT_GAIT_CLIMB
  .stride_len=0.16` look tuned for the SYNTHETIC generator's own base speeds
  (`generate_follow_frames`'s `forward_speed=0.4`, `generate_climb_frames`'s
  `climb_speed=0.33`), and a REAL bake's clip can be much slower (measured on this run's
  real `robot_frames.jsonl`: "follow" clip avg 0.263 m/s; "climb" clip avg 0.108 m/s,
  with its flat-approach segment specifically at ~0.115-0.126 m/s) -- so it seemed
  plausible the leg was swinging through its full tuned `stride_len` amplitude while the
  body barely advanced, producing a sliding look. A first implementation scaled
  `stride_len` down by `measured_speed / reference_speed` (capped at 1.0, reference
  speeds 0.4/0.33) inside `anim_bake._bake_patient_legs`'s gait-driven fallback only.
- **WHY THIS WAS WRONG (found by direct standalone simulation, not reasoning from code):**
  `_patient_foot_target` (synthetic_motion.py) computes `touchdown_pos[leg]` at EVERY
  liftoff as `nominal_world(hip, AT THIS LIFTOFF) + stride_len/2`, and `liftoff_pos[leg]`
  is simply the PREVIOUS swing's committed `touchdown_pos[leg]` (via `stance_anchor`) --
  i.e. also `nominal_world(hip, AT THE PREVIOUS LIFTOFF) + stride_len/2`. The
  `+ stride_len/2` term is IDENTICAL at both ends and cancels in
  `dx = touchdown_pos - liftoff_pos`, leaving `dx ~= nominal_world(this liftoff) -
  nominal_world(prev liftoff)`, i.e. simply how far the HIP moved between two
  successive liftoffs of the SAME leg (`~= speed * cycle_period_s`) -- **entirely
  independent of `stride_len`**. Verified by a standalone constant-velocity
  single-leg simulation (speed=0.263 m/s, cycle_period_s=1.2): `stride_len=0.16`,
  `0.36`, and `0.6` ALL produced the identical steady-state per-stride `dx` (0.3163,
  0.3163, 0.3163 m) to 4 decimal places -- `stride_len` only shifts WHERE the
  foot-relative-to-hip sweep is CENTERED (how far ahead of the hip the plant sits),
  not its magnitude. This is a direct, if non-obvious, consequence of incident #6's
  own liftoff/touchdown-blend fix: that fix was designed to make the swing's actual
  displacement self-correct to the body's real travel regardless of any stride_len/
  speed mismatch (see incident #6 item 4's own docstring: "isn't something worth
  hand-tuning to stay true forever") -- which means the mismatch this hypothesis
  worried about CANNOT produce the "too short" symptom in the first place; the fix
  (and the whole premise) was reverted in full (both files restored to their
  pre-investigation state, verified via `git diff` showing no changes).
- **WHAT WAS ACTUALLY MEASURED (all checks passed, no defect found in the Python pipeline):**
  1. Stance-phase foot position: zero world-space drift across 20 consecutive stance
     runs sampled from the real "follow" clip (every run's x range was a single value
     to 5 decimals) -- the plant is solid, not sliding.
  2. hip_pitch/knee_bend angular trajectories: smooth and continuous frame-to-frame
     (angular velocity spot-checked across a full cycle, no stutters/plateaus/pops),
     including across the mid-clip FLAT<->CLIMB gait swap the real "follow" clip
     briefly makes (incident #9's own noted case: patient's lead x crosses
     `start_x-0.05` around t=20.0s in this run) -- foot position was bit-identical
     across that swap frame (both still mid-stance), confirming incident #6's
     phase-matching fix (`cycle_period_s` identical between gaits) holds for real
     data too.
  3. Swing amplitude DOES shrink at slower real speeds, but via the swing's own
     time-window kinematics (`speed * swing_frac * cycle_period_s`), not via
     `stride_len`: a standalone probe holding `stride_len` fixed and varying only
     speed found hip_pitch span 25.2 deg at the real "follow" speed (0.263 m/s) vs
     29.3 deg at the old synthetic tuning speed (0.4 m/s) for the FLAT gait, and
     19.6 deg vs 30.3 deg (real climb-approach ~0.108 m/s vs synthetic 0.33 m/s) for
     CLIMB -- a real, measurable, and CORRECT effect (a slower walker takes
     proportionally shorter steps at a similar ~100 steps/min combined cadence,
     which is physically reasonable for a mobility patient) -- not a bug to fix.
  4. The one genuine oddity found (pre-existing, NOT newly introduced): the IK reach
     clamp (`PATIENT_MAX_REACH_M=0.86`, incident #8) engages on 27-46% of frames in
     both real clips (measured directly from `foot_rel_hip`'s pre-clamp `reach`,
     independent of the baked angles) and floors `knee_bend` at ~24.5 degrees even at
     mid-stance (never straightens closer to 0) -- but this is incident #8's OWN
     documented, deliberate tuning (`PATIENT_STANCE_TARGET_Z=-0.858` was intentionally
     placed close to the reach cap to avoid a worse "sitting" look at a shorter-of-max-
     reach value), confirmed unchanged in the current data, not a new regression.
- **CONCLUSION (per this investigation's own task framing, option (d)):** No fixable
  stride/gait-tuning defect exists in `synthetic_motion.py`/`anim_bake.py` for this
  specific complaint. The most plausible remaining explanation is that the reported
  "sliding/moonwalking" look is a visual byproduct of the SEPARATE ground/toe-clipping
  bug being fixed concurrently in `js/PatientHuman.js` (untouched by this
  investigation, per instruction) -- a foot that clips into/through the ground instead
  of clearing it during swing reads as "gliding along the surface" rather than
  "lifting and stepping," which is exactly the classic visual signature of a
  moonwalk, and is a rendering-layer (Z-clearance-at-the-mesh) issue, not an X-stride-
  length one. If this complaint persists AFTER the clipping fix lands and is
  re-verified in the live viewer, re-open this incident and look at the RENDERED
  (post-retarget) toe trajectory specifically, not the raw `patient_pose.json`
  scalars (which this investigation already exhaustively checked and found sound).
- **WHY THIS BELONGS HERE:** A plausible-sounding, well-reasoned hypothesis ("gait
  constants tuned for a faster reference speed than this slow real clip actually
  moves at") turned out to target a parameter (`stride_len`) that this codebase's own
  prior incident (#6) had already made irrelevant to the symptom in question, via a
  fix whose FULL implications weren't re-derived before reaching for that parameter
  again. Per CLAUDE.md incident 8.7's own warning about trusting comments/assumptions:
  a docstring calling `stride_len` "the fore-aft SWEEP" (synthetic_motion.py's own
  `GaitParams.stride_len` field comment) is technically true only for the SHAPE of the
  swing's fore-aft profile within one call, not for the NET displacement across a full
  liftoff-to-touchdown cycle once the liftoff/touchdown-blend blends from a PRIOR
  commitment -- always verify a "which parameter controls X" belief with a standalone,
  minimal numeric simulation (as done here) before editing the parameter you assume
  controls it.

### 14 — Toe clips 2-4.5cm into ground/treads at every frame (user-reported, 2026-07-09): a bind-pose-only clearance constant, and a units bug found while fixing it
- **TRIGGER:** Touching `js/PatientHuman.js`'s `load()`/`sync()` ground-clearance
  logic (`_ankleGroundClearanceM`, `_legToeDropLocal`, `groundClearanceAboveRootM`),
  or the anchor-height formula in general.
- **LESSON, bug A (root cause):** Incident #5 fixed the ankle-vs-ground anatomical gap
  with a SINGLE constant, `_ankleGroundClearanceM = footWorld.y - toeWorld.y`, measured
  ONCE at Xbot's BIND pose (hip_pitch=knee_bend=0) and added to the anchor height every
  frame regardless of the CURRENT hip_pitch. This under-corrects whenever hip_pitch
  isn't 0 (i.e. essentially always during actual walking): incident #10 already
  established that Foot's world rotation equals `hip_pitch` alone (Leg's own
  `-kneeBend` and Foot's own `+kneeBend` cancel through the FK chain), and ToeBase's
  local rotation is identity, so the whole Foot->ToeBase offset rotates rigidly with
  hip_pitch. That offset has both a "down" and a "forward" component in Foot's local
  frame (measured: `(x=0, y=-8.73, z=10.71)` in raw local units) -- pitching it
  forward rotates more of the forward component into "down", so the true
  ankle-to-toe vertical drop GROWS with hip_pitch (measured: 8.73cm at hip_pitch=0,
  10.89cm at hip_pitch=0.224 rad, 12.67cm at hip_pitch=0.476 rad -- a smooth,
  monotonic, exactly rotation-predicted growth, confirmed to match a hand-derived
  `rotateAboutX` formula to 4+ decimal places). The bind-pose constant is the SMALLEST
  possible value this drop ever takes, so it under-raises the anchor at every other
  hip_pitch -- exactly why the reported clipping fluctuated with gait phase (0-4.5cm)
  instead of being a fixed depth: the STANCE foot (the one actually touching down, at
  the calibrated ~24 degree minimum knee_bend from incident #8) clipped WORST because
  its hip_pitch is generally larger in magnitude during a normal gait's stance portion
  than the old constant assumed.
  Fix: `_legToeDropLocal` in `js/PatientHuman.js` analytically re-derives the FULL
  Hips->UpLeg->Leg->Foot->ToeBase drop (not just the Foot->ToeBase piece) as a
  function of a leg's CURRENT hip_pitch/knee_bend, using each bone's own bind-pose
  local `.position` (never modified elsewhere -- only `.quaternion` is) and the exact
  same rotation composition `sync()` already applies to the live bones. `sync()` now
  samples `hip_pitch_l/r`/`knee_bend_l/r` BEFORE positioning the anchor (reordered
  from the original code, which positioned the anchor first and sampled pose data
  after -- this is the same "read a value before its producer runs" hazard as the
  repo-root CLAUDE.md's `debug_info` incident, just local to this module instead of
  that dict) and picks whichever leg has the SMALLER knee_bend (closer to full
  extension -- the stance leg) as the one whose toe must be exactly at ground/tread;
  the other (swinging) leg's exact height doesn't matter since it's airborne anyway.
  Verified numerically: the stance-foot toe-vs-ground clip is now ~0 (floating-point
  noise, ~1e-7 m) at EVERY sampled frame across both the full 23.6s 'follow' clip (81
  samples) and the full 42.7s 'climb' clip (81 samples), down from a 0.017-0.045m
  penetration range before the fix. The (irrelevant, airborne) swing foot stays within
  a small, sane band (-0.016 to +0.13m, i.e. it's correctly in the air, never buried
  more than ~1.6cm even at its worst instant) with no new frame-to-frame pops (checked
  at 0.03-0.05s resolution across the whole climb clip, worst per-step delta ~1.8cm at
  a normal walking-speed cadence, no discontinuity at the flat->stairs gait-mode
  switch).
- **LESSON, bug B (found WHILE fixing bug A -- a units/reference-point mixup, not a
  geometry mistake):** The first attempt at this fix computed the per-leg drop
  correctly but added it to the anchor formula using the SAME structure as the old
  code (`anchor.z = root.position.z - hipsHeightM + X`) without noticing that `X`'s
  MEANING had changed. The old `_ankleGroundClearanceM` was implicitly "how much
  ABOVE `root.position` (which is itself already ground + `PATIENT_HIP_HEIGHT_M`,
  0.92m, per `anim_bake.py`'s own `patient_root` convention) Xbot's Hips must sit" --
  but the new per-frame drop is naturally derived as "how far ABOVE GROUND Xbot's
  Hips must sit" (it comes from an analytic Hips->ToeBase FK chain with no reference
  to `PATIENT_HIP_HEIGHT_M` at all). Plugging the new (ground-relative) value into a
  slot that expected a root/hip-relative value put the anchor a full
  `PATIENT_HIP_HEIGHT_M` (0.92m) too high -- caught immediately by the SAME
  verification sweep used to confirm the fix (toe landed flush with the HIP height
  instead of the ground, `mixamorigLeftToeBase.y` came back ~equal to `patient_root.y`
  instead of ~0). Fixed by explicitly subtracting `PATIENT_HIP_HEIGHT_M` when
  converting the new ground-relative drop into the old root-relative slot.
- **WHY:** Bug A is the same shape as incident #10 (a fix verified/derived at one
  specific angle regime silently stops applying once the system explores a WIDER
  range -- here, "verified" was never even done at a nonzero angle in the first
  place, since the constant was measured at bind pose and just assumed to generalize).
  Bug B is the same shape as incident #12 (reusing an existing formula/slot without
  re-deriving what its inputs are actually measured RELATIVE TO) -- two numbers can
  both be correct "clearances" in isolation while disagreeing by a full anatomical
  constant on WHAT they're clearances above (ground vs. hip), and mixing them produces
  an error large enough to be obvious once measured (0.92m), but only once someone
  actually re-verifies numerically after the "fix" rather than trusting that fixing
  the geometry fixed the whole bug.
