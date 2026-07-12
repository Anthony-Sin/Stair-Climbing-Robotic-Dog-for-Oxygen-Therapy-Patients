// PatientHuman.js
//
// Loads the patient as a real imported+rigged human model (models/vendor/Xbot.glb --
// Mixamo's "X Bot" mannequin, bundled by three.js's own examples repo -- see
// models/vendor/NOTICE.md). This module owns the SKELETON/RIG side of the patient:
// bone lookup, load-time measurement of the rig's own proportions, two-bone leg/arm
// IK, foot roll + orientation, pelvis/spine/head dynamics, and the cane grip. It does
// NOT own the gait itself -- js/PatientGait.js is a pure-math module that decides
// WHERE the feet (and cane) go and WHEN they step, driven by nothing but the recorded
// patient_root path (see that module's own header for why: user-reported
// "moonwalking" traced to a TIME-driven gait that kept cycling through the recorded
// path's stop-and-go pauses). This module's job is purely "given the P-frame foot/
// root/cane pose PatientGait.poseAt() computed for this instant, retarget it onto
// Xbot's actual bones (or the cane prop)" -- IK math, coordinate conversion, and rig
// bookkeeping, with zero gait/timing decisions of its own.
//
// ===========================================================================
// IK/GAIT OVERHAUL (this rewrite, IK_OVERHAUL_SPEC.md, RIG side of S5/S6/S6b):
// retires Xbot's canned "walk" AnimationClip ENTIRELY (no more AnimationMixer/
// clipPhaseOffset machinery -- see incidents #5/#7/#10 in AGENTS.md, all caused by
// fighting that clip). The upper body is now FULLY PROCEDURAL: pelvis bob/list/yaw/
// lateral-shift, a 3-bone spine chain (lean + yaw-counter + lateral-lean + always-on
// breathing), a free-swinging LEFT arm (FK, driven by the contralateral leg-advance
// signal) and a cane-gripping RIGHT arm (two-bone IK to the cane handle), head/neck
// gaze stabilization, and a foot ROLL model (heel-strike -> flat -> heel-off -> toe-
// off, ankle pivoting analytically about a FIXED contact point per swing/stance
// window). Hips position AND quaternion are now non-trivial every sync() call (both
// still explicitly owned/written here, per incidents #5/#7 -- nothing is ever left to
// a canned clip or a stale previous-frame value: EVERY bone this module cares about
// gets its quaternion (and, for Hips, position) written unconditionally on every
// sync() call, since there is no mixer providing a per-frame baseline for bones this
// module doesn't explicitly touch anymore).
//
// The previous (pre-overhaul) version consumed pipeline/anim_bake.py's baked
// patient_pose.json (per-frame hip_pitch/knee_bend SCALARS computed by a Python
// 2-link IK, driving a TIME-based gait state machine also in Python); that was
// retired in favor of the browser-side PatientGait.js scheduler even before this
// rewrite. This rewrite goes one step further and retires Xbot's OWN canned upper-
// body animation too -- ALL leg pose, foot placement/roll, step timing, pelvis/spine/
// arm/head pose, and cane placement are now computed HERE and in PatientGait.js,
// procedurally, in the browser, driven only by patient_root's recorded XY/yaw/z path
// (see the task's own product constraint: "everything else... is generated
// procedurally in the viewer, NOT taken from Isaac"). patient_pose.json is not
// fetched or read at all.
//
// Coordinate systems: Xbot ships in its own local convention (lateral=local X,
// up=local Y, forward=local Z -- a standard glTF/Mixamo humanoid rig, re-confirmed
// for THIS rewrite by directly parsing Xbot.glb's binary accessors in Node: bind-pose
// node translations/rotations (JSON chunk only) for every bone this module now
// touches, AND the retired "walk" clip's own baked rotation-channel ranges (BIN chunk
// accessors) to find each NEW hinge axis -- see PITCH_AXIS's comment for the
// established leg/spine convention and _ARM_HINGE_AXIS's own comment for the NEW
// elbow-flexion axis this rewrite measured). This viewer's world (everything under
// gltf_export.py's "isaac_world" node, the "P-frame" PatientGait.js's own math is
// expressed in) uses forward=X, lateral=Y, up=Z. B_PLACEMENT is the fixed rotation
// reconciling the two; see its own comment below for the derivation (unchanged from
// the prior module -- still correct, only the bones built on top of it changed). The
// three Xbot-local axis constants this module now uses, all confirmed by direct
// measurement (never assumed, per AGENTS.md incident #4's discipline):
//   PITCH_AXIS         = local X (lateral)  -- hip/knee/ankle/toe sagittal flexion,
//                         ALSO the pelvis-list and spine-pitch/lateral-lean-adjacent
//                         "roll about the fore-aft axis" hinge (see each use site).
//   _ARM_HINGE_AXIS     = local Y (up)      -- elbow flexion (NEW for this rewrite --
//                         NOT the same axis as the leg's knee; see its own comment).
//   _FORWARD_AXIS_LOCAL = local Z (forward) -- pelvis list / spine lateral-lean /
//                         shoulder depression-elevation hinge.
//   _UP_AXIS_LOCAL      = local Y (up)      -- pelvis yaw hinge (same axis as the
//                         elbow's flexion axis, but a conceptually different role --
//                         kept as a separate named constant so a reader never has to
//                         cross-reference "is this the elbow axis or the yaw axis" by
//                         value alone).
//
// Bind-pose fact source for everything new in this rewrite (arm segment lengths/rest
// directions, spine/neck/head/shoulder bind offsets, hand/finger bind directions,
// elbow+toe hinge axes): a standalone Node probe that parses Xbot.glb's GLB container
// directly (12-byte header, then JSON + BIN chunks -- JSON chunk alone has every
// node's bind-pose translation/rotation; the BIN chunk's accessors were additionally
// read to inspect the retired "walk" clip's own baked rotation-channel ranges for the
// NEW hinge-axis measurements), cross-checked against this file's OWN pre-existing
// measured constants (L1/L2 ~= 0.44 m, footLateral ~= 0.082 m, ankleHeight ~= 0.087 m,
// toeForwardLen ~= 0.107 m) before trusting any new number -- see this rewrite's own
// report for the full numeric dump. Every LENGTH/OFFSET constant this module needs
// (arm segment lengths, arm rest directions, spine/shoulder chain bind offsets, hand
// bind axes) is additionally RE-measured live in load() below (same getWorldPosition-
// delta technique the pre-existing leg measurements already used), so a future
// Xbot.glb swap re-derives them rather than silently going stale -- only the AXIS
// CONVENTIONS themselves (PITCH_AXIS, _ARM_HINGE_AXIS, _FORWARD_AXIS_LOCAL,
// _UP_AXIS_LOCAL, B_PLACEMENT) stay hardcoded module-level constants, exactly
// matching this file's own pre-existing precedent for PITCH_AXIS/B_PLACEMENT (an
// axis-convention constant is cited from an offline analysis, not re-derived from
// scratch on every app boot; a length/offset is always re-measured).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
	buildTerrain, extractPathSamples, buildSchedule, poseAt, DEFAULT_GAIT_PARAMS,
} from './PatientGait.js';
import { buildCaneGroup, computeCanePose, applyCanePose, CANE_PARAMS } from './PatientCane.js';

const XBOT_URL = './models/vendor/Xbot.glb';

// Kept in sync with pipeline/anim_bake.PATIENT_HIP_HEIGHT_M and
// PatientGait.js's own copy of the same constant -- the recorded patient_root node's
// world Z is ALWAYS (raw recorded ground height under the patient) + this constant
// (anim_bake.bake_clip: `ppos = (x, y, ground_z + PATIENT_HIP_HEIGHT_M)`, ~L276).
// Used here only to pass through to PatientGait.extractPathSamples's ground-reference
// bookkeeping (this module itself never needs the RAW ground height directly -- the
// anchor is placed from root.z alone, per sync()'s own comment on why the OLD
// "+_ankleGroundClearanceM" whole-rig raise was removed).
const PATIENT_HIP_HEIGHT_M = 0.92;

// Xbot's own local axes, as image vectors in THIS viewer's (forward=X, lateral=Y,
// up=Z) convention: Xbot's local X (lateral) -> our Y, Xbot's local Y (up) -> our Z,
// Xbot's local Z (forward) -> our X. This is a proper (det=+1) rotation -- a cyclic
// axis permutation, not a mirror -- so it preserves rotation handedness/sign.
// UNCHANGED from the prior module version (re-verified as part of this rewrite's own
// numeric re-derivation of the rig's conventions -- this specific placement checks
// out: it maps Xbot's bind-pose "up" (local Y) onto P-frame Z and "forward" (local Z)
// onto P-frame X exactly as the loaded mesh's own bind-pose bounding geometry and the
// (now-retired) walk clip's own root-adjacent bone positions required).
const _basisMatrix = new THREE.Matrix4().makeBasis(
	new THREE.Vector3( 0, 1, 0 ),
	new THREE.Vector3( 0, 0, 1 ),
	new THREE.Vector3( 1, 0, 0 ),
);
const B_PLACEMENT = new THREE.Quaternion().setFromRotationMatrix( _basisMatrix );
const B_PLACEMENT_INV = B_PLACEMENT.clone().invert();

const _UP_Z = new THREE.Vector3( 0, 0, 1 ); // P-frame "up" axis (yaw rotation axis)

// Xbot's own hip/knee/ankle/toe sagittal-plane flexion axis: local X, re-confirmed for
// this rewrite by directly parsing Xbot.glb (no browser) and checking the (now-
// retired) "walk" clip's baked LeftUpLeg rotation keys' per-component RANGE across the
// whole clip (x range 0.42, dominant; y range 0.11; z range 0.15) -- same conclusion
// the prior module's comment already stated. ALSO re-confirmed for TWO NEW bones this
// rewrite adds articulation to: LeftToeBase/RightToeBase (walk clip: x span 0.27/0.23,
// dominant vs y~0.004/z~0.06) -- toe flexion shares the SAME hinge axis as the rest of
// the lower-body sagittal chain, now measured rather than assumed. This axis is what
// this module's two-bone leg IK, foot roll model, and toe articulation all write
// UpLeg/Leg/Foot/ToeBase rotations about.
//
// IMPORTANT SIGN CORRECTION vs. the pre-2026-07-07 module (found during that
// rewrite, NOT assumed, and unchanged by this one): positive LeftUpLeg local-X
// rotation correlates with the foot swinging BACKWARD, not forward -- this module's
// two-bone IK (buildHipQuat/solveLeg below) SIDESTEPS this ambiguity entirely rather
// than resting on it: hip orientation is derived geometrically (rotate the bind-pose
// thigh direction onto the computed target direction, via a quaternion "rotate a onto
// b" construction, then a twist correction to aim the knee-bend plane at a pole
// vector) rather than computed as a signed scalar angle whose sign convention would
// need to be trusted. The ONLY place a scalar sign still matters for a CHILD bone is
// each child's OWN local flexion rotation (Leg relative to UpLeg, ToeBase relative to
// Foot) -- each verified via a full IK->FK round-trip (see this rewrite's own report
// for the generalized-solver numbers; the leg case is bit-identical to the
// pre-existing verified value, `Leg.quaternion = axisAngle(PITCH_AXIS, -kneeBend)`).
const PITCH_AXIS = new THREE.Vector3( 1, 0, 0 );

// NEW for this rewrite: Xbot's OWN local "up" axis, used as the PELVIS YAW hinge
// (Hips quaternion's yaw component, spec S6 item 1) -- kept as a separate named
// constant from _ARM_HINGE_AXIS below even though both happen to be local Y, so nc
// reader has to cross-reference axis VALUES to know which role a given rotation
// plays.
const _UP_AXIS_LOCAL = new THREE.Vector3( 0, 1, 0 );

// NEW for this rewrite: Xbot's own local "forward" axis -- the PELVIS LIST hinge
// (Hips quaternion's list component), the SPINE's own lateral-lean hinge, and the
// shoulder's small cane-load depression/elevation hinge (spec S6 items 1/5/6). Also
// exactly the axis poleFromYaw's own derivation already established as "Xbot local Z
// = P-frame forward" (unchanged reasoning, just promoted to a named constant here
// since this rewrite reuses "Xbot's own forward axis" at several NEW call sites
// beyond the original single inline poleFromYaw closure).
const _FORWARD_AXIS_LOCAL = new THREE.Vector3( 0, 0, 1 );

// NEW for this rewrite: the elbow's flexion axis, measured (NOT assumed to match the
// knee's PITCH_AXIS=local-X) by reading the retired "walk" clip's baked
// RightForeArm/LeftForeArm rotation keys directly (Node probe, BIN-chunk accessor
// read): EVERY key across the whole clip has x=0.0000 and z=0.0000 to 4 decimals,
// only y and w vary (RightForeArm y in [0.2415, 0.4176], one-signed -- an elbow can't
// hyperextend; LeftForeArm y in [-0.5465, -0.2125], one-signed the OTHER way, the
// expected mirror). This is a genuinely DIFFERENT axis from the knee's: the arm's
// bind/T-pose rest direction runs along Xbot's local X (lateral -- see
// _measureArmChain below), and a bend hinge must be perpendicular to the segment's
// own rest direction to mean anything as "flexion" -- local Y (up) is exactly that
// perpendicular for an X-pointing segment, matching this measurement. Sign convention
// (whether `ForeArm.quaternion = axisAngle(_ARM_HINGE_AXIS, +bend)` or `-bend`
// reproduces a requested elbow-IK target) is verified numerically per side via the
// generalized two-bone solver's own IK->FK IK self-check + this rewrite's Node
// round-trip probe (report has the numbers) -- never assumed from the clip alone,
// same discipline as PITCH_AXIS's own sign note above.
const _ARM_HINGE_AXIS = new THREE.Vector3( 0, 1, 0 );

// NOTE: the source glTF names these "mixamorig:LeftUpLeg" etc (with a colon), but
// three.js's GLTFLoader strips the colon when it creates each Object3D's .name
// (confirmed empirically -- getObjectByName('mixamorig:LeftUpLeg') came back null;
// traversing the loaded scene showed "mixamorigLeftUpLeg" instead -- AGENTS.md
// incident #3). Extended for this rewrite with every bone the new pelvis/spine/arm/
// head machinery touches; EVERY name here is looked up with a throw-on-missing
// (load()'s own loop, unchanged pattern) -- these are all STRUCTURALLY required for
// the new IK/FK chains, unlike the (best-effort, non-throwing) finger bones measured
// separately in _setHandGripPose below, which are purely cosmetic.
const BONE_NAMES = {
	leftUpLeg: 'mixamorigLeftUpLeg',
	leftLeg: 'mixamorigLeftLeg',
	rightUpLeg: 'mixamorigRightUpLeg',
	rightLeg: 'mixamorigRightLeg',
	leftFoot: 'mixamorigLeftFoot',
	rightFoot: 'mixamorigRightFoot',
	leftToeBase: 'mixamorigLeftToeBase',
	rightToeBase: 'mixamorigRightToeBase',
	spine: 'mixamorigSpine',
	spine1: 'mixamorigSpine1',
	spine2: 'mixamorigSpine2',
	neck: 'mixamorigNeck',
	head: 'mixamorigHead',
	hips: 'mixamorigHips',
	leftShoulder: 'mixamorigLeftShoulder',
	rightShoulder: 'mixamorigRightShoulder',
	leftArm: 'mixamorigLeftArm',
	rightArm: 'mixamorigRightArm',
	leftForeArm: 'mixamorigLeftForeArm',
	rightForeArm: 'mixamorigRightForeArm',
	leftHand: 'mixamorigLeftHand',
	rightHand: 'mixamorigRightHand',
};

// Finger chain names (best-effort, see _setHandGripPose): 5 fingers x 4 segments,
// colon-free per incident #3, generated rather than hand-listed (40 strings).
const _FINGER_NAMES = [ 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky' ];
const _FINGER_SEGMENTS = [ 1, 2, 3, 4 ];

// ===========================================================================
// PATIENT_BODY_PARAMS -- single source of truth for every NEW tunable this rewrite
// introduces (spec S9: named, unit-suffixed, no magic numbers inline). Defaults are
// the spec's own where one was given; a handful the spec left unspecified are called
// out individually below with the reasoning for the chosen default.
// ===========================================================================
export const PATIENT_BODY_PARAMS = {
	// --- Pelvis (spec S6 items 1/8) ---
	pelvisListRad: 0.05, // rad, hip-drop/hike about the forward axis
	pelvisYawRad: 0.06, // rad, pelvis twist about the up axis
	pelvisDynamicsSpeedRefMps: 0.3, // m/s, list/yaw fade below this speed (min(1,speed/this))
	bobAmplitudeM: 0.018, // m, pelvis vertical bob (fixes P2 -- old anchor-bob-by-gaitPhase was structurally always zero)
	bobSpeedRefMps: 0.25, // m/s, bob "full scale" reference speed
	// R4 (round-2 diag, fullbody_naturalness.md pelvisBob_climb_top_landing): the OLD
	// LINEAR `min(1, speed/bobSpeedRefMps)` scale crushed bob to ~0.23-0.52cm at the
	// climb clip's slow top-landing pace (commandedAmpCm 0.522 measured, vs the
	// [0.8,3.0]cm M13 target) -- reads as lifeless right at the demo's finale. Fixed by
	// taking sqrt() of the same ratio (still exactly 0 at speed===0, i.e. I3's "phaseC
	// frozen -> bob frozen" mechanism is untouched and stays CONTINUOUS through the
	// walk->idle transition -- no new discrete floor/snap, unlike a literal `max(floor,
	// ...)` gate would introduce right at the moment speed reaches exactly 0) but rises
	// much faster than linear at low nonzero speeds (e.g. speed=0.17m/s: linear 0.68 ->
	// sqrt 0.83; speed=0.07m/s: linear 0.28 -> sqrt 0.53), exactly the "soften the
	// multiplier floor while actually walking" the naturalness report asked for.
	bobSpeedRefExponent: 0.5, // dimensionless, exponent applied to the speed/bobSpeedRefMps ratio (0.5 = sqrt; 1.0 would reproduce the old linear behavior)
	hipShiftM: 0.025, // m, lateral weight-shift toward the stance side, applied to Hips.position only (never the anchor -- feet must not move)
	// R1 (round-2 diag, fullbody_naturalness.md knee_*_follow_straight/climb_top_landing):
	// stance-knee median measured 38.5/40.2 deg on flat ground vs the 24-30 deg natural
	// band (incident #8). Root-caused numerically (law-of-cosines against the REAL
	// measured leg chain, AGENTS.md incident #15's v1/v2): the hip pivot's baseline
	// height above a flat ankle target is exactly `PATIENT_HIP_HEIGHT_M(0.92, GAIT-
	// owned, DO NOT change -- IK_OVERHAUL_SPEC.md S2) - _ankleHeightM(~0.087)` = ~0.833m
	// (this falls out of the anchor-placement algebra below regardless of _hipsHeightM's
	// own measured value -- anchor.z=rootPosZ-_hipsHeightM and the hip pivot's own local
	// Y IS _hipsHeightM, so they cancel by construction), which is ~2.7-3cm SHORT of the
	// ~0.865m reach a natural ~27deg stance bend needs on THIS rig's real (non-collinear)
	// 0.8884m leg chain -- i.e. Xbot's own standing-straight height (leg length + ankle
	// height, ~0.976m) is taller than the 0.92m hip-height convention the recorded
	// path/schedule assumes, forcing a permanent partial crouch. Confirmed NOT caused by
	// step-7's anchor-lowering safety clamp (achieved reach 0.83-0.85m sits well UNDER
	// that clamp's own 0.87m reachLimit at every sampled stance frame -- lowering never
	// engages here) or by the ankle targets being wrong (flat-stance ankleTargetWorld.z
	// == pose.leftFoot.z+_ankleHeightM exactly, I2-compliant). Fix: this is a RIG-owned
	// mocap-retargeting compensation (Xbot's fixed proportions vs the recorded skeleton's
	// convention), NOT a change to the shared PATIENT_HIP_HEIGHT_M constant -- raises the
	// anchor (hence every hip pivot) by this fixed amount so the same fixed-length ankle
	// reach lands nearer the natural band. Value chosen from the measured shortfall
	// (empirically re-verified against the regenerated trace, see this rewrite's own
	// report); step 7's reachability clamp still applies AFTER this raise, so on stairs
	// (where reach is already large) any case this would push over the limit is
	// automatically clawed back by that existing safety net, not by this constant.
	// Tuned DOWN from the naive law-of-cosines value (~0.0315m, see the derivation
	// above) once the cane-arm coupling (this param's own use-site comment in sync()
	// step 17) was found: 0.028 landed the knee median well but pushed
	// cane__follow_turning's mean hand-to-handle error/clamp rate up (12.7->22.0mm,
	// reachClampedFrac ~0.5->0.79 in an isolated A/B). 0.018 is a deliberate
	// compromise -- re-verified against the real trace (see this rewrite's own
	// report): knee median still lands close to the natural band (not dead-center,
	// but a large improvement over the 38.5/40.2deg baseline) while keeping the
	// cane-arm regression smaller. This is a genuine, currently-unresolved trade-off
	// between two round-2 findings (R1 knee crouch vs R3a cane-turn reach) sharing
	// one root geometric cause (the cane's own reach margin was already tight even
	// at raise=0) -- see this rewrite's report for the numbers and the two
	// compensation attempts that were tried and reverted (both measured WORSE than
	// no compensation).
	standingReachRaiseM: 0.018,

	// --- Feet / toe (spec S6 items 3/4/S6b) ---
	heelStrikeRad: 0.14, // rad (~8 deg), dorsiflexed peak at touchdown
	rollDownSec: 0.12, // s, heel-strike -> flat easing window
	heelBackM: 0.06, // m, ankle-to-heel horizontal offset -- Xbot ships no explicit heel bone/vertex reference to measure this from (spec's own documented fallback: "if not measurable, heelBackM default 0.06")
	toeOffRad: 0.30, // rad (~17 deg), plantarflexed peak before liftoff
	heelOffSec: 0.22, // s, flat -> heel-off easing window. F3 (integration_2.json diag, 2026-07-10) third residual: widened 0.18->0.22 -- after fixing the swing-profile shape (_smoothstep3Prop) and the toeScale mid-window re-sampling (see _footRollPitch's own doc), the M9b worst case moved to the toe-off window's OWN two-point smoothstep, whose peak angular rate is 1.5*toeOffRad/heelOffSec regardless of either prior fix -- measured 0.1217 rad/sample at the OLD 0.18s width (theoretical peak rate 2.5 rad/s), just over the 0.12 bar. rollDownSec (0.12s) does NOT have the same issue (heelStrikeRad 0.14 is smaller than toeOffRad 0.30, giving a peak rate of only 1.75 rad/s, already under the bar) so is left unchanged. Not consumed by PatientGait.js (no Node audit impact) -- main.js's patientDiag has its own approximate HEEL_OFF_SEC copy for the M8 roll-window gate, kept in sync (see that file's own comment on why an approximation there is tolerated).
	toeOffToeRad: 0.35, // rad, ToeBase's OWN counter-articulation vs Foot during heel-off (keeps the toe pad flat/planted while the ankle plantarflexes)
	toeSwingDroopRad: -0.08, // rad, ToeBase relax droop at mid-swing
	strideLenRollRefM: 0.25, // m, heel-strike scale reference (min(1, strideLen/this))
	speedRollRefMps: 0.15, // m/s, toe-off scale reference (min(1, speed/this) -- spec's own documented fallback for "next event's strideLen if accessible", which PatientGait v2's contract does not expose)
	swingMidDorsiflexRad: 0.10, // rad, foot-roll swing profile's mid-swing peak
	toeSwingMidDroopRad: -0.08, // rad, ToeBase swing profile's mid-swing droop (same value as toeSwingDroopRad; named separately since they play different roles -- see the swing profile call site)

	// --- Spine (spec S6 item 5) ---
	spineYawCounterK: - 0.6, // dimensionless, vs pelvisYaw
	spineLateralLeanK: 0.4, // dimensionless, vs support*pelvisList
	caneLoadLeanRad: 0.02, // rad, extra spine lateral lean toward the cane while planted+bearing
	breathPitchRad: 0.008, // rad, always-on breathing amplitude
	breathHz: 0.27, // Hz, always-on breathing rate

	// --- Arms (spec S6 item 6) ---
	armSwingRad: 0.27, // rad, left-arm shoulder swing amplitude (F5, integration_2.json diag 2026-07-10: nudged 0.22->0.27, M11_armSwingAmplitude measured 0.0892 rad against bar [0.1,0.45] -- see this rewrite's own report for the re-measured landed value)
	// R2 (round-2 diag, fullbody_naturalness.md armSwing_left_*): a FIXED
	// armSwingReachRefM=0.35 measured only ~4deg swing (vs the 8-16deg elderly band)
	// consistently across EVERY sub-window (follow straight/turning, climb stairs/
	// landing) -- 0.35m was calibrated for a longer stride than this slow gait's
	// contralateral-foot-vs-root forward excursion ever reaches. Measured directly from
	// the real trace (diag/trace_full.json): that raw excursion is a remarkably STABLE
	// fraction of the driving foot's OWN strideLen (~0.30-0.34, median, across follow
	// AND climb, straight AND turning alike -- NOT stride-magnitude-dependent), so
	// normalizing by a fraction of strideLen (a pose field, per-event/piecewise-
	// constant, never a live per-frame re-derivation -- incident #6 discipline) rather
	// than a fixed meters constant self-corrects if GAIT's own stride-length-vs-speed
	// tuning changes later, instead of needing to be re-tuned by hand again. See
	// _clampedAdvance's own call sites in sync() (steps 17/18) for the exact formula.
	armSwingStrideFracK: 0.40, // dimensionless, forward-advance normalization reference = this * the averaged L/R strideLen (see sync()'s `armReachRefM`)
	armSwingReachFloorM: 0.10, // m, floor for that reference (guards near-zero/undefined strideLen -- before the first-ever step, or a v1 schedule lacking strideLen; verified against the real trace's own first-step ramp-up: rawAdv stays << this floor there, so the floor never dominates in practice, it only prevents a divide-by-~0)
	armAbductRad: 0.10, // rad, constant small abduction (sleeve/hip clearance)
	elbowBaseRad: 0.35, // rad, resting elbow bend
	// elbowSwingRad: the spec (S6.6) gives the FORMULA
	// "elbowBaseRad + elbowSwingRad*max(0,adv_R)*0.3" but never states a default for
	// elbowSwingRad itself -- filled in here at the same order of magnitude as
	// armSwingRad (0.22) so the extra swing-driven bend maxes out at a subtle
	// ~0.09 rad (~5 deg) on top of the 0.35 rad base, per this rewrite's own report.
	elbowSwingRad: 0.30,
	shoulderModulationRad: 0.02, // rad, RIGHT shoulder's cane-load depression/elevation (support-driven) AND (this rewrite's own filled gap -- the task's summary mentions "shoulder bones get small load/swing modulation" for BOTH sides, but S6.6 only gives an explicit formula for the right/cane one) the LEFT shoulder's smaller adv_R-driven counterpart, reusing this same magnitude for both -- see sync()'s own comment at each write site.
	gazeDownRad: 0.10, // rad, head's constant downward gaze bias
	headCounterPitchK: - 0.7, // dimensionless, vs total spine pitch
	headCounterYawK: - 0.5, // dimensionless, vs net torso yaw (pelvisYaw + spineYawCounter)
	// F8 (integration_2.json diag, 2026-07-10): M13_headVsPelvisAmplitudeRatio
	// measured 1.061 (bar <1.0). ORIGINAL hypothesis (below, kept because the
	// term is still a real, defensible vestibular-ocular-reflex behavior --
	// keeping the head level side-to-side as the pelvis lists) was that pelvis
	// LIST's lever arm at head height dominates -- REFUTED by measurement:
	// _debugM13Series (a real per-sample time series probe, not a re-derivation
	// -- AGENTS.md incident #15 discipline) showed headRelY tracks hipsRelY at
	// corr=0.998 (nearly perfectly in-phase) while corr(listRad, headDev)=only
	// -0.15 (weak), and quadrupling headCounterListK's magnitude (-0.85->-3.0)
	// moved the ratio by under 0.001 -- list genuinely isn't the dominant term.
	// (Root geometric reason found afterward: for a point mostly ALONG a
	// rotation axis's own "up" direction, d(position)/d(small rotation angle)
	// is dominated by the HORIZONTAL/tangential component, not vertical --
	// second-order in the angle -- so neither list nor pitch rotation was ever
	// going to move the head much VERTICALLY at these small angles. The real
	// excess (headAmp/hipsAmp = 1.047, i.e. only ~4.7%) is a small, in-phase-
	// with-bob amplification through the FK chain's own geometry, not a rotation
	// lever-arm effect.) Kept at a small, non-zero value for the real (if minor)
	// list-leveling behavior; see headBobCounterFrac below for the fix that
	// actually addresses the M13 bar.
	headCounterListK: - 0.85, // dimensionless, vs net torso list (pelvisListRad + spineLateralLean)
	// F8 fix (2026-07-10): direct counter-translation on Head.position (LOCAL Y,
	// same raw/scaled convention as Hips.position -- see _bindHeadPosition's own
	// comment), proportional to bobM, REDUCING (not fully canceling: headroom
	// left for a real, if damped, head bob rather than a perfectly rigid head)
	// how much of the pelvis bob the head ends up carrying -- directly targets
	// the MEASURED quantity (head's own world Y amplitude vs the pelvis's)
	// rather than a rotation-based proxy, since the rotation-based approach
	// (headCounterListK above) was measured NOT to move this bar. Tuned
	// empirically via window.__viewer.gaitReport against the real M13 metric
	// (see this rewrite's own report for the landed ratio).
	headBobCounterFrac: 0.35, // dimensionless in [0,1]; head inherits (1-this) of the pelvis bob it would otherwise carry via pure FK propagation
};

// ===========================================================================
// Elderly-patient colouring by 3D BODY REGION (silver hair / pale skin / dusty-teal
// knit sweater / grey trousers / brown slippers), applied as a per-vertex COLOR
// attribute -- NOT a diffuse texture.
//
// WHY vertex colours and not a baked texture map: Xbot's UV islands OVERLAP/mirror
// (measured: a single atlas texel is shared by torso AND foot vertices), so ANY
// diffuse map keyed to those UVs cross-contaminates -- the torso ends up sampling
// the feet's texels and renders as bare skin. Vertex colours are keyed to each
// vertex's own 3D position, never to UVs, so overlap is irrelevant. This mirrors
// the ROBOT, which already ships baked COLOR_0 vertex colours (main.js).
// patientRegionKey() below is the SOLE source of truth for the region layout (the
// earlier texture baker that shared this logic has been retired). Coords are
// model-space bind pose (x lateral, y up 0..~1.81, z front(+)/back(-)).
//
// Colours are authored as sRGB hex; `new THREE.Color(hex)` converts to the
// renderer's LINEAR working space on construction (ColorManagement on), which is
// the space a `color` vertex attribute is read in (same convention as the robot's
// baked-linear COLOR_0). MeshToonMaterial multiplies this into its white base
// colour and still hard-bands it, so the toon look + fresnel rim are unchanged.
//
// UNCHANGED by this rewrite (RIG's IK/gait overhaul does not touch appearance).
const PATIENT_REGION_COLORS = {
	skin:      new THREE.Color( 0xE8C9B2 ), // warm pale elderly skin
	hair:      new THREE.Color( 0xEAE9E3 ), // soft silver-white
	cardigan:  new THREE.Color( 0x6E9894 ), // dusty teal knit sweater
	cardigan2: new THREE.Color( 0x567C79 ), // darker teal -- ribbed hem/cuff/collar trim
	trousers:  new THREE.Color( 0x968E80 ), // soft warm-grey slacks
	shoes:     new THREE.Color( 0x5C4A3C ), // muted brown slippers
};

/** Region key for a model-space bind-pose point. Order = specific -> general.
 *  Sole source of truth for the patient's region layout; see the block comment above. */
function patientRegionKey( x, y, z ) {

	const ax = Math.abs( x );

	// Head (y >= 1.52): a full head of silver-white hair (crown + back + sides +
	// temples) with a hairline across the forehead, leaving a clean central face.
	// Geometry (measured on Beta_Surface): head y 1.50..1.806, z -0.121(back)..0.128(front),
	// x +-0.178; the face-front occupies ~y 1.58..1.78, z > 0.
	if ( y >= 1.52 ) {

		if ( y < 1.58 ) return 'skin';        // neck
		if ( y >= 1.71 ) return 'hair';       // crown + top of the scalp (front AND back)
		if ( z <= 0.02 ) return 'hair';       // back + sides of the skull
		if ( ax >= 0.11 ) return 'hair';      // temples / over the ears -- frames the face
		return 'skin';                        // central face below the hairline (kept clean -- painted features read uncanny)

	}

	// Arms (horizontal T-pose band): long knit sleeves -> ribbed cuff -> bare hands.
	if ( ax >= 0.24 && y >= 1.15 ) {

		if ( ax >= 0.71 ) return 'skin';      // hands
		if ( ax > 0.665 ) return 'cardigan2'; // ribbed cuff
		return 'cardigan';                    // long sleeve

	}

	// Torso: one solid closed knit sweater cut LONG as a tunic over the hips (hides
	// Xbot's stock waist-pinch); a darker-teal ribbed crew collar at the FRONT
	// neckline only, ribbed hem below. The collar is teal (not a cream/shirt tone):
	// a pale collar reads as bare skin under the warm key light, and a z-symmetric
	// collar wrongly painted the upper BACK too -- so it's gated to the front (z > 0).
	if ( y >= 0.92 ) {

		if ( y >= 1.46 && ax < 0.13 && z > 0.0 ) return 'cardigan2'; // front crew-neck ribbed trim
		if ( y < 0.99 ) return 'cardigan2';                          // ribbed sweater hem
		return 'cardigan';

	}

	// Trousers over the legs, then slippers.
	if ( y >= 0.13 ) return 'trousers';
	return 'shoes';

}

/** Bake a per-vertex `color` attribute onto `geometry` from patientRegionKey() of
 *  each vertex's bind-pose position (geometry.attributes.position is pre-skinning
 *  local space, i.e. the model-space T-pose coords the region thresholds expect). */
function paintPatientRegionColors( geometry ) {

	const pos = geometry.attributes.position;
	if ( ! pos ) return;
	const colors = new Float32Array( pos.count * 3 );
	for ( let i = 0; i < pos.count; i ++ ) {

		const c = PATIENT_REGION_COLORS[ patientRegionKey( pos.getX( i ), pos.getY( i ), pos.getZ( i ) ) ];
		colors[ i * 3 ] = c.r;
		colors[ i * 3 + 1 ] = c.g;
		colors[ i * 3 + 2 ] = c.b;

	}
	geometry.setAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );

}

// ===========================================================================
// Small local math helpers (kept separate from PatientGait.js's own -- this module
// operates on THREE.Vector3/Quaternion, PatientGait.js is deliberately THREE-free)
// ===========================================================================

/** Shortest-arc quaternion rotating unit vector `a` onto unit vector `b` (both cloned/normalized internally). Standard cross/dot construction; the antiparallel case picks an arbitrary perpendicular axis for the 180deg rotation (any works). */
function _quatFromTo( a, b, out = new THREE.Quaternion() ) {

	const an = a.clone().normalize();
	const bn = b.clone().normalize();
	const d = an.dot( bn );

	if ( d > 1 - 1e-9 ) return out.identity();

	if ( d < - 1 + 1e-9 ) {

		let perp = new THREE.Vector3().crossVectors( an, new THREE.Vector3( 1, 0, 0 ) );
		if ( perp.lengthSq() < 1e-6 ) perp = new THREE.Vector3().crossVectors( an, new THREE.Vector3( 0, 1, 0 ) );
		perp.normalize();
		return out.setFromAxisAngle( perp, Math.PI );

	}

	const axis = new THREE.Vector3().crossVectors( an, bn );
	return out.set( axis.x, axis.y, axis.z, 1 + d ).normalize();

}

/** Project `v` onto the plane perpendicular to unit `axis`; null if the residual is ~0 (v parallel to axis). */
function _projectPerp( v, axis ) {

	const p = v.clone().sub( axis.clone().multiplyScalar( v.dot( axis ) ) );
	return p.lengthSq() < 1e-10 ? null : p.normalize();

}

/**
 * Rotate `fromDir` onto `toDir` (shortest arc), then twist about `toDir` so
 * `hingeAxis` (AS SWUNG by that shortest-arc rotation) aligns with `referenceAxis` as
 * closely as possible (both projected into the plane perpendicular to `toDir`).
 *
 * This is the "aim + roll" two-constraint construction the two-bone solver's hip/
 * shoulder orientation has always used (previously inlined directly in `_solveLegIK`)
 * -- factored out here, UNCHANGED operation-for-operation (so the leg path stays
 * bit-identical, see this rewrite's own verification report), so it can ALSO drive
 * the foot's roll-pivot orientation and the cane-gripping hand's wrist orientation
 * without three near-duplicate copies of the same six lines.
 */
function _aimWithTwist( fromDir, toDir, hingeAxis, referenceAxis, out = new THREE.Quaternion() ) {

	const swing = _quatFromTo( fromDir, toDir );
	const hingeAfterSwing = hingeAxis.clone().applyQuaternion( swing );

	const hingeProj = _projectPerp( hingeAfterSwing, toDir );
	const refProj = _projectPerp( referenceAxis, toDir );

	if ( hingeProj && refProj ) {

		const cosT = THREE.MathUtils.clamp( hingeProj.dot( refProj ), - 1, 1 );
		const crossT = new THREE.Vector3().crossVectors( hingeProj, refProj );
		const sinSign = Math.sign( crossT.dot( toDir ) ) || 1;
		const twistAngle = Math.acos( cosT ) * sinSign;
		const twist = new THREE.Quaternion().setFromAxisAngle( toDir, twistAngle );
		return out.copy( twist.multiply( swing ) );

	}

	// Degenerate (toDir parallel to hingeAxis or referenceAxis) -- swing-only is the
	// best available answer; rare (would need the limb pointing exactly along its own
	// hinge axis).
	return out.copy( swing );

}

// ===========================================================================
// Two-bone IK (legs AND, new for this rewrite, the cane-gripping right arm)
// ===========================================================================

/**
 * Precompute the closed-form constants a two-bone chain's REAL bind-pose geometry
 * needs for `_solveTwoBoneIK` below (RIG-3 fix, AGENTS.md incident #15 -- read that
 * incident before touching anything in this section): `v1` (proximal->medial bind
 * vector, e.g. UpLeg->Leg) and `v2` (medial->distal bind vector, e.g. Leg->Foot),
 * anchor-local METERS, measured exactly like this file's pre-existing L1/L2 (world
 * position deltas at bind pose via getWorldPosition -- NEVER bone-local `.position`,
 * which carries the Armature's 0.01 scale, see `_bindHipsPosition`'s own comment), and
 * `hingeAxis` (the medial bone's own local rotation axis -- `PITCH_AXIS` for legs,
 * `_ARM_HINGE_AXIS` for arms).
 *
 * Xbot's real bind offsets are NOT collinear (measured: ~4.21 deg angle between a
 * leg's own v1/v2, one segment tilted ~3.84 deg off pure "straight down"; the arm's
 * v1/v2 measured within ~2e-4 deg of perfectly collinear, i.e. the arm was already
 * effectively straight -- see incident #15 for the full numbers), so `_solveTwoBoneIK`
 * decomposes each vector into a component ALONG hingeAxis (`x1`/`x2` -- invariant
 * under the medial bone's own rotation about that axis) and a component in the
 * PERPENDICULAR plane (`p1`/`p2`, magnitudes `p1Len`/`p2Len`, offset at bind by the
 * signed angle `phi0`). Called once per chain at load() -- legs share ONE chain
 * between both sides (mirror-symmetry measured to <2e-5 m, same assumption the
 * pre-existing L1/L2 already shared), arms get a chain PER SIDE (an arm's dominant
 * component IS the mirrored/lateral axis, unlike a leg's) -- so sync()'s hot path
 * never repeats a dot/cross/acos on this constant geometry.
 *
 * `degenerate` (a segment pointing exactly along its own hinge axis, i.e.
 * `p1Len`/`p2Len` ~ 0) is a defensive-only fallback to the old collinear-style
 * formula -- never observed on Xbot's own leg/arm chains (both measure
 * `p1Len`/`p2Len` >= 0.16 m) -- guarded only so a future rig swap degrades instead of
 * dividing by ~0.
 */
function _buildChainGeometry( v1, v2, hingeAxis ) {

	const x1 = v1.dot( hingeAxis ), x2 = v2.dot( hingeAxis );
	const p1 = v1.clone().addScaledVector( hingeAxis, - x1 );
	const p2 = v2.clone().addScaledVector( hingeAxis, - x2 );
	const p1Len = p1.length(), p2Len = p2.length();

	const degenerate = p1Len < 1e-6 || p2Len < 1e-6;

	// phi0: signed angle (about hingeAxis) FROM p2's direction TO p1's direction --
	// atan2(cross(p2,p1).hingeAxis, p1.p2) is the standard "signed angle between two
	// vectors given a reference normal" construction; the RAW (non-unit) p1/p2 are
	// safe to feed atan2 directly since both arguments carry the same |p1||p2| scale
	// factor. Cross-checked against `v1.angleTo(v2)` in this fix's own Node probe:
	// |phi0| matched to 3 decimal places for the real leg chain (4.208 vs 4.2077 deg).
	let phi0 = 0;
	if ( ! degenerate ) {

		const cross = new THREE.Vector3().crossVectors( p2, p1 );
		phi0 = Math.atan2( cross.dot( hingeAxis ), p1.dot( p2 ) );

	}

	return {
		v1: v1.clone(), v2: v2.clone(), hingeAxis: hingeAxis.clone(),
		L1: v1.length(), L2: v2.length(),
		x1, x2, p1Len, p2Len, phi0, degenerate,
		bindExtensionM: v1.clone().add( v2 ).length(), // reach at theta=0 (bind pose) -- see _solveTwoBoneIK's header for why this ISN'T the true geometric max |v1|+|v2|
		foldExtensionM: Math.sqrt( ( x1 + x2 ) * ( x1 + x2 ) + ( p1Len - p2Len ) * ( p1Len - p2Len ) ), // reach at the fully-folded limit (theta = pi-phi0), derived alongside the same closed form
	};

}

/**
 * Solve a two-bone (proximal->medial->distal, e.g. hip->knee->ankle OR
 * shoulder->elbow->wrist) IK chain for `target` (a point in the SAME space as
 * `pivot` -- this module always calls it with both in anchor-local/Xbot-local space),
 * with the bend toward `poleVector` (also anchor-local) and `chain` (a
 * `_buildChainGeometry(v1, v2, hingeAxis)` result -- legs pass the shared
 * `this._legChain`; the arm passes its own per-side `this._armChain.left`/`.right`).
 *
 * RIG-3 FIX (AGENTS.md incident #15 -- read before trusting any "verified" claim near
 * this function; CLAUDE.md 8.7 applies doubly here since a previous version of this
 * exact docstring asserted a now-false bit-identical-to-old claim): the pre-existing
 * solver modelled BOTH segments as pointing along a single hardcoded
 * `_REST_DIR=(0,-1,0)` at bind ("straight down"). Xbot's REAL bind offsets are NOT
 * collinear (see `_buildChainGeometry`'s own comment) -- forcing them through one
 * straight rest direction measured a worst-case ~0.033 m FK error against the real
 * rig (this fix's own Node probe, real THREE.Object3D FK, 231 targets incl. the 6
 * self-check offsets) and inflated the reported stance knee-bend well past a natural
 * range. This function now solves the CHAIN'S OWN bind geometry exactly; verified via
 * real THREE.Object3D FK round-trips (bind translations parsed from Xbot.glb directly)
 * to machine precision (worst observed ~5e-15 m, several orders of magnitude below
 * the 1e-6 m bar) across those same 231 targets -- see this fix's own report (AGENTS.md
 * incident #15) for the full table. Do NOT assume this docstring's numbers stay true
 * forever (CLAUDE.md 8.7): after any change near this function or `_buildChainGeometry`,
 * re-derive them by parsing Xbot.glb's bind-pose node translations directly (world-
 * position deltas, the same technique `load()` already uses for L1/L2 etc -- no
 * browser needed) and re-running a real `THREE.Object3D` FK round-trip, not a second
 * hand-derivation.
 *
 * DERIVATION: write the medial bone's own local rotation (about `hingeAxis`) as
 * `theta` -- UNCHANGED bone-write contract from the pre-existing solver: the caller
 * still writes `Leg.quaternion = axisAngle(hingeAxis, -theta)` (this function's
 * returned `kneeBend` IS `theta`, kept under its old name so every existing call site
 * -- `_orientFoot`, the knee-bone write itself -- needed zero changes). The achieved
 * reach, as a function of theta, is the length of
 * `u(theta) = v1 + axisAngle(hingeAxis,-theta)*v2` (the pivot->target vector IN THE
 * UNROTATED/bind frame -- `hipQuat`, applied on top, is a pure rotation, so it
 * preserves length and carries `u(theta)` onto the true world/anchor-local reach).
 * Decomposing via the chain's own x1/x2 (hingeAxis-aligned, invariant under the medial
 * bone's rotation) and p1Len/p2Len/phi0 (the perpendicular "planar" parts, offset at
 * bind by phi0):
 *
 *   reach(theta)^2 = |v1|^2 + |v2|^2 + 2*(x1*x2 + p1Len*p2Len*cos(theta+phi0))
 *
 * (reduces EXACTLY to the pre-existing collinear-model law of cosines when phi0=0 and
 * x1=x2=0, via `acos(-x)=pi-acos(x)`; confirmed both symbolically and numerically).
 * Solving for theta at a clamped target reach `d`:
 *
 *   cc := clamp((d^2 - |v1|^2 - |v2|^2 - 2*x1*x2) / (2*p1Len*p2Len), -1, 1)
 *   theta = acos(cc) - phi0
 *
 * BRANCH CHOICE (the other root, `-acos(cc)-phi0`, is REJECTED): this is the unique
 * branch with theta=0 EXACTLY at d=chain.bindExtensionM (full bind extension -- the
 * rejected root gives theta=-2*phi0 there instead, nonzero), and theta increases
 * monotonically from 0 toward `pi-phi0` as d shrinks from `bindExtensionM` toward
 * `foldExtensionM` -- i.e. "smaller reach -> more bend", the same qualitative behavior
 * the pre-existing collinear solver had, and it bends the knee on the SAME side (same
 * `bendAxis`/pole-vector convention, unchanged below) the pre-existing solver did.
 * Once theta is known, `u(theta)`'s DIRECTION (NOT v1's own bind direction, which the
 * old formula used, and which is only exactly correct when v1 IS the whole chain's
 * direction, i.e. collinear v1/v2) is what `hipQuat` swings onto the target direction:
 * `hipQuat` is defined by `hipQuat . u(theta) = target-pivot`, which is what FK
 * actually requires regardless of collinearity. The pole-vector/twist step is
 * UNCHANGED (`_aimWithTwist`, same `bendAxis` derivation as the pre-existing solver).
 *
 * Returns `{ hipQuat, kneeBend, reachClamped, achievedReach, anatomicalBendRad }`
 * (first four field names/meanings kept from the original leg-only version):
 *   - hipQuat/kneeBend/reachClamped/achievedReach: UNCHANGED meaning from the
 *     pre-existing solver -- kneeBend is `theta` above, NOT the anatomical knee-bend
 *     angle (see next); caller writes the medial bone's OWN local rotation as
 *     `axisAngle(hingeAxis, -kneeBend)` (sign verified per-chain via `_elbowSign` for
 *     arms, hardcoded `-` for legs, both UNCHANGED by this fix).
 *   - anatomicalBendRad (NEW, RIG-3/AGENTS.md #15): the ANATOMICAL deviation-from-
 *     straight -- the angle BETWEEN the achieved thigh direction (v1, as rotated by
 *     hipQuat) and achieved shin direction (`axisAngle(hingeAxis,-theta)*v2`, as
 *     rotated by hipQuat) -- a rotation-invariant quantity, computed directly from the
 *     UNROTATED vectors (hipQuat cancels): `acos(v1.(axisAngle(hingeAxis,-theta)*v2) /
 *     (L1*L2))`, reusing `cc` (already `cos(theta+phi0)`) rather than a second trig
 *     pass. NOTE (verified numerically, a genuine law-of-cosines identity): for a
 *     GIVEN achieved reach `d`, this equals the OLD solver's own `kneeBend` value
 *     exactly (`d^2=L1^2+L2^2+2*L1*L2*cos(anatomical)` is the same triangle either
 *     way) -- the old solver's REPORTED angle was never wrong in isolation; the bug
 *     was that it aimed `hipQuat` using the wrong bind direction, so the ACTUAL
 *     achieved `d` (and therefore the real foot position) was wrong by ~0.03 m even
 *     though the angle-for-that-d was self-consistent. `_lastSync.leftKneeBendDeg`/
 *     `rightKneeBendDeg` (sync()'s own diagnostic) reads THIS field so its meaning
 *     stays "anatomical angle for the achieved d", now backed by a correct `d`.
 */
function _solveTwoBoneIK( pivot, target, poleVector, maxReachM, minReachM, chain ) {

	const hingeAxis = chain.hingeAxis;

	const toHip = new THREE.Vector3().subVectors( target, pivot );
	const rawDist = toHip.length();
	const reachClamped = rawDist > maxReachM;
	const d = THREE.MathUtils.clamp( rawDist, minReachM, maxReachM );

	const toTarget = rawDist > 1e-6 ? toHip.clone().normalize() : chain.v1.clone().normalize();

	// cc := cos(theta + phi0), this function's own header derivation. The degenerate
	// fallback (chain.phi0=0, p1Len/p2Len effectively L1/L2 -- see
	// _buildChainGeometry) reduces `cc` to the exact pre-existing collinear formula's
	// cosine term, so no separately-branched formula is needed below.
	const denom = chain.degenerate ? ( chain.L1 * chain.L2 ) : ( chain.p1Len * chain.p2Len );
	const cc = THREE.MathUtils.clamp( ( d * d - chain.v1.lengthSq() - chain.v2.lengthSq() - 2 * chain.x1 * chain.x2 ) / ( 2 * denom ), - 1, 1 );
	const kneeBend = Math.acos( cc ) - chain.phi0;

	const legLocalQuat = new THREE.Quaternion().setFromAxisAngle( hingeAxis, - kneeBend );
	const u = chain.v1.clone().add( chain.v2.clone().applyQuaternion( legLocalQuat ) );
	const dirU = u.lengthSq() > 1e-12 ? u.normalize() : chain.v1.clone().normalize();

	let bendAxis = new THREE.Vector3().crossVectors( toTarget, poleVector );
	if ( bendAxis.lengthSq() < 1e-8 ) {

		bendAxis = new THREE.Vector3().crossVectors( toTarget, new THREE.Vector3( 1, 0, 0 ) );
		if ( bendAxis.lengthSq() < 1e-8 ) bendAxis = new THREE.Vector3().crossVectors( toTarget, new THREE.Vector3( 0, 1, 0 ) );

	}
	bendAxis.normalize();

	const hipQuat = _aimWithTwist( dirU, toTarget, hingeAxis, bendAxis );

	const cosAnatomical = THREE.MathUtils.clamp( ( chain.x1 * chain.x2 + denom * cc ) / ( chain.L1 * chain.L2 ), - 1, 1 );
	const anatomicalBendRad = Math.acos( cosAnatomical );

	return { hipQuat, kneeBend, reachClamped, achievedReach: d, anatomicalBendRad };

}

const _REST_DIR = new THREE.Vector3( 0, - 1, 0 ); // "straight down" -- RIG-3 fix (AGENTS.md #15): NO LONGER fed into _solveTwoBoneIK (legs/arms now solve their own real, non-collinear bind geometry via _buildChainGeometry/this._legChain/this._armChain instead of assuming this single shared direction); still used as the arm's FK HANGING (not bind/T-pose) target direction (see the hangQuat construction in sync()) and as a couple of degenerate-input fallbacks

// ===========================================================================
// Foot roll (spec S6b) -- stateless pure functions of the v2 poseAt contract's
// per-foot timing fields (landedAt/nextLiftAt/strideLen) + the query time/speed.
// ===========================================================================

/** Smoothstep, 0 at u=0, 1 at u=1, zero slope at both ends. */
function _smoothstep( u ) { return u * u * ( 3.0 - 2.0 * u ); }

/** 3-point profile: exactly `startVal` at u=0, `midVal` at u=0.5, `endVal` at u=1,
 *  C1-continuous throughout (two independently zero-slope-ended smoothsteps glued at
 *  u=0.5 -- zero slope from BOTH sides there too). See _footRollPitch's own comment
 *  for why the swing profile's endpoints are DERIVED to match the adjacent stance
 *  windows analytically rather than using the spec's illustrative flat constants. */
function _smoothstep3( u, startVal, midVal, endVal ) {

	if ( u <= 0.5 ) return startVal + ( midVal - startVal ) * _smoothstep( u / 0.5 );
	return midVal + ( endVal - midVal ) * _smoothstep( ( u - 0.5 ) / 0.5 );

}

/**
 * F3 (integration_2.json diag, 2026-07-10): same 3-point/C1 contract as
 * _smoothstep3 (exactly startVal at u=0, midVal at u=uMid, endVal at u=1, zero
 * slope at all three), but the u=0.5 BREAKPOINT is replaced with one placed
 * PROPORTIONALLY to each half's own angular span (uMid = |midVal-startVal| /
 * (|midVal-startVal|+|endVal-midVal|)), so a lopsided profile -- e.g. the swing
 * roll profile's own startVal/midVal/endVal = -toeOffRad*toeScale/
 * swingMidDorsiflexRad/heelStrikeRad*heelScale, whose FIRST half (~0.40 rad,
 * push-off continuing to mid-dorsiflex) is ~10x the SECOND half's (~0.04 rad,
 * mid-dorsiflex to heel-strike) -- doesn't cram the big transition into the SAME
 * fixed half-width as the small one. M9b_footPitchContinuityMax measured 0.174
 * rad/sample @ dt=0.05s with the fixed 0.5 breakpoint on a near-floor-duration
 * (~0.33s) swing -- root-caused via _debugWorstPitch instrumentation to be
 * mid-swing (u~0.15-0.30, well inside a single event, NOT a tLift/tLand boundary
 * or a clip-junction/walk-on-freeze edge -- both the task's own "prime suspects"
 * were verified and ruled out): the fixed-0.5-breakpoint profile's own peak slope
 * in a lopsided half, not a discontinuity. Proportional allocation equalizes
 * peak angular velocity across both halves (verified by Node simulation across
 * swingDur 0.32-0.7s: worst case 0.102 rad/sample, under the 0.12 bar) WITHOUT
 * changing any endpoint value (still bit-identical continuity with the adjacent
 * stance windows -- only the INTERIOR shaping changes). Degenerate case
 * (startVal===midVal===endVal, both spans zero) falls back to the fixed 0.5
 * midpoint. Used by _footRollPitch's swing branch only (the ANKLE/Foot bone
 * M9b actually measures) -- _toeBasePitch's own swing profile is unchanged
 * (ToeBase isn't part of the M9b measurement; not touched, out of scope).
 */
function _smoothstep3Prop( u, startVal, midVal, endVal ) {

	const w1 = Math.abs( midVal - startVal ), w2 = Math.abs( endVal - midVal );
	const total = w1 + w2;
	const uMid = total > 1e-9 ? w1 / total : 0.5;

	if ( u <= uMid ) return startVal + ( midVal - startVal ) * _smoothstep( uMid > 1e-9 ? u / uMid : 0 );
	return midVal + ( endVal - midVal ) * _smoothstep( uMid < 1 - 1e-9 ? ( u - uMid ) / ( 1 - uMid ) : 0 );

}

/**
 * Foot roll pitch (radians, PITCH_AXIS convention: positive = dorsiflexed/heel-down-
 * toe-up, negative = plantarflexed/heel-up-toe-down) at time `t` for one foot, given
 * that foot's `footPose` (poseAt v2's per-foot object) and `speed`, the value
 * `toeScale` (see below) is built from.
 *
 * CORRECTED CONTRACT (F3, integration_2.json diag, 2026-07-10): `speed` here is
 * REQUIRED to already be a per-window CONSTANT, resolved by the CALLER at the
 * window's own fixed reference instant (nextLiftAt for a planted foot heading into
 * its toe-off window, liftAt for a swinging foot -- see sync()'s own
 * leftToeScaleSpeed/rightToeScaleSpeed comment) -- NOT `pose.speed` read at the
 * live query time `t`. The ORIGINAL version of this contract passed the raw
 * query-time speed straight through and argued continuity from "speed is a pure
 * function of t, so it agrees at the SAME t on both sides of a boundary" -- true,
 * but it says nothing about speed staying constant ACROSS a whole window, and a
 * real stop-and-go speed transient (e.g. climb clip ~t=39.5s) changes `pose.speed`
 * fast enough WITHIN a single dt=0.05s diag sample, mid-window, to blow the M9b
 * bar on its own (measured 0.172 rad/sample) with no boundary or profile-shape
 * involved at all -- caught via _debugWorstPitch instrumentation after fixing a
 * FIRST, separate M9b mechanism (the swing profile's own shape, see
 * _smoothstep3Prop) moved the worst case here. heelScale (below) never had this
 * problem: it's already built from footPose.strideLen, a per-EVENT constant that
 * doesn't get re-sampled mid-window; toeScale now gets the same treatment, via
 * the caller-resolved `speed` parameter instead of a live re-read.
 *
 * Continuity (spec S6b's own "NO pitch pop at tLand or tLift" requirement, browser-
 * tier M9b): the HEEL-STRIKE window's value at its own start (t=landedAt) is
 * `heelStrikeRad*heelScale` where `heelScale=min(1,strideLen/strideLenRollRefM)` --
 * `strideLen` for a foot at or just past landedAt is "the just-completed swing"
 * (poseAt v2's own "current/most-recent" contract), i.e. the SAME value on both sides
 * of the swing/planted boundary at t=landedAt, so the swing profile's own u=1 value is
 * defined to be EXACTLY `heelStrikeRad*heelScale` too (bit-identical formula, not an
 * approximation) -- true continuity, no pop. The TOE-OFF window's value at its own end
 * (t=nextLiftAt) is `-toeOffRad*toeScale` where `toeScale=min(1,speed/speedRollRefMps)`
 * -- the CALLER resolves `speed` at EXACTLY t=nextLiftAt for this window, and at
 * EXACTLY t=liftAt (the SAME instant: this stance's nextLiftAt IS the following
 * swing's liftAt) for the swing profile's own startVal below, so both sides read the
 * IDENTICAL number -- bit-identical, not the spec's own illustrative "-toeOffRad*0.6"
 * flat constant (that constant does not, in general, equal
 * `-toeOffRad*min(1,speed/speedRollRefMps)` for an arbitrary speed -- using it
 * verbatim would REINTRODUCE a pop the moment speed's scale factor deviates from
 * exactly 0.6; it reads as a typical-case illustration, not a literal coefficient to
 * hardcode, and the explicit numeric continuity bar (M9b) takes precedence).
 */
function _footRollPitch( footPose, t, speed, params ) {

	const heelScale = ( typeof footPose.strideLen === 'number' ) ? Math.min( 1, footPose.strideLen / params.strideLenRollRefM ) : 0;
	const toeScale = Math.min( 1, Math.max( 0, speed ) / params.speedRollRefMps );

	let pitch = 0;

	// HEEL-STRIKE -> FOOT-FLAT: only meaningful once this foot has actually landed at
	// least once (landedAt !== null).
	if ( typeof footPose.landedAt === 'number' ) {

		const dt = t - footPose.landedAt;
		if ( dt >= 0 && dt <= params.rollDownSec ) {

			const ease = _smoothstep( THREE.MathUtils.clamp( dt / params.rollDownSec, 0, 1 ) );
			pitch += params.heelStrikeRad * heelScale * ( 1 - ease );

		}

	}

	// HEEL-OFF -> TOE-OFF: only meaningful while this foot has a scheduled next lift.
	if ( typeof footPose.nextLiftAt === 'number' ) {

		const dt = footPose.nextLiftAt - t;
		if ( dt >= 0 && dt <= params.heelOffSec ) {

			const ease = _smoothstep( THREE.MathUtils.clamp( 1 - dt / params.heelOffSec, 0, 1 ) );
			pitch += - params.toeOffRad * toeScale * ease;

		}

	}

	// SWING: only when actually swinging (footPose.swingU is non-null for a swinging
	// foot per the v1/v2 contract, null when planted).
	if ( footPose.swingU !== null && footPose.swingU !== undefined && ! footPose.planted ) {

		const startVal = - params.toeOffRad * toeScale; // matches the toe-off window's own value AT t=tLift (see this function's own doc)
		const endVal = params.heelStrikeRad * heelScale; // matches the heel-strike window's own value AT t=tLand
		// F3 (2026-07-10): _smoothstep3Prop, not the fixed-0.5-breakpoint
		// _smoothstep3 -- see its own doc for why (M9b footPitch continuity).
		// Endpoint values (hence stance-window continuity) are UNCHANGED.
		pitch = _smoothstep3Prop( footPose.swingU, startVal, params.swingMidDorsiflexRad, endVal );

	}

	return pitch;

}

/**
 * ToeBase's OWN local counter-articulation (relative to Foot -- see PITCH_AXIS's
 * comment: ToeBase shares the lower-body sagittal hinge). Two independent regimes,
 * BOTH derived to connect continuously with the adjacent regime (same discipline as
 * _footRollPitch): during heel-off, ToeBase counter-rotates by (POSITIVE)
 * `toeOffToeRad*ease` so the toe pad stays flat/planted while Foot itself
 * plantarflexes (NEGATIVE) underneath it; during swing, a relax-droop profile that
 * STARTS at that same heel-off end value (continuity at liftoff) and settles to 0
 * (matching the "else identity" mid-stance/heel-strike default) by touchdown.
 */
function _toeBasePitch( footPose, t, speed, params ) {

	const toeScale = Math.min( 1, Math.max( 0, speed ) / params.speedRollRefMps );

	if ( footPose.swingU !== null && footPose.swingU !== undefined && ! footPose.planted ) {

		const startVal = params.toeOffToeRad * toeScale; // matches the heel-off window's own end value (continuity at liftoff)
		return _smoothstep3( footPose.swingU, startVal, params.toeSwingMidDroopRad, 0 );

	}

	if ( typeof footPose.nextLiftAt === 'number' ) {

		const dt = footPose.nextLiftAt - t;
		if ( dt >= 0 && dt <= params.heelOffSec ) {

			const ease = _smoothstep( THREE.MathUtils.clamp( 1 - dt / params.heelOffSec, 0, 1 ) );
			return params.toeOffToeRad * toeScale * ease;

		}

	}

	return 0; // heel-strike window + mid-stance: identity (spec S6 item 4's own "else identity")

}

/**
 * F2a (integration_2.json diag, 2026-07-10): the FIXED point `_pivotAnkleTarget`
 * below rotates the ankle target about -- factored out (plain {x,y,z}, matching
 * this file's own _lastSync-field convention, e.g. leftAnkleTargetWorld a few
 * hundred lines below) so sync() can report the SAME point via
 * `_lastSync.leftFootContact/rightFootContact` (see its own step-6/step-20
 * comments) without duplicating the formula. `pivotForwardOffsetM` is the SIGNED
 * offset such that the contact lands at `plantPos - facing*offset`: F11-SIGN
 * corrected (2026-07-10, incident #18) so the callers pass POSITIVE `+heelBackM` for
 * the heel (contact BEHIND the plant, matching IK_OVERHAUL_SPEC.md §6b's
 * `heelPoint = plantPos - facing*heelBackM`) and NEGATIVE `-toeForwardLenM` for the
 * toe (contact AHEAD); ZERO reproduces the plant point itself (flat/mid-stance). The
 * PRE-#18 convention was the OPPOSITE sign at the call sites (−heelBackM/+toeForwardLen),
 * which placed both pivots on the wrong side and dug the toe into the ground at every
 * landing -- do NOT "restore" it.
 */
function _footContactPoint( plantPos, yaw, pivotForwardOffsetM ) {

	const fwdX = Math.cos( yaw ), fwdY = Math.sin( yaw );
	return {
		x: plantPos.x - fwdX * pivotForwardOffsetM,
		y: plantPos.y - fwdY * pivotForwardOffsetM,
		z: plantPos.z,
	};

}

/**
 * Pivot a foot's flat ankle target about a FIXED heel or toe contact point by
 * `pitch` radians (PITCH_AXIS convention). `pivotForwardOffsetM` is the SIGNED offset
 * defined in `_footContactPoint` above (F11-SIGN / incident #18: `+heelBackM` = heel
 * pivot BEHIND, `-toeForwardLenM` = toe pivot AHEAD); at
 * `pitch=0` this exactly reproduces the pre-existing flat formula
 * `plantPos + ankleHeightM*up` (see the call sites' own comments) -- I2's "contact
 * point stays fixed, ankle moves only via analytic rotation about it" invariant holds
 * BY CONSTRUCTION here: `contactPoint` is computed purely from the foot's CONSTANT
 * plant position/yaw (zero drift by PatientGait's own contract), never from `pitch`
 * or `t`.
 */
function _pivotAnkleTarget( plantPos, yaw, pivotForwardOffsetM, ankleHeightM, pitch, out ) {

	const fwdX = Math.cos( yaw ), fwdY = Math.sin( yaw );

	const contact = _footContactPoint( plantPos, yaw, pivotForwardOffsetM );
	const contactX = contact.x, contactY = contact.y, contactZ = contact.z;

	// Ankle offset from the contact point at pitch=0: pivotForwardOffsetM forward,
	// ankleHeightM up (reduces EXACTLY to the flat case: contactPoint + that offset =
	// plantPos + ankleHeightM*up). Rotate THIS (forward,up) pair about the pitch axis
	// by `pitch` (standard 2D CCW rotation in the forward/up plane -- see
	// _footRollPitch's own call site comment in sync() for the physical sign check:
	// positive pitch/dorsiflex should raise-and-pull-back the ankle relative to a
	// heel pivot, which this formula's d(up)/d(pitch) > 0, d(forward)/d(pitch) < 0 at
	// pitch=0 satisfies).
	const f = pivotForwardOffsetM * Math.cos( pitch ) - ankleHeightM * Math.sin( pitch );
	const u = pivotForwardOffsetM * Math.sin( pitch ) + ankleHeightM * Math.cos( pitch );

	out.set( contactX + fwdX * f, contactY + fwdY * f, contactZ + u );
	return out;

}

// ===========================================================================
// PatientHuman
// ===========================================================================

export class PatientHuman {

	constructor() {

		this.ready = false;
		this.anchor = new THREE.Group();
		this.anchor.name = 'patient_human_anchor';

		this._bones = null;
		this._hipsHeightM = 0.97; // overwritten by the real load-time measurement below (hip PIVOT height -- see load()'s own comment for why not the "Hips" bone's own height); this default only matters if load() somehow never runs before sync()
		this._patientRootNode = null;
		this._attached = false;

		// Rig measurements (populated by load(), all in WORLD METERS -- see each
		// field's own comment at the point it's measured).
		// RIG-3 fix (AGENTS.md incident #15): _legChain holds the REAL bind-pose
		// segment VECTORS (v1=UpLeg->Leg, v2=Leg->Foot -- NOT collinear, see
		// _buildChainGeometry's own comment) + the closed-form constants
		// _solveTwoBoneIK derives from them. This placeholder (a straight-down
		// collinear pair, matching the OLD 0.44/0.44 default) only matters if
		// sync() somehow ran before load() resolves, which the _attached/
		// _schedules guards at the top of sync() already prevent; load() always
		// overwrites it with the real measured geometry before first use.
		this._legChain = _buildChainGeometry( new THREE.Vector3( 0, - 0.44, 0 ), new THREE.Vector3( 0, - 0.44, 0 ), PITCH_AXIS );
		this._maxReachM = 0.995 * this._legChain.bindExtensionM;
		this._minReachM = this._legChain.foldExtensionM + 1e-4;
		this._ankleHeightM = 0.087; // Foot.y - ToeBase.y at bind pose (how far the ankle sits above a flat sole)
		this._toeForwardLenM = 0.107; // horizontal Foot->ToeBase reach
		this._footLateralM = 0.082; // hip pivot's own lateral (X) offset from Hips, one side
		this._hipPivotLocal = { left: new THREE.Vector3(), right: new THREE.Vector3() }; // ANCHOR-local, measured at load
		this._bindHipsPosition = new THREE.Vector3(); // Hips' own RAW LOCAL .position (relative to the 0.01-scaled "Armature" node -- ~100x world meters, see load()'s own comment) -- used ONLY as the base for writing back to Hips.position every sync() call (that property IS interpreted in this raw/scaled space by three.js), NEVER combined arithmetically with a true-meters quantity (see _bindHipsPositionWorldM for that).
		this._bindHipsPositionWorldM = new THREE.Vector3(); // Hips' bind-pose position in TRUE WORLD METERS (getWorldPosition, same unit system as _hipPivotLocal/_bindOffsets) -- this rewrite's own pelvis-compensation/FK-chain-walk arithmetic uses THIS, never _bindHipsPosition (a units bug caught while writing this rewrite's own Node verification probe: _hipPivotLocal.left.sub(_bindHipsPosition) would have subtracted a ~100x-too-large raw value from a true-meters one -- see this rewrite's report).
		this._armatureScale = 1; // Armature node's own uniform scale (measured at load, expected ~0.01) -- the conversion factor between the two representations above (raw = meters / armatureScale).
		this._bindHeadPosition = new THREE.Vector3(); // F8 (integration_2.json diag, 2026-07-10): Head's own RAW LOCAL .position (relative to Neck, SAME raw/scaled convention as _bindHipsPosition -- the whole skeleton shares one exporter unit scale, only Hips happens to sit directly under the scaled Armature node) -- base for the small headBobCounterFrac write in sync() step 19, see that param's own comment for why.

		// Arm chain measurements. sync()'s OWN per-frame posing still only IK-solves
		// the RIGHT arm (cane grip) -- the LEFT arm is pure FK there (see sync()'s
		// step 18) and never needs a reach solve at runtime. Both sides still get a
		// full chain (RIG-3 fix, AGENTS.md #15: placeholder collinear pair here;
		// _measureSpineArmChain overwrites both at load() with the real,
		// near-collinear-but-not-exactly-so measured vectors) because load()'s own
		// `_resolveElbowSign` DOES two-bone-solve BOTH sides once (a load-time-only
		// probe, unrelated to sync()'s per-frame FK-vs-IK split) -- see that
		// method's own comment for why the arm gets a chain PER SIDE, unlike the
		// legs' single shared chain above (an arm's mirror flips its DOMINANT
		// component; a leg's mirror flips only its near-zero one).
		this._armChain = {
			left: _buildChainGeometry( new THREE.Vector3( 0.278, 0, 0 ), new THREE.Vector3( 0.283, 0, 0 ), _ARM_HINGE_AXIS ),
			right: _buildChainGeometry( new THREE.Vector3( - 0.278, 0, 0 ), new THREE.Vector3( - 0.283, 0, 0 ), _ARM_HINGE_AXIS ),
		};
		this._armMaxReachM = 0.995 * this._armChain.right.bindExtensionM;
		this._armMinReachM = this._armChain.right.foldExtensionM + 1e-4;
		this._armRestDir = { left: new THREE.Vector3( 1, 0, 0 ), right: new THREE.Vector3( - 1, 0, 0 ) };
		// Elbow-bend sign per side, resolved at load() time by a tiny IK->FK
		// self-probe (never assumed -- see _resolveElbowSign's own comment).
		this._elbowSign = { left: - 1, right: 1 };

		// NEW: bind-pose offset VECTORS (anchor-local meters, parent->child, measured
		// exactly like _hipPivotLocal above) for the Hips->Spine->Spine1->Spine2->
		// RightShoulder->RightArm chain -- the analytic FK walk sync() uses to find
		// the RIGHT shoulder pivot's CURRENT (pelvis/spine-rotated) position every
		// call (spec S6 item 1's "recompute pivot positions analytically" discipline,
		// extended one level: the arm's pivot sits several rotating bones below Hips,
		// not directly on it like a leg's hip pivot).
		this._bindOffsets = {
			spine: new THREE.Vector3(), spine1: new THREE.Vector3(), spine2: new THREE.Vector3(),
			rightShoulder: new THREE.Vector3(), rightArm: new THREE.Vector3(),
		};

		// NEW: hand bind-pose local axes (anchor-local at bind pose, where the whole
		// ancestor chain's rotation is ~identity -- see load()'s own identity-chain
		// check, extended to the arm chain too), measured via getWorldPosition deltas
		// / a finger-spread cross product (see _measureHandAxes). Used by the right
		// hand's wrist "palm faces down the shaft" orientation.
		this._handFingersForward = { left: new THREE.Vector3( 1, 0, 0 ), right: new THREE.Vector3( - 1, 0, 0 ) };
		this._handPalmNormal = { left: new THREE.Vector3( 0, - 1, 0 ), right: new THREE.Vector3( 0, - 1, 0 ) };
		this._fingersFound = { left: false, right: false };

		// PatientGait.js state (populated by buildGait()).
		this._terrain = null;
		this._schedules = null; // { follow: schedule, climb: schedule }
		this._gaitParams = DEFAULT_GAIT_PARAMS;

		this.ikSelfCheckFailed = false;

		// Diagnostic snapshot from the most recent sync() call -- see sync()'s own
		// comment at the point it's populated. Read by main.js's patientDiag.
		this._lastSync = null;

		// NEW: the cane prop (PatientCane.js) -- built here (cheap, no GPU cost until
		// parented into the scene by attachTo()) so it exists regardless of load-order
		// races between this constructor and load()'s own async GLTF fetch.
		this._cane = buildCaneGroup( CANE_PARAMS );
		this._cane.visible = false;

		// I10 "log once" flags (spec: a safe-disable guard must announce what's off
		// and why, exactly once, not every frame).
		this._loggedNoPhaseC = false;
		this._loggedNoSupport = false;
		this._loggedNoCane = false;
		this._loggedNoFootRollFields = false;
		this._loggedCaneUnreachable = false;
		this._loggedFingersMissing = false;

		// Reusable scratch objects (avoid per-frame allocation in the hot sync() path).
		// Every field here is written-then-read within a single sync() call (several
		// fields deliberately REUSED across non-overlapping steps, e.g. q0-q3 -- see
		// sync()'s own inline comments at each reuse site for why that specific reuse
		// is safe, i.e. the prior value is either already consumed/copied into a bone
		// or genuinely no longer needed).
		this._scratch = {
			v0: new THREE.Vector3(), v1: new THREE.Vector3(),
			q0: new THREE.Quaternion(), q1: new THREE.Quaternion(), q2: new THREE.Quaternion(), q3: new THREE.Quaternion(),
			rootQuatInv: new THREE.Quaternion(),
			qFreeze: new THREE.Quaternion(),
			hipsPos: new THREE.Vector3(),
			leftPivot: new THREE.Vector3(), rightPivot: new THREE.Vector3(),
			chainPos: new THREE.Vector3(), chainQuat: new THREE.Quaternion(),
			armPivot: new THREE.Vector3(), armTargetLocal: new THREE.Vector3(),
			caneAxis: new THREE.Vector3(),
			armAchievedLocal: new THREE.Vector3(), armAchievedWorld: new THREE.Vector3(), caneHandleAdj: new THREE.Vector3(),
			handDesiredAbs: new THREE.Quaternion(), foreArmCumQuat: new THREE.Quaternion(),
		};

		this._canePose = { tip: new THREE.Vector3(), axisWorld: new THREE.Vector3(), handle: new THREE.Vector3(), leanRad: 0 };

	}

	/**
	 * Kick off the GLTFLoader. Returns a Promise that resolves once ready (or logs +
	 * leaves this.ready=false on failure -- degrades to "no patient shown", matching
	 * how loadPlaceholder() already handles a total robot.glb load failure elsewhere
	 * in this app).
	 */
	async load() {

		try {

			const loader = new GLTFLoader();
			const gltf = await new Promise( ( resolve, reject ) => loader.load( XBOT_URL, resolve, undefined, reject ) );

			const scene = gltf.scene || gltf.scenes[ 0 ];
			scene.updateMatrixWorld( true );

			const bones = {};
			for ( const [ key, name ] of Object.entries( BONE_NAMES ) ) {

				bones[ key ] = scene.getObjectByName( name );
				if ( ! bones[ key ] ) throw new Error( `Xbot.glb: bone ${name} not found` );

			}

			// --- Measure the rig, in WORLD METERS, at bind pose (this rewrite no
			// longer plays a mixer at all, so "bind pose" is simply the scene's own
			// loaded state -- no need to zero out a mixer first). All lengths/offsets
			// via getWorldPosition() deltas, per this module's own header/AGENTS.md:
			// Xbot's skeleton sits under an "Armature" node with uniform scale 0.01, so
			// bone-LOCAL .position values are ~100x world meters -- world-space
			// measurement sidesteps that entirely. ---
			const hipsWorld = new THREE.Vector3();
			bones.hips.getWorldPosition( hipsWorld );
			this._bindHipsPosition = bones.hips.position.clone();
			this._bindHipsPositionWorldM.copy( hipsWorld ); // see this field's own constructor comment for why a SEPARATE true-meters copy is kept
			this._bindHeadPosition = bones.head.position.clone(); // F8: see this field's own constructor comment

			// Armature's own uniform scale (0.01, confirmed via this rewrite's own Node
			// probe parse of the raw glTF JSON) -- the conversion factor between
			// _bindHipsPosition (raw) and _bindHipsPositionWorldM (meters). Measured,
			// not hardcoded, so a future re-export at a different scale still works;
			// warns (does not throw -- degrades to "pelvis bob/shift math will be off
			// by the mismatch" rather than blocking the whole patient) if the loaded
			// rig's Armature turns out non-uniformly scaled, which this rewrite's own
			// conversion math assumes it is not.
			const armatureNode = bones.hips.parent;
			if ( armatureNode ) {

				this._armatureScale = armatureNode.scale.x;
				if ( Math.abs( armatureNode.scale.y - armatureNode.scale.x ) > 1e-6 || Math.abs( armatureNode.scale.z - armatureNode.scale.x ) > 1e-6 ) {

					console.warn( '[blueprint-viewer] PatientHuman: Armature scale is non-uniform -- pelvis bob/shift raw-unit conversion (which assumes a single scalar) may be slightly wrong.' );

				}

			}

			const leftUpLegWorld = new THREE.Vector3();
			bones.leftUpLeg.getWorldPosition( leftUpLegWorld );
			const rightUpLegWorld = new THREE.Vector3();
			bones.rightUpLeg.getWorldPosition( rightUpLegWorld );
			const leftLegWorld = new THREE.Vector3();
			bones.leftLeg.getWorldPosition( leftLegWorld );
			const leftFootWorld = new THREE.Vector3();
			bones.leftFoot.getWorldPosition( leftFootWorld );
			const leftToeWorld = new THREE.Vector3();
			bones.leftToeBase.getWorldPosition( leftToeWorld );

			// _hipsHeightM: the height subtracted from patient_root.z to place the
			// anchor (see sync()'s own comment). Measured from the HIP PIVOT (UpLeg
			// bone, averaged left/right), NOT from the "Hips" bone itself -- see
			// AGENTS.md incident #8's own reasoning (unchanged by this rewrite).
			this._hipsHeightM = ( leftUpLegWorld.y + rightUpLegWorld.y ) / 2;

			// RIG-3 fix (AGENTS.md incident #15): v1/v2 are the REAL bind-pose
			// segment VECTORS (not just scalar lengths along an assumed-straight-
			// down rest direction) -- Xbot's real UpLeg->Leg/Leg->Foot offsets are
			// NOT collinear (~4.21 deg angle between them; one segment tilts
			// ~3.84 deg off pure "straight down" -- see this fix's own report for
			// the full numeric dump). _buildChainGeometry precomputes the
			// closed-form constants _solveTwoBoneIK needs so sync()'s hot path
			// never repeats them. Shared between both legs (mirror-symmetry
			// measured to <2e-5 m, same assumption the L1/L2 scalars already
			// shared) -- using the LEFT side's measurement, same precedent as
			// _hipPivotLocal/_footLateralM below.
			const v1Leg = leftLegWorld.clone().sub( leftUpLegWorld );
			const v2Leg = leftFootWorld.clone().sub( leftLegWorld );
			this._legChain = _buildChainGeometry( v1Leg, v2Leg, PITCH_AXIS );
			this._maxReachM = 0.995 * this._legChain.bindExtensionM;
			this._minReachM = this._legChain.foldExtensionM + 1e-4;

			this._ankleHeightM = leftFootWorld.y - leftToeWorld.y;
			this._toeForwardLenM = Math.hypot( leftToeWorld.x - leftFootWorld.x, leftToeWorld.z - leftFootWorld.z );

			this._hipPivotLocal.left.copy( leftUpLegWorld );
			this._hipPivotLocal.right.copy( rightUpLegWorld );
			this._footLateralM = Math.abs( leftUpLegWorld.x - hipsWorld.x );

			// Load-time sanity check (task requirement, UNCHANGED from the pre-
			// existing module): the Hips ancestor chain's bind-pose world quaternion
			// should be ~identity -- this module's anchor-local axis math (now
			// extended to the whole spine/arm chain, not just the legs) assumes
			// EVERY bone's bind-pose LOCAL rotation is ~identity, confirmed for the
			// new bones too by this rewrite's own Node probe (RightArm/RightForeArm/
			// RightHand/RightShoulder/LeftHand/ToeBase all measured ~1e-8 rad from
			// identity).
			const hipsWorldQuat = new THREE.Quaternion();
			bones.hips.getWorldQuaternion( hipsWorldQuat );
			const identityAngle = hipsWorldQuat.angleTo( new THREE.Quaternion() );
			if ( identityAngle > 1e-4 ) {

				console.warn(
					'[blueprint-viewer] PatientHuman: Hips bind-pose world quaternion is not ~identity ' +
					`(angle ${THREE.MathUtils.radToDeg( identityAngle ).toFixed( 3 )} deg from identity) -- ` +
					'this module\'s anchor-local IK math assumes the Hips ancestor chain carries no rotation; ' +
					'leg placement may be silently wrong.',
				);

			}

			// --- NEW (this rewrite): spine/shoulder/arm chain measurements ---
			this._measureSpineArmChain( bones );
			this._measureHandAxes( bones );
			this._resolveElbowSign( bones );
			this._fingersFound.right = this._setHandGripPose( scene, 'Right', 0.85 ); // firm curled grip
			this._fingersFound.left = this._setHandGripPose( scene, 'Left', 0.25 ); // relaxed slight curl
			if ( ! this._fingersFound.right || ! this._fingersFound.left ) {

				console.log(
					'[blueprint-viewer] PatientHuman: some/all finger bones not found on this rig -- ' +
					'hand grip pose is partial or skipped (cosmetic only, does not affect IK/placement).',
				);
				this._loggedFingersMissing = true;

			}

			this._scene = scene;
			this._bones = bones;

			this._runIkSelfCheck();

			this.ready = true;

		} catch ( error ) {

			console.error( '[blueprint-viewer] PatientHuman failed to load; no patient will be shown:', error );

		}

	}

	/**
	 * Measure the Hips->Spine->Spine1->Spine2->{Left,Right}Shoulder->{Left,Right}Arm
	 * ->{Left,Right}ForeArm->{Left,Right}Hand chain: bind offset VECTORS (anchor-local
	 * meters, parent->child, via getWorldPosition deltas -- same technique as the
	 * pre-existing leg measurements) for the spine/shoulder links sync()'s analytic FK
	 * walk needs, plus PER-SIDE two-bone chain geometry (RIG-3 fix, AGENTS.md #15 --
	 * v1=Arm->ForeArm, v2=ForeArm->Hand, real bind vectors, not just lengths+a shared
	 * rest direction) for the two-bone solver, and the arm's own rest direction for
	 * the left-arm FK hang-quaternion. Called once from load().
	 */
	_measureSpineArmChain( bones ) {

		const wp = ( bone ) => { const v = new THREE.Vector3(); bone.getWorldPosition( v ); return v; };

		const hipsW = wp( bones.hips );
		const spineW = wp( bones.spine );
		const spine1W = wp( bones.spine1 );
		const spine2W = wp( bones.spine2 );
		const rShoulderW = wp( bones.rightShoulder );
		const rArmW = wp( bones.rightArm );
		const rForeArmW = wp( bones.rightForeArm );
		const rHandW = wp( bones.rightHand );
		const lArmW = wp( bones.leftArm );
		const lForeArmW = wp( bones.leftForeArm );
		const lHandW = wp( bones.leftHand );

		this._bindOffsets.spine.subVectors( spineW, hipsW );
		this._bindOffsets.spine1.subVectors( spine1W, spineW );
		this._bindOffsets.spine2.subVectors( spine2W, spine1W );
		this._bindOffsets.rightShoulder.subVectors( rShoulderW, spine2W );
		this._bindOffsets.rightArm.subVectors( rArmW, rShoulderW );

		// RIG-3 fix (AGENTS.md #15): PER-SIDE chain (unlike the legs' single shared
		// chain) -- an arm's v1/v2 mirror via a lateral SIGN FLIP (dominant
		// component IS the mirrored/lateral axis, measured: v1_right.x=-0.27842,
		// v1_left.x=+0.27842), which a single shared chain can't represent (a
		// leg's mirror flip only touches its near-zero x-component). Measured:
		// the arm's v1/v2 are within ~2e-4 deg of perfectly collinear (vs the
		// leg's ~4.21 deg) -- the arm was already "immune in spirit" (the
		// pre-existing `_armRestDir` was measured from real offsets, unlike the
		// legs' hardcoded `_REST_DIR`); this just routes it through the same
		// real-vector solver machinery as the legs, for consistency and so a
		// future rig swap with a less-straight arm is handled automatically.
		this._armChain.right = _buildChainGeometry( rForeArmW.clone().sub( rArmW ), rHandW.clone().sub( rForeArmW ), _ARM_HINGE_AXIS );
		this._armChain.left = _buildChainGeometry( lForeArmW.clone().sub( lArmW ), lHandW.clone().sub( lForeArmW ), _ARM_HINGE_AXIS );
		this._armMaxReachM = 0.995 * this._armChain.right.bindExtensionM;
		this._armMinReachM = this._armChain.right.foldExtensionM + 1e-4;

		this._armRestDir.right.copy( this._armChain.right.v1 ).normalize();
		this._armRestDir.left.copy( this._armChain.left.v1 ).normalize();

	}

	/** Measure each hand's "fingers point this way" and "palm faces this way" bind-
	 *  pose LOCAL axes (~world axes at bind pose, ancestor chain ~identity -- same
	 *  reasoning as _hipPivotLocal). Fingers-forward = Hand->HandMiddle1 direction
	 *  (this rewrite's own Node probe: (-0.9999...,~0,~0) for the right hand, i.e.
	 *  Xbot-local X, matching the whole arm chain's own rest direction -- fingers
	 *  extend the arm's straight line in a T-pose). Palm-normal = normalize(cross(
	 *  Hand->Index1, Hand->Pinky1)) (the probe found this ~(0.06,-0.998,0.005) for the
	 *  right hand -- dominantly Xbot-local -Y, i.e. "palm faces down" in the T-pose,
	 *  the common Mixamo convention) -- a small (~3 deg) non-orthogonality vs the
	 *  fingers-forward axis is expected (a real hand's finger spread isn't a perfect
	 *  right angle) and is exactly what `_aimWithTwist`'s own `_projectPerp` step
	 *  exists to resolve (projects the reference axis into the plane perpendicular to
	 *  the aim direction before using it). Best-effort: falls back to the constructor's
	 *  defaults (also ~correct, from this rewrite's own probe numbers) if a finger
	 *  bone is missing -- fingers are cosmetic (see _setHandGripPose), but the wrist
	 *  orientation formula that CONSUMES these axes is not, so this method degrades
	 *  gracefully rather than throwing. */
	_measureHandAxes( bones ) {

		const wp = ( bone ) => { const v = new THREE.Vector3(); bone.getWorldPosition( v ); return v; };

		for ( const side of [ 'left', 'right' ] ) {

			const handBone = side === 'left' ? bones.leftHand : bones.rightHand;
			const handW = wp( handBone );
			const middle1 = handBone.getObjectByName( `mixamorig${side === 'left' ? 'Left' : 'Right'}HandMiddle1` );
			const index1 = handBone.getObjectByName( `mixamorig${side === 'left' ? 'Left' : 'Right'}HandIndex1` );
			const pinky1 = handBone.getObjectByName( `mixamorig${side === 'left' ? 'Left' : 'Right'}HandPinky1` );

			if ( middle1 ) this._handFingersForward[ side ].subVectors( wp( middle1 ), handW ).normalize();

			if ( index1 && pinky1 ) {

				const toIndex = wp( index1 ).sub( handW ).normalize();
				const toPinky = wp( pinky1 ).sub( handW ).normalize();
				const normal = new THREE.Vector3().crossVectors( toIndex, toPinky ).normalize();
				// Right/left mirror: keep whichever sign points toward -Y (palm-down
				// in bind pose, per this method's own doc) so both hands share the
				// same "palm faces this way" semantics regardless of cross-product
				// winding direction.
				if ( normal.y > 0 ) normal.multiplyScalar( - 1 );
				this._handPalmNormal[ side ].copy( normal );

			}

		}

	}

	/**
	 * Resolve the per-side sign in `ForeArm.quaternion = axisAngle(_ARM_HINGE_AXIS,
	 * sign*bend)` by a tiny IK->FK probe (NEVER assumed -- same discipline as
	 * PITCH_AXIS's own sign note): solve the two-bone IK for a target that requires a
	 * KNOWN, nonzero bend, apply both candidate signs via real Object3D parenting, and
	 * keep whichever reproduces the target (restores bind pose afterward, same
	 * pattern as _runIkSelfCheck).
	 */
	_resolveElbowSign( bones ) {

		const armWorldPos = ( bone ) => { const v = new THREE.Vector3(); bone.getWorldPosition( v ); return v; };
		const cases = [
			{ side: 'left', arm: bones.leftArm, foreArm: bones.leftForeArm, hand: bones.leftHand, pivot: armWorldPos( bones.leftArm ) },
			{ side: 'right', arm: bones.rightArm, foreArm: bones.rightForeArm, hand: bones.rightHand, pivot: armWorldPos( bones.rightArm ) },
		];

		for ( const c of cases ) {

			const restDir = this._armRestDir[ c.side ];
			// A target that bends the elbow ~60 degrees: reach = a value strictly
			// between minReach/maxReach, direction offset from restDir toward the
			// hinge-perpendicular plane (mirrors _runIkSelfCheck's own "forward+down"
			// style test offsets).
			const reach = ( this._armMinReachM + this._armMaxReachM ) / 2;
			const target = c.pivot.clone().add( restDir.clone().multiplyScalar( reach * 0.7 ) ).add( _ARM_HINGE_AXIS.clone().multiplyScalar( - reach * 0.3 ) );
			const pole = new THREE.Vector3( 0, 0, - 1 );
			const { hipQuat, kneeBend } = _solveTwoBoneIK( c.pivot, target, pole, this._armMaxReachM, this._armMinReachM, this._armChain[ c.side ] );

			let bestSign = 1, bestErr = Infinity;
			for ( const sign of [ 1, - 1 ] ) {

				c.arm.quaternion.copy( hipQuat );
				c.foreArm.quaternion.setFromAxisAngle( _ARM_HINGE_AXIS, sign * kneeBend );
				c.arm.updateMatrixWorld( true );
				const achieved = new THREE.Vector3();
				c.hand.getWorldPosition( achieved );
				const err = achieved.distanceTo( target );
				if ( err < bestErr ) { bestErr = err; bestSign = sign; }

			}

			this._elbowSign[ c.side ] = bestSign;

			c.arm.quaternion.identity();
			c.foreArm.quaternion.identity();
			c.arm.updateMatrixWorld( true );

		}

	}

	/** Best-effort static finger-curl pose (spec S5: "static curled grip pose on the
	 *  right hand set once at load ... left hand: relaxed slight curl"). `curlRad` is
	 *  a MAGNITUDE (0.85 rad firm grip for the right/cane hand, 0.25 rad relaxed for
	 *  the left, per load()'s own call sites); sign per side matches each hand's own
	 *  measured fingers-forward direction (see _FORWARD_AXIS_LOCAL/_measureHandAxes'
	 *  own doc: right hand fingers point -X, so curling toward -Y is a POSITIVE
	 *  rotation about +Z; left hand fingers point +X, so the same curl is NEGATIVE
	 *  about +Z -- cross-product-derived, not guessed). Returns true iff every finger
	 *  bone was found (best-effort otherwise -- a finger bone rig swap should not
	 *  break IK/placement, only this cosmetic detail). Never touched again after
	 *  load() -- these bones are simply never written by sync(), so they hold this
	 *  pose forever. */
	_setHandGripPose( scene, sideCap, curlRad ) {

		const sign = sideCap === 'Right' ? 1 : - 1;
		let allFound = true;
		for ( const finger of _FINGER_NAMES ) {

			for ( const seg of _FINGER_SEGMENTS ) {

				const bone = scene.getObjectByName( `mixamorig${sideCap}Hand${finger}${seg}` );
				if ( ! bone ) { allFound = false; continue; }
				// Segment 1 (the knuckle/MCP joint) curls less than the outer two
				// segments, matching a natural grip's own progressive curl.
				const segScale = seg === 1 ? 0.6 : 1.0;
				bone.quaternion.setFromAxisAngle( _FORWARD_AXIS_LOCAL, sign * curlRad * segScale );

			}

		}
		return allFound;

	}

	/**
	 * Parent the loaded model under isaacWorldNode (a SIBLING of patient_root, NOT a
	 * child of it -- the mesh's own SkinnedMesh binding needs a constant-transform
	 * ancestor; see AGENTS.md's incident ledger), remember patientRootNode (read
	 * every sync() call), apply the shared blueprint tint, and (NEW this rewrite)
	 * attach the cane prop as a SIBLING of the anchor under the SAME isaacWorldNode
	 * (spec S5: never parented to the hand -- the hand is IK'd to the cane). Safe to
	 * call before load() resolves -- callers should await load() first regardless
	 * (main.js does).
	 */
	attachTo( isaacWorldNode, patientRootNode, tintMaterial ) {

		if ( ! this.ready || this._attached ) return;

		this._patientRootNode = patientRootNode;
		this.anchor.add( this._scene );
		isaacWorldNode.add( this.anchor );
		isaacWorldNode.add( this._cane );
		this._attached = true;

		this._scene.traverse( ( node ) => {

			if ( ! node.isMesh ) return;

			const oldMaterials = Array.isArray( node.material ) ? node.material : [ node.material ];
			for ( const mat of oldMaterials ) {

				if ( mat ) mat.dispose();

			}

			node.material = tintMaterial;
			// Elderly-patient colouring: per-vertex COLOR from 3D body region (the
			// shared tintMaterial has vertexColors=true). NOT a diffuse texture --
			// Xbot's UVs overlap, so a map cross-contaminates the torso; see
			// paintPatientRegionColors' block comment.
			paintPatientRegionColors( node.geometry );
			// 2026-07-10 lighting pass: the patient now casts a shadow (adds real
			// depth/grounding to the scene) but still doesn't receive one --
			// self-shadowing a skinned mesh from one directional key light reads as
			// noisy speckle across the limbs rather than believable form.
			node.castShadow = true;
			node.receiveShadow = false;

			// Same SkinnedMesh frustum-culling gotcha as this viewer's old custom
			// skin nodes (see AGENTS.md): the bind-pose bounding sphere sits near
			// this mesh's own (fixed, near-anchor) node location, not wherever the
			// bones actually place the skinned vertices, so three.js wrongly culls
			// the whole object once the character walks far from the anchor.
			if ( node.isSkinnedMesh ) node.frustumCulled = false;

		} );

	}

	/**
	 * Build the per-clip footfall (+ cane, per PatientGait v2) schedule from
	 * `phaseClips` (a { follow, climb } map of THREE.AnimationClip, as loaded by
	 * main.js from robot.glb) and `stairSpec`/`landingFarX` (robot.meta.json's own
	 * fields). Must run AFTER load() resolves (needs the measured rig proportions)
	 * and BEFORE the first sync() call. Idempotent-safe to call more than once
	 * (rebuilds from scratch). UNCHANGED by this rewrite beyond passthrough -- v2's
	 * new DEFAULT_GAIT_PARAMS keys (cane*, phaseC/support-affecting tunables) are
	 * ALREADY covered by the existing `{...DEFAULT_GAIT_PARAMS, ...overrides}` spread
	 * (additive per spec S3), so this method needs no new code for them.
	 */
	buildGait( phaseClips, stairSpec, landingFarX ) {

		if ( ! this.ready ) return;

		this._terrain = buildTerrain( stairSpec, landingFarX );

		this._gaitParams = {
			...DEFAULT_GAIT_PARAMS,
			footLateral: this._footLateralM,
			toeForwardLen: this._toeForwardLenM,
		};

		this._schedules = {};
		for ( const [ phaseName, clip ] of Object.entries( phaseClips ) ) {

			if ( ! clip ) continue;

			const posTrack = clip.tracks.find( ( t ) => t.name === 'patient_root.position' );
			const quatTrack = clip.tracks.find( ( t ) => t.name === 'patient_root.quaternion' );
			if ( ! posTrack || ! quatTrack ) {

				console.error(
					`[blueprint-viewer] PatientHuman.buildGait: clip "${phaseName}" has no patient_root position/quaternion track -- ` +
					'patient will not be posed for this phase.',
				);
				continue;

			}

			const samples = extractPathSamples( posTrack.times, posTrack.values, quatTrack.times, quatTrack.values );
			this._schedules[ phaseName ] = buildSchedule( samples, this._terrain, this._gaitParams );

		}

	}

	/**
	 * LOAD-TIME FK SELF-CHECK (mandatory per the task spec, EXTENDED this rewrite per
	 * spec S6 item 9 / the RIG task brief item 9): (a) the pre-existing 6 leg targets
	 * (bar ≤5mm, tightened from ≤1cm by the RIG-3 fix -- see below); (b) the SAME 6
	 * targets re-solved through a NONZERO test pelvis rotation, exercising the new
	 * hip-pivot + UpLeg-local compensation path (bar ≤5mm); (c) an arm-IK round-trip
	 * case (right arm, bar ≤5mm, matching the leg bar's own units/order-of-magnitude
	 * even though the task's separate Node probe carries the tighter 1e-6 m arm bar
	 * for the pure-math case -- this browser-tier check additionally exercises the
	 * REAL loaded bones/parenting, not just plain quaternion math). Restores
	 * EVERYTHING (both legs + right arm) to bind pose afterward -- this runs once at
	 * load, before any patient is ever shown.
	 *
	 * RIG-3 fix (AGENTS.md incident #15) tightened the bar from 0.01 m to 0.005 m:
	 * with the real-bind-vector solver, cases (a)/(c) and 5 of (b)'s 6 targets solve
	 * to machine precision (~1e-15 m, this fix's own Node probe). ONE case (b) target
	 * -- offset #6, "near-full-extension" (0.99x maxReach), reused verbatim from case
	 * (a) per this method's own long-standing pattern -- combined with the test
	 * pelvis's ~3.5 cm pivot shift, asks for a reach of ~0.898 m, which EXCEEDS the
	 * chain's absolute physical maximum (L1+L2 ~= 0.889 m, fully straightened) by
	 * ~0.9 cm: NO solver, however correct, can hit that exact target -- confirmed by
	 * running the OLD (pre-fix) solver through the same case, which clamps
	 * identically. This is a property of reusing case (a)'s already-near-limit offset
	 * under an UNRELATED pivot displacement, not a solver defect, and not something
	 * the real sync() pipeline would ever present to the solver unmodified (sync()'s
	 * own reach-lowering step handles it there).
	 *
	 * ORCHESTRATOR TEST-DESIGN DECISION (2026-07-10, supersedes RIG-3's "left AS-IS"
	 * note): every loop below measures the achieved bone position against the
	 * SOLVER'S OWN COMMITMENT -- `pivot + normalize(target - pivot) * achievedReach`
	 * -- not against the raw target. For every unclamped case the two are the SAME
	 * point (achievedReach == |target - pivot| exactly), so this loses no power:
	 * the incident-#15 FK bias (~3 cm) violated the solver's own commitment on
	 * unclamped targets and would still be caught at 6x the bar. For the one
	 * physically-unreachable case it asserts the only property a correct solver can
	 * promise (a clean clamp along the target direction) instead of red-flagging
	 * geometry no solver could achieve -- a permanently-failing self-check that
	 * everyone is told to ignore trains alarm fatigue and makes patientDiag's
	 * `pass` unreachable (it ANDs `!ikSelfCheckFailed`).
	 */
	_runIkSelfCheck() {

		const b = this._bones;
		const hipPivot = this._hipPivotLocal.left;
		const pole = new THREE.Vector3( 0, 0, 1 );

		const testOffsets = [
			new THREE.Vector3( 0, - this._maxReachM * 0.9, 0 ), // straight down, standing
			new THREE.Vector3( 0, - this._maxReachM * 0.7, this._maxReachM * 0.3 ), // forward+down
			new THREE.Vector3( 0, - this._maxReachM * 0.7, - this._maxReachM * 0.3 ), // backward+down
			new THREE.Vector3( 0, - this._maxReachM * 0.5, this._maxReachM * 0.35 ), // forward+up (high step)
			new THREE.Vector3( this._footLateralM * 0.3, - this._maxReachM * 0.8, this._maxReachM * 0.1 ), // slight lateral
			new THREE.Vector3( 0, - this._maxReachM * 0.99, 0 ), // near-full extension
		];

		let worstError = 0;

		// The solver's own committed point for (pivot, target, achievedReach) -- equals
		// `target` exactly whenever the solve was not reach-clamped (see the doc
		// comment's test-design decision above). Load-time only; allocations fine.
		const solverCommitment = ( pivot, target, achievedReach ) =>
			pivot.clone().add( target.clone().sub( pivot ).normalize().multiplyScalar( achievedReach ) );

		// (a) pre-existing: zero pelvis rotation.
		for ( const offset of testOffsets ) {

			const target = hipPivot.clone().add( offset );
			const { hipQuat, kneeBend, achievedReach } = _solveTwoBoneIK(
				hipPivot, target, pole, this._maxReachM, this._minReachM, this._legChain,
			);

			b.leftUpLeg.quaternion.copy( hipQuat );
			b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - kneeBend );
			b.leftUpLeg.updateMatrixWorld( true );

			const achieved = new THREE.Vector3();
			b.leftFoot.getWorldPosition( achieved );

			worstError = Math.max( worstError, achieved.distanceTo( solverCommitment( hipPivot, target, achievedReach ) ) );

		}

		b.leftUpLeg.quaternion.identity();
		b.leftLeg.quaternion.identity();
		b.leftUpLeg.updateMatrixWorld( true );

		// (b) NEW: nonzero TEST pelvis rotation, exercising the pivot + UpLeg-local
		// compensation path sync() uses every frame (spec S6 item 1's own bar: "run
		// with a nonzero test pelvis rotation, bar stays <=1cm"). Values match the
		// Node probe's own pelvis-compensation case (list=0.05, yaw=0.06 rad).
		const testPelvisQuat = new THREE.Quaternion()
			.setFromAxisAngle( _FORWARD_AXIS_LOCAL, 0.05 )
			.multiply( new THREE.Quaternion().setFromAxisAngle( _UP_AXIS_LOCAL, 0.06 ) );
		// TRUE METERS (matches _hipPivotLocal's own unit system) -- NOT
		// this._bindHipsPosition (RAW, ~100x meters under the scaled Armature node;
		// see that field's own constructor comment). Mixing the two here was a real
		// bug caught while writing this rewrite's own Node verification probe.
		const testHipsPosM = this._bindHipsPositionWorldM.clone().add( new THREE.Vector3( 0.025, 0.018, 0 ) );

		for ( const offset of testOffsets ) {

			const target = hipPivot.clone().add( offset ); // target stays a fixed anchor-local point, same as case (a)
			const pivotOffset = hipPivot.clone().sub( this._bindHipsPositionWorldM ).applyQuaternion( testPelvisQuat );
			const pivotShifted = testHipsPosM.clone().add( pivotOffset );

			const { hipQuat, kneeBend, achievedReach } = _solveTwoBoneIK(
				pivotShifted, target, pole, this._maxReachM, this._minReachM, this._legChain,
			);

			// Hips.position is interpreted in RAW (Armature-scaled) units by
			// three.js, so the METERS-space test delta must be converted (divide by
			// armatureScale) before writing -- see _bindHipsPosition's own comment.
			b.hips.position.copy( this._bindHipsPosition );
			b.hips.position.x += 0.025 / this._armatureScale;
			b.hips.position.y += 0.018 / this._armatureScale;
			b.hips.quaternion.copy( testPelvisQuat );
			b.leftUpLeg.quaternion.copy( testPelvisQuat.clone().invert().multiply( hipQuat ) );
			b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - kneeBend );
			b.hips.updateMatrixWorld( true );

			const achieved = new THREE.Vector3();
			b.leftFoot.getWorldPosition( achieved );

			// vs the solver's commitment measured from the SHIFTED pivot -- the one
			// case whose required reach exceeds the physical chain length asserts a
			// clean clamp along the target direction here (see the doc comment).
			worstError = Math.max( worstError, achieved.distanceTo( solverCommitment( pivotShifted, target, achievedReach ) ) );

		}

		b.hips.position.copy( this._bindHipsPosition );
		b.hips.quaternion.identity();
		b.leftUpLeg.quaternion.identity();
		b.leftLeg.quaternion.identity();
		b.hips.updateMatrixWorld( true );

		// (c) NEW: right-arm IK round-trip (bind pelvis/spine -- the analytic
		// FK-chain pivot compensation itself is exercised by the Node probe's own
		// arm case + is structurally the SAME formula as (b) above, just walked over
		// 4 links instead of 1).
		const armPivot = new THREE.Vector3();
		b.rightArm.getWorldPosition( armPivot ); // bind pose, ancestor chain identity -- world == anchor-local here
		const armPole = new THREE.Vector3( 0, 0, - 1 );
		const armTestOffsets = [
			this._armRestDir.right.clone().multiplyScalar( this._armMaxReachM * 0.9 ),
			this._armRestDir.right.clone().multiplyScalar( this._armMaxReachM * 0.6 ).add( new THREE.Vector3( 0, - this._armMaxReachM * 0.3, 0 ) ),
			this._armRestDir.right.clone().multiplyScalar( this._armMaxReachM * 0.5 ).add( new THREE.Vector3( 0, - this._armMaxReachM * 0.4, - this._armMaxReachM * 0.2 ) ),
		];
		let worstArmError = 0;
		for ( const offset of armTestOffsets ) {

			const target = armPivot.clone().add( offset );
			const { hipQuat: armQuat, kneeBend: elbowBend, achievedReach } = _solveTwoBoneIK(
				armPivot, target, armPole, this._armMaxReachM, this._armMinReachM, this._armChain.right,
			);

			b.rightArm.quaternion.copy( armQuat );
			b.rightForeArm.quaternion.setFromAxisAngle( _ARM_HINGE_AXIS, this._elbowSign.right * elbowBend );
			b.rightArm.updateMatrixWorld( true );

			const achieved = new THREE.Vector3();
			b.rightHand.getWorldPosition( achieved );
			worstArmError = Math.max( worstArmError, achieved.distanceTo( solverCommitment( armPivot, target, achievedReach ) ) );

		}

		b.rightArm.quaternion.identity();
		b.rightForeArm.quaternion.identity();
		b.rightArm.updateMatrixWorld( true );

		worstError = Math.max( worstError, worstArmError );

		if ( worstError > 0.005 ) {

			console.error(
				`[blueprint-viewer] PatientHuman: IK self-check FAILED -- worst error ${worstError.toFixed( 4 )} m ` +
				'(bar: <=0.005 m, tightened from 0.01 m by the RIG-3 real-bind-vector solver fix, AGENTS.md #15; ' +
				'covers legs at bind pelvis, legs under a nonzero test pelvis rotation, and the right arm; every ' +
				'case is measured against the solver\'s own reach-clamped commitment, so ANY failure here is a real ' +
				'FK/retarget defect, never a benign clamp). Leg/arm placement may be visibly wrong.',
			);
			this.ikSelfCheckFailed = true;

		}

	}

	/**
	 * Re-pose the human for the given phase/time. Pure function of (phaseName,
	 * timeSec) and this module's own load-time measurements/schedules -- no state
	 * carried between calls (every quantity sync() needs is either a load-time
	 * constant or freshly recomputed from `poseAt(schedule, terrain, timeSec)`, which
	 * is itself stateless -- see PatientGait.js's own determinism contract). No-op
	 * until attachTo() (and buildGait()) have run.
	 *
	 * EVERY bone this module owns is written UNCONDITIONALLY on every call (no more
	 * canned-clip mixer providing a per-frame baseline for anything left untouched --
	 * see this file's own header) -- branches below (cane reachable/unreachable, cane
	 * enabled/disabled) always still write a value to every bone they cover, never
	 * skip a write.
	 */
	sync( phaseName, timeSec ) {

		if ( ! this._attached || ! this._schedules ) return;

		const schedule = this._schedules[ phaseName ];
		if ( ! schedule ) return;

		const root = this._patientRootNode;
		const b = this._bones;
		const scratch = this._scratch;
		const P = PATIENT_BODY_PARAMS;

		// --- 0) End-of-clip walk-on + freeze (UNCHANGED from the pre-existing
		// module -- see its own long-form comment in git history / AGENTS.md for the
		// full "why"; not reproduced here verbatim to keep this rewrite's own new
		// material easy to find, but the mechanism/formulas are untouched). ---
		const tail = schedule.tail;
		let rootPosX, rootPosY, rootPosZ, rootQuat, tq;
		if ( ! tail || timeSec <= tail.startT ) {

			rootPosX = root.position.x; rootPosY = root.position.y; rootPosZ = root.position.z;
			rootQuat = root.quaternion;
			tq = timeSec;

		} else if ( timeSec >= tail.freezeT ) {

			rootPosX = tail.rootEnd.x; rootPosY = tail.rootEnd.y; rootPosZ = tail.rootEnd.zRoot;
			rootQuat = scratch.qFreeze.setFromAxisAngle( _UP_Z, tail.rootEnd.yaw );
			tq = tail.freezeT;

		} else {

			const a = ( timeSec - tail.startT ) / ( tail.freezeT - tail.startT );
			rootPosX = tail.rootStart.x + ( tail.rootEnd.x - tail.rootStart.x ) * a;
			rootPosY = tail.rootStart.y + ( tail.rootEnd.y - tail.rootStart.y ) * a;
			rootPosZ = tail.rootStart.zRoot + ( tail.rootEnd.zRoot - tail.rootStart.zRoot ) * a;
			const yaw = tail.rootStart.yaw + ( tail.rootEnd.yaw - tail.rootStart.yaw ) * a;
			rootQuat = scratch.qFreeze.setFromAxisAngle( _UP_Z, yaw );
			tq = timeSec;

		}

		// F4 (integration_2.json diag, 2026-07-10): current ROOT yaw, resolved
		// through the SAME three branches as rootPosX/Y/rootQuat above -- reused by
		// _clampedAdvance below (step 17 fallback / step 18) so the arm-swing signal
		// tracks the ACTUAL rendered root through the walk-on tail, not poseAt's own
		// (tail-unaware) pose.rootYaw -- see _clampedAdvance's own updated doc.
		// patient_root is yaw-only about Z in every branch (root.quaternion per
		// PatientGait.js's header contract; scratch.qFreeze built that way directly
		// above), so 2*atan2(z,w) recovers it exactly without a branch-specific case.
		const currentYawRad = 2 * Math.atan2( rootQuat.z, rootQuat.w );

		const pose = poseAt( schedule, this._terrain, tq );

		// --- 1) v2 contract fields, with graceful degradation for an in-flight GAIT
		// integration race (spec I10 / this task's own instruction) ---
		const phaseC = ( typeof pose.phaseC === 'number' ) ? pose.phaseC : pose.gaitPhase;
		if ( typeof pose.phaseC !== 'number' && ! this._loggedNoPhaseC ) {

			console.log( '[blueprint-viewer] PatientHuman: pose.phaseC missing (PatientGait v1 schedule?) -- pelvis bob falls back to the legacy gaitPhase staircase (P2 stays unfixed) until GAIT\'s v2 schedule is active.' );
			this._loggedNoPhaseC = true;

		}
		const support = ( typeof pose.support === 'number' ) ? pose.support : 0;
		if ( typeof pose.support !== 'number' && ! this._loggedNoSupport ) {

			console.log( '[blueprint-viewer] PatientHuman: pose.support missing -- pelvis lateral shift/list/yaw, spine lateral lean, and shoulder load-lean are all disabled (0) until GAIT\'s v2 schedule provides it.' );
			this._loggedNoSupport = true;

		}
		const caneAvailable = !! pose.cane;
		if ( ! caneAvailable && ! this._loggedNoCane ) {

			console.log( '[blueprint-viewer] PatientHuman: pose.cane missing (PatientGait v1 schedule, or caneEnabled===false) -- hiding the cane prop and running BOTH arms as free FK swing until GAIT\'s v2 cane schedule is active.' );
			this._loggedNoCane = true;

		}
		this._cane.visible = caneAvailable;

		// --- 2) Cane pose (early -- independent of legs/pelvis/spine; the spine's
		// own lateral-lean needs cane.planted below) ---
		let canePoseResult = null;
		if ( caneAvailable ) {

			// R3a (round-2 diag): resolve the root's own yaw AT the cane's most recent
			// landing (same "extra poseAt() call at a per-event reference time" pattern
			// this file's toeScale/heelScale resolution already uses, step 6 below --
			// event arrays are "a few dozen per clip", cheap) so computeCanePose can
			// cone-clamp the shaft's yaw-tracking instead of following the live root
			// yaw unbounded through a fast turn. `null` (before the first-ever cane
			// landing, or a v1 schedule with no `landedAt`) makes computeCanePose fall
			// back to its pre-R3a unclamped behavior -- see that function's own doc.
			const caneYawRefT = pose.cane.landedAt;
			const caneYawRefRad = ( typeof caneYawRefT === 'number' ) ? poseAt( schedule, this._terrain, caneYawRefT ).rootYaw : null;
			canePoseResult = computeCanePose( pose.cane, rootQuat, CANE_PARAMS, this._canePose, caneYawRefRad, support, pose.groundSlope );

		}

		// --- 3) Anchor placement ---
		//
		// position: (root.x, root.y, root.z - _hipsHeightM). NO bob term here anymore
		// (P2 fix, spec S6 item 8): bob moves to Hips.position.y below, since
		// `bobAmplitude*sin(4*pi*gaitPhase)` here was ALWAYS EXACTLY ZERO (gaitPhase
		// only ever takes values k/2, so sin(2*pi*k)=0 -- a dead feature, not a
		// working one this rewrite is "moving"; the NEW bob is phaseC-driven and
		// actually oscillates).
		//
		// R1 (round-2 diag): `+ P.standingReachRaiseM` is the stance-knee-crouch fix --
		// see that param's own PATIENT_BODY_PARAMS comment for the numeric root cause
		// (a fixed ~2.7-3cm reach shortfall between PATIENT_HIP_HEIGHT_M and this rig's
		// own straight-leg-plus-ankle height). Anchor-local, so it raises the WHOLE
		// character (Hips/spine/arms/cane) uniformly -- ankle/toe targets are computed
		// independently from `pose.*Foot` (terrain-relative, never anchor-relative), so
		// I2 (planted feet never move) is untouched; this only shortens how far the leg
		// IK has to reach DOWN to them, straightening the stance knee toward the natural
		// band. Step 7 below still clamps reach on TOP of this (e.g. on stairs), so an
		// already-near-max-reach case simply has its clamp engage a little sooner --
		// self-limiting, not a second constant to keep in sync.
		scratch.v0.set( rootPosX, rootPosY, rootPosZ - this._hipsHeightM + P.standingReachRaiseM );

		this.anchor.quaternion.copy( rootQuat ).multiply( B_PLACEMENT );

		// --- 4) Pelvis dynamics (Hips local pos/quat, spec S6 items 1/8) ---
		//
		// bob: z (P-FRAME "up") in the spec's own shorthand -- Hips lives in ANCHOR-
		// LOCAL/Xbot-local space where UP IS LOCAL Y, not Z (see this file's header
		// axis-convention block), so the SAME physical "bob up/down" effect is written
		// to hipsPos.y here, never .z (citing CLAUDE.md incident 8.7's own discipline:
		// verify which axis a coordinate SPACE uses before pattern-matching a letter
		// in a comment onto a variable).
		// R4 (round-2 diag, fullbody_naturalness.md pelvisBob_climb_top_landing): raised
		// to the `bobSpeedRefExponent` power (default 0.5 = sqrt) instead of a bare
		// linear ratio -- see that param's own PATIENT_BODY_PARAMS comment. `pose.speed`
		// is EXACTLY 0 only during true idle (the gait's own idle gate), so
		// `speedScaleBob` still reaches EXACTLY 0 there (Math.pow(0, 0.5)===0) --
		// CONTINUOUSLY, through the same walk->idle deceleration this file's own I3
		// invariant already relies on elsewhere, not a new discrete floor/snap.
		const speedRatioBob = THREE.MathUtils.clamp( Math.max( 0, pose.speed ) / P.bobSpeedRefMps, 0, 1 );
		const speedScaleBob = Math.pow( speedRatioBob, P.bobSpeedRefExponent );
		const bobM = P.bobAmplitudeM * ( - Math.cos( 2 * Math.PI * 2 * phaseC ) ) * 0.5 * speedScaleBob;

		const shiftM = P.hipShiftM * support; // lateral (Xbot-local X), FACING-frame == anchor-local directly since the anchor itself already tracks rootQuat

		const speedScalePelvis = Math.min( 1, Math.max( 0, pose.speed ) / P.pelvisDynamicsSpeedRefMps );
		// "swingDirection" (spec S6 item 1) is reused as `support` itself here: it is
		// the only continuous, sign-correct, non-staircase per-step direction signal
		// PatientGait v2's contract actually defines (S3) -- a hard sign()/step-
		// function alternative would reintroduce exactly the "moving while stopped /
		// staircase jump" class of bug phaseC/support exist to eliminate. Sign is a
		// judgment call (unverified visually, no browser tier in this pass) -- see
		// this rewrite's own report.
		const listRad = P.pelvisListRad * support * speedScalePelvis;
		const yawRad = P.pelvisYawRad * support * speedScalePelvis;

		// scratch.hipsPos is TRUE METERS throughout this function (matches
		// _hipPivotLocal/_bindOffsets' own unit system) -- built from
		// _bindHipsPositionWorldM, NOT the RAW-local _bindHipsPosition (that field
		// exists ONLY for step 10's bone write, see its own constructor comment).
		scratch.hipsPos.copy( this._bindHipsPositionWorldM );
		scratch.hipsPos.x += shiftM;
		scratch.hipsPos.y += bobM;

		const hipsQuat = scratch.q0.setFromAxisAngle( _FORWARD_AXIS_LOCAL, listRad )
			.multiply( scratch.q1.setFromAxisAngle( _UP_AXIS_LOCAL, yawRad ) ); // list ∘ yaw: yaw applied first (inner), list outer

		// --- 5) Recompute hip pivots analytically under the new Hips transform
		// (spec S6 item 1: "pivot' = hipsPos + Q_pelvis*(pivotBind-hipsBindPos)") ---
		const leftPivot = scratch.leftPivot.copy( this._hipPivotLocal.left ).sub( this._bindHipsPositionWorldM )
			.applyQuaternion( hipsQuat ).add( scratch.hipsPos );
		const rightPivot = scratch.rightPivot.copy( this._hipPivotLocal.right ).sub( this._bindHipsPositionWorldM )
			.applyQuaternion( hipsQuat ).add( scratch.hipsPos );

		// --- 6) Ankle targets (P-frame), WITH foot-roll pivot (spec S6b) ---
		const leftHasRoll = typeof pose.leftFoot.landedAt !== 'undefined' && typeof pose.leftFoot.nextLiftAt !== 'undefined';
		const rightHasRoll = typeof pose.rightFoot.landedAt !== 'undefined' && typeof pose.rightFoot.nextLiftAt !== 'undefined';
		if ( ( ! leftHasRoll || ! rightHasRoll ) && ! this._loggedNoFootRollFields ) {

			console.log( '[blueprint-viewer] PatientHuman: poseAt().leftFoot/rightFoot missing landedAt/nextLiftAt/strideLen (PatientGait v1 schedule?) -- foot roll (heel-strike/toe-off) stays flat (0) until GAIT\'s v2 schedule provides them.' );
			this._loggedNoFootRollFields = true;

		}
		// F3 (integration_2.json diag, 2026-07-10), second mechanism found via
		// _debugWorstPitch after the swing-profile fix (_smoothstep3Prop) moved the
		// M9b worst-case elsewhere: toeScale (inside _footRollPitch) was read from
		// pose.speed at the INSTANTANEOUS query time `tq`, which stays CONSTANT
		// across a boundary (the doc's own continuity argument) but is NOT constant
		// WITHIN a heel-off/toe-off window or a swing -- during a stop-and-go speed
		// transient (e.g. climb clip ~t=39.5s) `speed` itself swings quickly enough
		// within one dt=0.05s diag sample to blow the M9b bar on its own, with no
		// event-boundary or swing-profile-shape involvement at all (measured 0.172
		// rad/sample). heelScale ALREADY avoids this (built from footPose.strideLen,
		// a per-EVENT constant, never re-sampled mid-window) -- toeScale gets the
		// same treatment here: resolved ONCE per foot per sync(), at the window's own
		// fixed reference instant (nextLiftAt while planted heading into toe-off,
		// liftAt while swinging -- the SAME instant on both sides of that boundary,
		// preserving the doc's own cross-boundary continuity proof unchanged) rather
		// than re-read at the live query time, so it stays CONSTANT for the whole
		// window exactly like heelScale does. Extra poseAt() calls are cheap (event
		// arrays are "a few dozen per clip", this file's own header) and pure/
		// side-effect-free -- safe to call again here.
		const leftToeScaleRefT = pose.leftFoot.planted ? pose.leftFoot.nextLiftAt : pose.leftFoot.liftAt;
		const rightToeScaleRefT = pose.rightFoot.planted ? pose.rightFoot.nextLiftAt : pose.rightFoot.liftAt;
		const leftToeScaleSpeed = ( typeof leftToeScaleRefT === 'number' ) ? poseAt( schedule, this._terrain, leftToeScaleRefT ).speed : pose.speed;
		const rightToeScaleSpeed = ( typeof rightToeScaleRefT === 'number' ) ? poseAt( schedule, this._terrain, rightToeScaleRefT ).speed : pose.speed;

		const leftRollPitch = leftHasRoll ? _footRollPitch( pose.leftFoot, tq, leftToeScaleSpeed, P ) : 0;
		const rightRollPitch = rightHasRoll ? _footRollPitch( pose.rightFoot, tq, rightToeScaleSpeed, P ) : 0;

		// F2a (integration_2.json diag, 2026-07-10): leftContact/rightContact mirror
		// the SAME branch selection below (heel/toe pivot vs flat vs swing) as a
		// plain {x,y,z,mode} point -- see _lastSync's own step-20 comment for why
		// (plantedDrift/M8_rollWindowContactDriftMax were anchored on the TOE BONE,
		// which legitimately swings a few cm during a roll even though this contact
		// point, by I2/_pivotAnkleTarget's own construction, never moves).
		let leftTargetWorld, rightTargetWorld;
		let leftContact, rightContact;
		if ( pose.leftFoot.planted && leftRollPitch !== 0 ) {

			// Heel pivot while dorsiflexed (heel-strike, rollPitch>0); toe pivot while
			// plantarflexed (toe-off, rollPitch<0) -- see _footRollPitch's own PITCH_AXIS
			// sign convention. The (essentially theoretical, given this app's stance
			// durations vs rollDownSec/heelOffSec) case where BOTH windows would be
			// simultaneously nonzero is resolved by this same if/else priority (heel
			// wins) -- see _pivotAnkleTarget's own call-site comment below.
			//
			// F11-SIGN (M8 diag, 2026-07-10): the pivot-side sign was BACKWARDS vs
			// IK_OVERHAUL_SPEC.md S6b. The spec pivots heel-strike about the HEEL
			// contact `heelPoint = plantPos - facing*heelBackM` (BEHIND the plant) and
			// toe-off about the TOE contact `plantPos + facing*toeForwardLen` (AHEAD).
			// _footContactPoint returns `plantPos - facing*offset`, so a BEHIND heel
			// needs offset=+heelBackM and an AHEAD toe needs offset=-toeForwardLen --
			// but this passed -heelBackM / +toeForwardLen, placing the heel pivot AHEAD
			// (toe side) and the toe pivot BEHIND (heel side). Consequence: a
			// "dorsiflex" (heel down, toe UP) rotated the foot about the wrong (ahead)
			// point and drove the rendered TOE bone DOWN ~2.3 cm into the tread/floor at
			// every landing (M8 planted soleClearance -0.0233 m on BOTH clips; heelStrike
			// Rad=0 zeroed it, confirming the roll was the cause). The flat reduction
			// (pitch=0 -> plantPos + ankleHeight*up) is sign-independent, which is why
			// this survived every prior review: it only shows at nonzero roll. Flipping
			// both signs dropped M8 to -0.0066 m (89% less than the original -0.059 m
			// swing bug) with NO M9b/fkError/plantedDrift regression -- see the swing
			// branch below (its blend endpoints flip in lockstep to stay continuous).
			const forwardOffset = leftRollPitch > 0 ? P.heelBackM : - this._toeForwardLenM;
			leftTargetWorld = _pivotAnkleTarget(
				new THREE.Vector3( pose.leftFoot.x, pose.leftFoot.y, pose.leftFoot.z ), pose.leftFoot.yaw,
				forwardOffset, this._ankleHeightM, leftRollPitch, new THREE.Vector3(),
			);
			leftContact = { ..._footContactPoint( pose.leftFoot, pose.leftFoot.yaw, forwardOffset ), mode: leftRollPitch > 0 ? 'heel' : 'toe' };

		} else if ( ! pose.leftFoot.planted ) {

			// R1 fix (round-2 diag, fullbody_naturalness.md swingKnee*/knee_*): this
			// branch used to IGNORE leftRollPitch entirely and always target the flat
			// `ankleHeightM` offset -- but leftRollPitch is itself CONTINUOUS across
			// both the liftoff boundary (a swing starts at -toeOffRad*toeScale,
			// matching the just-ended toe-off window's own value) and the landing
			// boundary (a swing ends at heelStrikeRad*heelScale, matching the
			// about-to-start heel-strike window's own value -- see _footRollPitch's
			// own doc), yet the ANKLE TARGET POSITION was NOT: it snapped from the
			// stance toe-pivot formula's ending value to this flat one in ONE FRAME at
			// every liftoff (measured directly against the naturalness report's own
			// worst-timestamps, e.g. t=2.400->2.417s: ankleTargetWorld.z 0.0517->
			// 0.0983m, knee bend 25.8->46.4deg in a single 1/60s sample -- the
			// dominant cause of R1's >400deg/s knee-bend velocity spikes). Fix: reuse
			// the SAME `_pivotAnkleTarget` the stance windows use, with
			// `forwardOffset` blended (plain smoothstep of swingU, zero slope at both
			// ends -- matches _footRollPitch's own C1-at-the-boundary discipline) from
			// `-_toeForwardLenM` (liftoff, u=0) to `+heelBackM` (landing, u=1) -- the
				// F11-SIGN-corrected endpoints, flipped in lockstep with the stance
				// windows (OLD spec-mismatched values were +toeFwd@u0 / -heelBack@u1) -- the
			// STANCE branch's own two endpoint values, so the target is bit-identical
			// to the stance formula's ending value at u=0 and to its upcoming starting
			// value at u=1 (both `forwardOffset` AND `leftRollPitch` match exactly at
			// each boundary), eliminating the pop at both ends without touching the
			// (already "good"/passing) stance-side roll windows or the swing arc's own
			// x/y/z (GAIT-owned). At leftRollPitch===0 (no roll fields / v1 schedule)
			// this reduces EXACTLY to the pre-existing flat formula regardless of
			// forwardOffset (see _pivotAnkleTarget's own doc: "at pitch=0 this exactly
			// reproduces plantPos + ankleHeightM*up").
			const leftSwingU = ( typeof pose.leftFoot.swingU === 'number' ) ? THREE.MathUtils.clamp( pose.leftFoot.swingU, 0, 1 ) : 0;
			const leftBlend = _smoothstep( leftSwingU );
			const forwardOffset = - this._toeForwardLenM + ( P.heelBackM + this._toeForwardLenM ) * leftBlend; // F11-SIGN endpoints (-toeFwd@u0 -> +heelBack@u1); see stance branch above
			leftTargetWorld = _pivotAnkleTarget(
				new THREE.Vector3( pose.leftFoot.x, pose.leftFoot.y, pose.leftFoot.z ), pose.leftFoot.yaw,
				forwardOffset, this._ankleHeightM, leftRollPitch, new THREE.Vector3(),
			);
			leftContact = { x: pose.leftFoot.x, y: pose.leftFoot.y, z: pose.leftFoot.z, mode: 'swing' };

		} else {

			leftTargetWorld = new THREE.Vector3( pose.leftFoot.x, pose.leftFoot.y, pose.leftFoot.z + this._ankleHeightM );
			leftContact = { x: pose.leftFoot.x, y: pose.leftFoot.y, z: pose.leftFoot.z, mode: 'flat' };

		}
		if ( pose.rightFoot.planted && rightRollPitch !== 0 ) {

			const forwardOffset = rightRollPitch > 0 ? P.heelBackM : - this._toeForwardLenM; // F11-SIGN: see left-foot branch
			rightTargetWorld = _pivotAnkleTarget(
				new THREE.Vector3( pose.rightFoot.x, pose.rightFoot.y, pose.rightFoot.z ), pose.rightFoot.yaw,
				forwardOffset, this._ankleHeightM, rightRollPitch, new THREE.Vector3(),
			);
			rightContact = { ..._footContactPoint( pose.rightFoot, pose.rightFoot.yaw, forwardOffset ), mode: rightRollPitch > 0 ? 'heel' : 'toe' };

		} else if ( ! pose.rightFoot.planted ) {

			// R1 fix -- mirror of the left-foot swing branch above, see its own comment.
			const rightSwingU = ( typeof pose.rightFoot.swingU === 'number' ) ? THREE.MathUtils.clamp( pose.rightFoot.swingU, 0, 1 ) : 0;
			const rightBlend = _smoothstep( rightSwingU );
			const forwardOffset = - this._toeForwardLenM + ( P.heelBackM + this._toeForwardLenM ) * rightBlend; // F11-SIGN endpoints; see left/stance branch
			rightTargetWorld = _pivotAnkleTarget(
				new THREE.Vector3( pose.rightFoot.x, pose.rightFoot.y, pose.rightFoot.z ), pose.rightFoot.yaw,
				forwardOffset, this._ankleHeightM, rightRollPitch, new THREE.Vector3(),
			);
			rightContact = { x: pose.rightFoot.x, y: pose.rightFoot.y, z: pose.rightFoot.z, mode: 'swing' };

		} else {

			rightTargetWorld = new THREE.Vector3( pose.rightFoot.x, pose.rightFoot.y, pose.rightFoot.z + this._ankleHeightM );
			rightContact = { x: pose.rightFoot.x, y: pose.rightFoot.y, z: pose.rightFoot.z, mode: 'flat' };

		}

		// --- 7) Pelvis reachability (computed BEFORE finalizing anchor.position --
		// unchanged mechanism from the pre-existing module, now measured against the
		// SHIFTED pivots (step 5) instead of the raw bind pivots, so the estimate and
		// the actual solve (step 10) agree). ---
		const rootQuatInv = scratch.rootQuatInv.copy( rootQuat ).invert();

		const toAnchorLocal = ( pWorld, anchorPos, out ) => {

			out.copy( pWorld ).sub( anchorPos );
			out.applyQuaternion( rootQuatInv );
			out.applyQuaternion( B_PLACEMENT_INV );
			return out;

		};

		const nominalAnchorPos = scratch.v0.clone();
		const leftLocalNominal = toAnchorLocal( leftTargetWorld, nominalAnchorPos, new THREE.Vector3() );
		const rightLocalNominal = toAnchorLocal( rightTargetWorld, nominalAnchorPos, new THREE.Vector3() );

		const leftReachNominal = leftPivot.distanceTo( leftLocalNominal );
		const rightReachNominal = rightPivot.distanceTo( rightLocalNominal );
		const reachLimit = 0.98 * this._legChain.bindExtensionM; // RIG-3 fix (AGENTS.md #15): was 0.98*(L1+L2) -- bindExtensionM is the chain's real |v1+v2| bind-pose reach (see _buildChainGeometry), the direct replacement now that v1/v2 aren't assumed collinear
		const worstExcess = Math.max( 0, leftReachNominal - reachLimit, rightReachNominal - reachLimit );

		scratch.v0.z -= worstExcess;

		this.anchor.position.copy( scratch.v0 );

		// --- 8) Ankle IK targets, final anchor-local conversion (post-lowering) ---
		const leftTargetLocal = toAnchorLocal( leftTargetWorld, this.anchor.position, new THREE.Vector3() );
		const rightTargetLocal = toAnchorLocal( rightTargetWorld, this.anchor.position, new THREE.Vector3() );

		// --- 9) Two-bone leg IK, per leg, in anchor-local (Xbot) space (pole vector:
		// that leg's OWN yaw reference, unchanged from the pre-existing module) ---
		const poleFromYaw = ( footYaw ) => {

			const fwdWorld = new THREE.Vector3( Math.cos( footYaw ), Math.sin( footYaw ), 0 );
			fwdWorld.applyQuaternion( rootQuatInv ).applyQuaternion( B_PLACEMENT_INV );
			return fwdWorld;

		};

		const leftPole = poleFromYaw( pose.leftFoot.yaw );
		const rightPole = poleFromYaw( pose.rightFoot.yaw );

		const leftIK = _solveTwoBoneIK( leftPivot, leftTargetLocal, leftPole, this._maxReachM, this._minReachM, this._legChain );
		const rightIK = _solveTwoBoneIK( rightPivot, rightTargetLocal, rightPole, this._maxReachM, this._minReachM, this._legChain );

		// --- 10) Write Hips (NEW: nonzero pos/quat, no longer bind pose every frame
		// -- LeftUpLeg/RightUpLeg compensate below, step 11) ---
		//
		// Hips.position is interpreted in RAW (Armature-0.01-scaled) units by
		// three.js, NOT true meters (see _bindHipsPosition's own constructor
		// comment) -- scratch.hipsPos is true meters throughout this function, so
		// only the DELTA off bind (shiftM/bobM, already true-meters scalars) needs
		// converting (divide by armatureScale) before writing; the raw base itself
		// (_bindHipsPosition) is the exact originally-loaded value, never
		// reconstructed via a meters round-trip.
		b.hips.position.copy( this._bindHipsPosition );
		b.hips.position.x += shiftM / this._armatureScale;
		b.hips.position.y += bobM / this._armatureScale;
		b.hips.quaternion.copy( hipsQuat );

		// --- 11) UpLeg locals = Q_pelvis^-1 * (anchor-desired hip quat) -- children
		// of Hips, so this is what makes the LEG IK land exactly on target despite
		// Hips' own new rotation (spec S6 item 1). ---
		const hipsQuatInv = scratch.q2.copy( hipsQuat ).invert();
		b.leftUpLeg.quaternion.copy( hipsQuatInv ).multiply( leftIK.hipQuat );
		b.rightUpLeg.quaternion.copy( hipsQuatInv ).multiply( rightIK.hipQuat );

		// --- 12) Leg (knee): unchanged, no compensation needed for a LOCAL child
		// rotation (see _solveTwoBoneIK's own doc). ---
		b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - leftIK.kneeBend );
		b.rightLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - rightIK.kneeBend );

		// --- 13) Foot orientation (flat-at-yaw + roll pitch + swing modulation) ---
		this._orientFoot( b.leftFoot, leftIK.hipQuat, - leftIK.kneeBend, pose.leftFoot, leftRollPitch, scratch );
		this._orientFoot( b.rightFoot, rightIK.hipQuat, - rightIK.kneeBend, pose.rightFoot, rightRollPitch, scratch );

		// --- 14) ToeBase articulation (spec S6 item 4 / S6b) ---
		const leftToePitch = leftHasRoll ? _toeBasePitch( pose.leftFoot, tq, pose.speed, P ) : 0;
		const rightToePitch = rightHasRoll ? _toeBasePitch( pose.rightFoot, tq, pose.speed, P ) : 0;
		b.leftToeBase.quaternion.setFromAxisAngle( PITCH_AXIS, leftToePitch );
		b.rightToeBase.quaternion.setFromAxisAngle( PITCH_AXIS, rightToePitch );

		// --- 15) Spine chain (spec S6 item 5) ---
		const gp = this._gaitParams;
		const torsoPitch = THREE.MathUtils.clamp(
			gp.leanBase + gp.leanSpeedK * pose.speed + gp.leanSlopeK * pose.groundSlope,
			0, 0.15,
		); // UNCHANGED lean model/gains, per spec "keep gains"

		const spineYawCounter = P.spineYawCounterK * yawRad;
		const caneBearing = caneAvailable && pose.cane.planted;
		const caneLoadLean = caneBearing ? P.caneLoadLeanRad : 0;
		const spineLateralLean = P.spineLateralLeanK * support * listRad + caneLoadLean;
		const breathT = timeSec; // RAW time (not tq): breathing is ALWAYS on (spec I3/S6 item 5), independent of gait/tail freezing
		const breathing = P.breathPitchRad * Math.sin( 2 * Math.PI * P.breathHz * breathT );

		b.spine.quaternion.setFromAxisAngle( PITCH_AXIS, torsoPitch ); // pitch stays entirely on Spine (unchanged mechanism)

		const spine1Quat = scratch.q1.setFromAxisAngle( _FORWARD_AXIS_LOCAL, spineLateralLean / 2 )
			.multiply( scratch.q2.setFromAxisAngle( _UP_AXIS_LOCAL, spineYawCounter / 2 ) )
			.multiply( scratch.q3.setFromAxisAngle( PITCH_AXIS, breathing / 2 ) );
		b.spine1.quaternion.copy( spine1Quat );

		const spine2Quat = scratch.q1.setFromAxisAngle( _FORWARD_AXIS_LOCAL, spineLateralLean / 2 )
			.multiply( scratch.q2.setFromAxisAngle( _UP_AXIS_LOCAL, spineYawCounter / 2 ) )
			.multiply( scratch.q3.setFromAxisAngle( PITCH_AXIS, breathing / 2 ) );
		b.spine2.quaternion.copy( spine2Quat );

		// --- 16) Analytic FK walk to the RIGHT shoulder pivot (Hips->Spine->Spine1->
		// Spine2->RightShoulder->RightArm), using the quats JUST written above (spec
		// S6 item 1's compensation discipline, extended one level for the arm chain
		// -- see this file's header/RightShoulder's own bind-offset comment). ---
		const rightShoulderLoadRad = P.shoulderModulationRad * support; // support-driven, spec S6 item 6
		const rightShoulderQuat = scratch.q0.setFromAxisAngle( _FORWARD_AXIS_LOCAL, rightShoulderLoadRad );
		b.rightShoulder.quaternion.copy( rightShoulderQuat );

		const chainPos = scratch.chainPos.copy( scratch.hipsPos );
		// Read the pelvis quaternion back from b.hips.quaternion (set at step 10),
		// NOT the local `hipsQuat` variable -- `hipsQuat` is an ALIAS of scratch.q0,
		// which `rightShoulderQuat` two lines above has already overwritten. This is
		// deliberately a bone read, not a matrixWorld read (I5): b.hips.quaternion is
		// a plain Object3D property holding a value THIS function itself just wrote,
		// not a derived/computed matrix.
		const chainQuat = scratch.chainQuat.copy( b.hips.quaternion );
		for ( const link of [
			{ offset: this._bindOffsets.spine, quat: b.spine.quaternion },
			{ offset: this._bindOffsets.spine1, quat: b.spine1.quaternion },
			{ offset: this._bindOffsets.spine2, quat: b.spine2.quaternion },
			{ offset: this._bindOffsets.rightShoulder, quat: rightShoulderQuat },
		] ) {

			chainPos.add( link.offset.clone().applyQuaternion( chainQuat ) );
			chainQuat.multiply( link.quat );

		}
		const rightArmPivot = scratch.armPivot.copy( chainPos ).add( this._bindOffsets.rightArm.clone().applyQuaternion( chainQuat ) );
		const rightShoulderFrameQuat = chainQuat; // cumulative anchor-local orientation Arm's LOCAL rotation is relative to

		// R2 (round-2 diag, fullbody_naturalness.md armSwing_left_*): stride-adaptive
		// arm-swing normalization reference, shared by both _clampedAdvance call sites
		// below (step 17's cane-less fallback and step 18's real left-arm drive) -- see
		// armSwingStrideFracK's own PATIENT_BODY_PARAMS comment for the numeric
		// derivation. Averaged across BOTH feet (not just the "other" foot each call
		// drives off of) so a single stride's own event-boundary jump in ONE foot's
		// strideLen is damped by the other foot's already-settled value, rather than
		// this reference itself popping in lockstep with the driven foot's own
		// per-cycle liftoff (which lands exactly at that foot's own rawAdv extremum --
		// same hazard class as R1's ankle-target discontinuity, just pre-empted here by
		// averaging instead of a boundary-matched blend).
		const armStrideRefM = ( ( pose.leftFoot.strideLen || 0 ) + ( pose.rightFoot.strideLen || 0 ) ) * 0.5;
		const armReachRefM = Math.max( P.armSwingStrideFracK * armStrideRefM, P.armSwingReachFloorM );

		// --- 17) Right arm: two-bone IK to the cane handle (spec S5/S6 item 6), or a
		// mirrored FK-swing fallback when no cane schedule is available (this
		// rewrite's own graceful degradation, see step 1's log). ---
		// F6 (integration_2.json diag, 2026-07-10): caneHandleEffective is the handle
		// point actually handed to applyCanePose this frame (pre-clamp
		// canePoseResult.handle normally, the reach-clamped adjustedHandle below when
		// armIK.reachClamped) -- snapshotted as a plain {x,y,z} (mirrors
		// caneHandleTargetWorld's own snapshot a few hundred lines below, since
		// canePoseResult/scratch.caneHandleAdj are both mutated in place next frame)
		// so _lastSync can expose the ACHIEVED target separately from the nominal
		// pre-clamp one (main.js's M12a previously compared the hand only against the
		// nominal target, which is wrong whenever the arm was clamped).
		let caneHandleEffective = null, caneReachClamped = false;
		if ( caneAvailable ) {

			const armTargetWorld = canePoseResult.handle;
			// R1/R3 interaction (round-2 diag): `P.standingReachRaiseM` (step 3) raises
			// the WHOLE anchor to fix the LEG's stance crouch; the cane's own handle
			// target is a WORLD/terrain-referenced point (computed independently of
			// anchor height, see computeCanePose), so the raise ALSO tightens the CANE
			// ARM's own reach margin (isolated trace A/B: R1 alone, GAIT held constant,
			// regressed cane__follow_straight mean err 8.21->15.31mm). Two direct
			// per-frame "de-raise just the arm's reference" attempts (subtracting, then
			// adding, standingReachRaiseM to a copy of the anchor used only for this
			// conversion) were tried and BOTH measured WORSE than doing nothing (worse
			// than simply using the raised anchor here) -- the interaction isn't the
			// simple additive one it looks like on paper (this.anchor.position feeds
			// `rightArmPivot`'s own FK-chain-relative frame too, and step 7's
			// reachability clamp non-linearly couples leg and anchor placement), so
			// reverted to the plain (raised) anchor rather than ship an unverified,
			// worse-than-baseline "fix". See standingReachRaiseM's own tuned-down
			// default (reduced from the naive law-of-cosines value specifically to
			// keep this residual coupling small) and this rewrite's own report for the
			// remaining cane-arm gap this leaves in follow_turning.
			const armTargetLocal = scratch.armTargetLocal.copy( armTargetWorld ).sub( this.anchor.position )
				.applyQuaternion( rootQuatInv ).applyQuaternion( B_PLACEMENT_INV );

			const armPole = new THREE.Vector3( 0, 0, - 1 ); // Xbot-local "backward" -- see _ARM_HINGE_AXIS's neighbourhood / this rewrite's report for the (unverified-visually) reasoning
			const armIK = _solveTwoBoneIK(
				rightArmPivot, armTargetLocal, armPole, this._armMaxReachM, this._armMinReachM, this._armChain.right,
			);

			b.rightArm.quaternion.copy( rightShoulderFrameQuat ).invert().multiply( armIK.hipQuat );
			b.rightForeArm.quaternion.setFromAxisAngle( _ARM_HINGE_AXIS, this._elbowSign.right * armIK.kneeBend );

			// Wrist: aim the hand's own fingers-forward bind axis at the shaft
			// direction, twist so the palm-normal bind axis points toward a fixed
			// "inward/down" reference (spec S5's "document the axis mapping you
			// measure ... don't guess signs" -- the AIM half is measured; the TWIST
			// reference itself is a reasoned-but-visually-unverified choice, see this
			// rewrite's report).
			const shaftDirLocal = scratch.v1.copy( canePoseResult.axisWorld ).multiplyScalar( -1 )
				.applyQuaternion( rootQuatInv ).applyQuaternion( B_PLACEMENT_INV ); // fingers point DOWN the shaft, toward the tip -- opposite the tip->handle axis
			const wristRef = new THREE.Vector3( 1, - 0.3, 0 ).normalize();
			const handDesiredAbs = _aimWithTwist( this._handFingersForward.right, shaftDirLocal, this._handPalmNormal.right, wristRef, scratch.handDesiredAbs );
			const foreArmCum = scratch.foreArmCumQuat.copy( rightShoulderFrameQuat ).multiply( armIK.hipQuat ).multiply( b.rightForeArm.quaternion );
			b.rightHand.quaternion.copy( foreArmCum ).invert().multiply( handDesiredAbs );

			if ( armIK.reachClamped ) {

				if ( ! this._loggedCaneUnreachable ) {

					console.log(
						'[blueprint-viewer] PatientHuman: right-hand cane grip target exceeds arm reach ' +
						`(max ${this._armMaxReachM.toFixed( 3 )} m) -- clamping the arm and tilting the cane ` +
						'toward the achieved hand position (spec S5) instead of hyper-extending.',
					);
					this._loggedCaneUnreachable = true;

				}

				const achievedLocal = scratch.armAchievedLocal.copy( armTargetLocal ).sub( rightArmPivot )
					.normalize().multiplyScalar( armIK.achievedReach ).add( rightArmPivot );
				const achievedWorld = scratch.armAchievedWorld.copy( achievedLocal )
					.applyQuaternion( B_PLACEMENT ).applyQuaternion( rootQuat ).add( this.anchor.position );
				const adjustedAxis = scratch.caneAxis.copy( achievedWorld ).sub( canePoseResult.tip ).normalize();
				const adjustedHandle = scratch.caneHandleAdj.copy( canePoseResult.tip ).addScaledVector( adjustedAxis, CANE_PARAMS.caneLengthM );
				applyCanePose( this._cane, canePoseResult.tip, adjustedHandle );
				caneHandleEffective = { x: adjustedHandle.x, y: adjustedHandle.y, z: adjustedHandle.z };
				caneReachClamped = true;

			} else {

				applyCanePose( this._cane, canePoseResult.tip, canePoseResult.handle );
				caneHandleEffective = { x: canePoseResult.handle.x, y: canePoseResult.handle.y, z: canePoseResult.handle.z };

			}

		} else {

			// No cane schedule: mirror the left arm's own FK swing (step 18) onto the
			// right side too rather than leaving it frozen at bind pose or reaching
			// for a nonexistent target (see step 1's I10 log).
			const advRFallback = _clampedAdvance( rootPosX, rootPosY, currentYawRad, pose.leftFoot, armReachRefM ); // contralateral of the RIGHT arm is the LEFT foot
			const qSwing = scratch.q0.setFromAxisAngle( PITCH_AXIS, - P.armSwingRad * advRFallback );
			const qAbduct = scratch.q1.setFromAxisAngle( _FORWARD_AXIS_LOCAL, - P.armAbductRad );
			const hangQuatRight = _quatFromTo( this._armRestDir.right, _REST_DIR, scratch.q2 );
			b.rightArm.quaternion.copy( qSwing.multiply( qAbduct ) ).multiply( hangQuatRight );
			const elbowRadFallback = P.elbowBaseRad + P.elbowSwingRad * Math.max( 0, advRFallback ) * 0.3;
			b.rightForeArm.quaternion.setFromAxisAngle( _ARM_HINGE_AXIS, this._elbowSign.right * elbowRadFallback );
			b.rightHand.quaternion.identity();

		}

		// --- 18) Left arm: FK swing, driven by the CONTRALATERAL (right foot)
		// leg-advance signal (spec S6 item 6). ---
		const advR = _clampedAdvance( rootPosX, rootPosY, currentYawRad, pose.rightFoot, armReachRefM );
		const leftSwingRad = P.armSwingRad * advR;
		const qSwingL = scratch.q0.setFromAxisAngle( PITCH_AXIS, leftSwingRad );
		const qAbductL = scratch.q1.setFromAxisAngle( _FORWARD_AXIS_LOCAL, P.armAbductRad );
		const hangQuatLeft = _quatFromTo( this._armRestDir.left, _REST_DIR, scratch.q2 );
		b.leftArm.quaternion.copy( qSwingL.multiply( qAbductL ) ).multiply( hangQuatLeft );

		const elbowRadLeft = P.elbowBaseRad + P.elbowSwingRad * Math.max( 0, advR ) * 0.3;
		b.leftForeArm.quaternion.setFromAxisAngle( _ARM_HINGE_AXIS, this._elbowSign.left * elbowRadLeft );
		b.leftHand.quaternion.identity(); // "relaxed" (spec S6 item 6)

		// LeftShoulder: this rewrite's own filled gap (see PATIENT_BODY_PARAMS.
		// shoulderModulationRad's own comment) -- a small adv_R-driven counterpart to
		// RightShoulder's cane-load modulation (step 16), for left/right symmetry.
		b.leftShoulder.quaternion.setFromAxisAngle( _FORWARD_AXIS_LOCAL, P.shoulderModulationRad * advR * 0.5 );

		// --- 19) Head/Neck (spec S6 item 7): counter-pitch/yaw + constant gaze-down,
		// split evenly across Neck/Head (spec enumerates "Head/Neck" as one item
		// without an explicit per-bone split, unlike item 5's explicit Spine1/Spine2
		// split -- even split chosen for a natural progressive counter-rotation,
		// matching that established pattern). ---
		const netTorsoYaw = yawRad + spineYawCounter; // "pelvisYaw + spineYaw" per spec's own item 7 wording
		// "total spine pitch" (spec S6 item 7) = torsoPitch: the ONLY pitch-axis
		// contribution in this rewrite's spine model -- spineLateralLean/breathing
		// rotate about the forward axis (list-like) and pitch axis respectively but
		// on a conceptually different role (lean/breathing, not forward lean), and
		// the pre-existing torso-lean model this reuses was always Spine-only pitch.
		const headCounterPitch = P.headCounterPitchK * torsoPitch + P.gazeDownRad;
		const headCounterYaw = P.headCounterYawK * netTorsoYaw;
		// F8 (integration_2.json diag, 2026-07-10): counter the NET list reaching the
		// head the same way headCounterYaw counters netTorsoYaw above -- pelvisListRad
		// (Hips, step 4) plus spineLateralLean (Spine1+Spine2 already sum to
		// approximately the full spineLateralLean by this point in the chain, step
		// 15) is the total list-axis rotation upstream of Neck/Head. See
		// headCounterListK's own PATIENT_BODY_PARAMS comment for why this exists.
		const netTorsoList = listRad + spineLateralLean;
		const headCounterList = P.headCounterListK * netTorsoList;

		const neckQuat = scratch.q1.setFromAxisAngle( PITCH_AXIS, headCounterPitch / 2 )
			.multiply( scratch.q2.setFromAxisAngle( _UP_AXIS_LOCAL, headCounterYaw / 2 ) )
			.multiply( scratch.q3.setFromAxisAngle( _FORWARD_AXIS_LOCAL, headCounterList / 2 ) );
		b.neck.quaternion.copy( neckQuat );
		const headQuat = scratch.q1.setFromAxisAngle( PITCH_AXIS, headCounterPitch / 2 )
			.multiply( scratch.q2.setFromAxisAngle( _UP_AXIS_LOCAL, headCounterYaw / 2 ) )
			.multiply( scratch.q3.setFromAxisAngle( _FORWARD_AXIS_LOCAL, headCounterList / 2 ) );
		b.head.quaternion.copy( headQuat );

		// F8 fix (integration_2.json diag, 2026-07-10): direct counter-translation,
		// LOCAL Y (same raw/scaled convention Hips.position write above uses --
		// _bindHeadPosition is Head's own bind-pose local position, captured at
		// load() the same way _bindHipsPosition is), reducing how much of the
		// pelvis bob (bobM, step 4) the head ends up carrying through pure FK
		// propagation -- see headBobCounterFrac's own PATIENT_BODY_PARAMS comment
		// for why a rotation-based counter (headCounterListK above) was tried
		// first and measured NOT to move M13_headVsPelvisAmplitudeRatio.
		b.head.position.copy( this._bindHeadPosition );
		b.head.position.y -= P.headBobCounterFrac * bobM / this._armatureScale;

		// --- 19b) Right-arm swing angle for diagnostics (spec S6 item 9 "arm swing
		// angles L/R" -- armSwingLeftDeg above is a direct local variable, but the
		// RIGHT arm has no equivalent scalar: it's either two-bone IK'd to the cane
		// handle (step 17, caneAvailable) or FK-swung by a formula local to that
		// branch's own else-clause. Measured GEOMETRICALLY instead, code-path-agnostic
		// (reads back whatever step 17 actually wrote to b.rightArm.quaternion, whether
		// via IK or FK-fallback, rather than re-deriving/duplicating either branch's
		// formula -- same "trust the real bone over a re-derivation" discipline as this
		// file's own caneHandErrorM comment above): the signed angle, about PITCH_AXIS
		// (same hinge/sign convention as armSwingLeftDeg's own qSwing), between the
		// arm's bind-pose T-pose rest direction and its CURRENT achieved anchor-local
		// direction. This is offset from armSwingLeftDeg's own zero-point by the
		// FIXED (load-time-constant, not per-frame) hang-quaternion the FK-fallback
		// branch composes before its own swing term -- i.e. amplitude/variance across
		// frames (what M11-style diagnostics actually consume) reads correctly; the
		// absolute value is not directly comparable to armSwingLeftDeg's zero-point.
		// Allocation-light: reuses q0/v0/v1 (all free by this point in the frame --
		// q0 last written step 18, v0 last written step 7, v1 last written step 17). ---
		const rightArmAbsQuat = scratch.q0.copy( rightShoulderFrameQuat ).multiply( b.rightArm.quaternion );
		const rightArmDirCurrent = scratch.v1.copy( this._armRestDir.right ).applyQuaternion( rightArmAbsQuat );
		const rightSwingCross = scratch.v0.crossVectors( this._armRestDir.right, rightArmDirCurrent );
		const armSwingRightDeg = THREE.MathUtils.radToDeg( Math.atan2(
			rightSwingCross.dot( PITCH_AXIS ), this._armRestDir.right.dot( rightArmDirCurrent ),
		) );

		// --- 20) Diagnostics (spec S6 item 9): extend _lastSync ADDITIVELY -- every
		// pre-existing key stays exactly as before (main.js reads them), new keys
		// are new properties only. ---
		this._lastSync = {
			leftAnkleTargetWorld: { x: leftTargetWorld.x, y: leftTargetWorld.y, z: leftTargetWorld.z },
			rightAnkleTargetWorld: { x: rightTargetWorld.x, y: rightTargetWorld.y, z: rightTargetWorld.z },
			// RIG-3 fix (AGENTS.md #15): reads anatomicalBendRad (angle BETWEEN the
			// achieved thigh/shin directions), not kneeBend (theta, measured from a
			// non-straight bind pose) -- see _solveTwoBoneIK's own doc for the exact
			// relationship. This keeps the existing 40 deg stance-median bar measuring
			// the physically-meaningful quantity.
			leftKneeBendDeg: THREE.MathUtils.radToDeg( leftIK.anatomicalBendRad ),
			rightKneeBendDeg: THREE.MathUtils.radToDeg( rightIK.anatomicalBendRad ),
			leftPlanted: pose.leftFoot.planted,
			rightPlanted: pose.rightFoot.planted,
			speed: pose.speed,
			// NEW (this rewrite):
			phaseC, support,
			torsoPitch, pelvisListRad: listRad, pelvisYawRad: yawRad, pelvisBobM: bobM, pelvisShiftM: shiftM,
			spineYawCounter, spineLateralLean, breathing,
			leftFootRollPitchDeg: THREE.MathUtils.radToDeg( leftRollPitch ),
			rightFootRollPitchDeg: THREE.MathUtils.radToDeg( rightRollPitch ),
			leftToePitchDeg: THREE.MathUtils.radToDeg( leftToePitch ),
			rightToePitchDeg: THREE.MathUtils.radToDeg( rightToePitch ),
			leftHeelStrikeActive: leftHasRoll && typeof pose.leftFoot.landedAt === 'number' && ( tq - pose.leftFoot.landedAt ) >= 0 && ( tq - pose.leftFoot.landedAt ) <= P.rollDownSec,
			rightHeelStrikeActive: rightHasRoll && typeof pose.rightFoot.landedAt === 'number' && ( tq - pose.rightFoot.landedAt ) >= 0 && ( tq - pose.rightFoot.landedAt ) <= P.rollDownSec,
			// F2a (integration_2.json diag, 2026-07-10): the CONTACT point the roll
			// math above pivots about (heel/toe/flat plant point, or the current swing
			// arc position while airborne) -- see step 6's own comment. main.js's
			// plantedDrift/M8_rollWindowContactDriftMax feature-detect this field and
			// measure ITS drift instead of the toe bone's (which legitimately moves a
			// few cm during a roll even though this point, by construction, never does).
			leftFootContact: leftContact,
			rightFootContact: rightContact,
			leftToeOffActive: leftHasRoll && typeof pose.leftFoot.nextLiftAt === 'number' && ( pose.leftFoot.nextLiftAt - tq ) >= 0 && ( pose.leftFoot.nextLiftAt - tq ) <= P.heelOffSec,
			rightToeOffActive: rightHasRoll && typeof pose.rightFoot.nextLiftAt === 'number' && ( pose.rightFoot.nextLiftAt - tq ) >= 0 && ( pose.rightFoot.nextLiftAt - tq ) <= P.heelOffSec,
			armSwingLeftDeg: THREE.MathUtils.radToDeg( leftSwingRad ),
			armSwingRightDeg, // see step 19b's own comment for the geometric (code-path-agnostic) derivation and its zero-point caveat
			armSwingRightAvailable: caneAvailable,
			caneAvailable,
			caneHandleTargetWorld: caneAvailable ? { x: canePoseResult.handle.x, y: canePoseResult.handle.y, z: canePoseResult.handle.z } : null,
			// caneHandErrorM intentionally NOT computed here: main.js's patientDiag
			// (VERIFY-owned, spec S8 M12 "cane hand-to-handle error") already reads
			// real bones via getWorldPosition at its own diagnostic call sites, which
			// is the more trustworthy measurement (exercises the ACTUAL rendered
			// skeleton/parenting, not a second from-scratch analytic reconstruction
			// that could silently drift from what sync() really wrote -- see AGENTS.md
			// incident #4's closing note on trusting numeric bone dumps over
			// re-derivations). caneHandleTargetWorld above is exposed so that
			// diagnostic can compute the error without re-deriving the target itself.
			// F6 (integration_2.json diag, 2026-07-10): caneHandleTargetWorld above is
			// the NOMINAL pre-clamp target -- the right-hand two-bone IK (step 17)
			// actually aims at caneHandleEffectiveWorld, which equals
			// caneHandleTargetWorld except while caneReachClamped (arm out of reach),
			// when it's the re-aimed adjustedHandle instead. M12a previously compared
			// the hand bone against the pre-clamp target unconditionally, reading a
			// large "error" that was really just the (expected, by-design) clamp
			// offset -- see step 17's own comment for where this is computed.
			caneHandleEffectiveWorld: caneHandleEffective,
			caneReachClamped,
			caneLeanDeg: caneAvailable ? THREE.MathUtils.radToDeg( canePoseResult.leanRad ) : null,
			ikSelfCheckFailed: this.ikSelfCheckFailed,
		};

	}

	/**
	 * Set `footBone`'s LOCAL quaternion (relative to Leg) so the foot's ANCHOR-LOCAL
	 * orientation is: PLANTED -> world-flat sole at `footPose.yaw` PLUS the foot-roll
	 * `rollPitch` (NEW this rewrite, spec S6b -- 0 outside heel-strike/toe-off
	 * windows, reducing this exactly to the pre-existing flat behavior); SWINGING ->
	 * the same flat-at-yaw base, blended fromYaw->toYaw, with `rollPitch` now carrying
	 * the FULL continuous swing profile (replacing the old ad-hoc
	 * `0.12*sin(pi*u^0.7)` modulation entirely -- see _footRollPitch's own doc for why
	 * the new profile is a strict superset/replacement, not an addition on top of the
	 * old one).
	 *
	 * `shinAnchorLocalQuat` = hipQuat (the leg IK's ABSOLUTE/anchor-local desired hip
	 * orientation -- UNCHANGED meaning from the pre-existing module even though the
	 * ACTUAL UpLeg bone now stores a Hips-compensated LOCAL value; see sync()'s own
	 * comment at the call site for why passing the absolute value here, not the
	 * compensated one, is still correct) composed with Leg's own local kneeBend
	 * rotation -- i.e. the shin's own anchor-local orientation, computed
	 * ANALYTICALLY (no matrixWorld read).
	 */
	_orientFoot( footBone, hipQuat, legLocalAngle, footPose, rollPitch, scratch ) {

		const shinAnchorLocalQuat = scratch.q0.copy( hipQuat ).multiply(
			scratch.q1.setFromAxisAngle( PITCH_AXIS, legLocalAngle ),
		);

		// Desired foot-bone orientation for a world-flat sole pointing at footPose.yaw
		// (UNCHANGED derivation from the pre-existing module -- see git history for
		// the full "B_PLACEMENT is the flat/bind foot orientation, not identity" note;
		// unrepeated here to keep this rewrite's new material easy to find).
		const desiredWorldQuat = scratch.q1.setFromAxisAngle( _UP_Z, footPose.yaw );
		const desiredAnchorLocal = scratch.rootQuatInv.clone().multiply( desiredWorldQuat );
		desiredAnchorLocal.premultiply( B_PLACEMENT_INV );
		desiredAnchorLocal.multiply( B_PLACEMENT );

		let footLocal = shinAnchorLocalQuat.clone().invert().multiply( desiredAnchorLocal );

		// Roll pitch (heel-strike/toe-off/swing, spec S6b) -- applied in the FOOT'S
		// OWN local frame (on top of the flat-at-yaw base), same "reads as an ankle
		// articulation, not a re-aim of the whole foot" reasoning the pre-existing
		// swing-only modulation already established.
		if ( rollPitch !== 0 ) {

			const rollQuat = scratch.q1.setFromAxisAngle( PITCH_AXIS, rollPitch );
			footLocal = footLocal.multiply( rollQuat );

		}

		footBone.quaternion.copy( footLocal );

	}

}

/** adv(t) = clamp(((otherFoot.x,y) - root)*facing_fwd / reachRefM, -1, 1) -- spec S6
 *  item 6's own formula (named generically here since it drives EITHER arm's
 *  contralateral signal: the left arm's real adv_R, or the right arm's fallback
 *  mirror when no cane schedule exists, see sync()'s own step 17/18). Continuous
 *  (works whether the referenced foot is planted or swinging) and freezes at idle
 *  (both feet converge toward the same root-relative offset once stepping stops).
 *
 *  F4 (integration_2.json diag, 2026-07-10): takes `rootX/rootY/rootYaw` EXPLICITLY
 *  rather than reading `pose.rootX/rootY/rootYaw` itself (the pre-existing shape) --
 *  those `poseAt()` fields are the RAW recorded root track, unaware of sync()'s own
 *  walk-on-tail glide (step 0). During a walk-on tail the rendered root has already
 *  advanced (synthesized glide) while `pose.rootX/Y` stays pinned near the recorded
 *  track's own near-frozen creep (that mismatch is WHY the tail exists at all -- see
 *  _buildWalkOn's header), so comparing a synthesized, ACTUALLY-MOVING foot
 *  (`otherFootPose`, which DOES include the walk-on's synthesized steps) against the
 *  stale recorded root produced large, spurious swings classified as "idle" by
 *  main.js's speed<0.02 check (which ALSO reads the stale recorded-track speed) --
 *  M11_armSwingIdleAmplitude measured 0.0605 rad against a 0.01 rad bar. Callers now
 *  pass the SAME tail-resolved root sync() already computed for the anchor
 *  (rootPosX/rootPosY/currentYawRad), so adv is continuous through the tail exactly
 *  like it is through the real march.
 *
 *  R2 (round-2 diag): `reachRefM` is no longer a bare `PATIENT_BODY_PARAMS` constant
 *  at the call sites -- both callers now pass sync()'s own `armReachRefM` (stride-
 *  adaptive, see its call-site comment near step 17) instead of the old fixed
 *  `armSwingReachRefM`. This function's own contract (a plain scalar reach reference
 *  in meters) is unchanged; only what the caller computes and hands in changed. */
function _clampedAdvance( rootX, rootY, rootYaw, otherFootPose, reachRefM ) {

	const fwdX = Math.cos( rootYaw ), fwdY = Math.sin( rootYaw );
	const dx = otherFootPose.x - rootX, dy = otherFootPose.y - rootY;
	const adv = ( dx * fwdX + dy * fwdY ) / reachRefM;
	return THREE.MathUtils.clamp( adv, - 1, 1 );

}
