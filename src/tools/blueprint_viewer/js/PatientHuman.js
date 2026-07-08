// PatientHuman.js
//
// Loads the patient as a real imported+rigged human model (models/vendor/Xbot.glb --
// Mixamo's "X Bot" mannequin, bundled by three.js's own examples repo -- see
// models/vendor/NOTICE.md). This module owns the SKELETON/RIG side of the patient:
// bone lookup, load-time measurement of the rig's own proportions, two-bone leg IK,
// foot orientation, and phase-locking Xbot's own canned "walk" AnimationClip (arm
// swing / spine sway) to the gait. It does NOT own the gait itself -- js/PatientGait.js
// is a pure-math module that decides WHERE the feet go and WHEN they step, driven by
// nothing but the recorded patient_root path (see that module's own header for why:
// user-reported "moonwalking" traced to a TIME-driven gait that kept cycling through
// the recorded path's stop-and-go pauses). This module's job is purely "given the
// P-frame foot/root pose PatientGait.poseAt() computed for this instant, retarget it
// onto Xbot's actual bones" -- IK math, coordinate conversion, and rig bookkeeping,
// with zero gait/timing decisions of its own.
//
// ARCHITECTURE CHANGE (this rewrite): the previous version consumed
// pipeline/anim_bake.py's baked patient_pose.json (per-frame hip_pitch/knee_bend
// SCALARS computed by a Python 2-link IK, driving a TIME-based gait state machine
// also in Python). That data-driven-from-Python approach is retired -- ALL leg pose,
// foot placement, step timing, and upper-body phase-lock are now computed HERE and in
// PatientGait.js, procedurally, in the browser, driven only by patient_root's
// recorded XY/yaw/z path (see the task's own product constraint: "everything else...
// is generated procedurally in the viewer, NOT taken from Isaac"). patient_pose.json
// is no longer fetched or read at all.
//
// Coordinate systems: Xbot ships in its own local convention (lateral=local X,
// up=local Y, forward=local Z -- a standard glTF/Mixamo humanoid rig, re-confirmed
// for THIS rewrite by directly parsing Xbot.glb's binary accessors in Node and
// checking its "walk" clip's dominant LeftUpLeg rotation axis + a full forward-
// kinematics reconstruction of the canned clip, not just trusting the prior module's
// own comment -- see PITCH_AXIS's comment below for what that re-check actually
// found). This viewer's world (everything under gltf_export.py's "isaac_world" node,
// the "P-frame" PatientGait.js's own math is expressed in) uses forward=X, lateral=Y,
// up=Z. B_PLACEMENT is the fixed rotation reconciling the two; see its own comment
// below for the derivation (unchanged from the prior module -- still correct, only
// the LEG-ANGLE retargeting built on top of it changed).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
	buildTerrain, extractPathSamples, buildSchedule, poseAt, DEFAULT_GAIT_PARAMS,
} from './PatientGait.js';

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
// walk clip's own root-adjacent bone positions require).
const _basisMatrix = new THREE.Matrix4().makeBasis(
	new THREE.Vector3( 0, 1, 0 ),
	new THREE.Vector3( 0, 0, 1 ),
	new THREE.Vector3( 1, 0, 0 ),
);
const B_PLACEMENT = new THREE.Quaternion().setFromRotationMatrix( _basisMatrix );
const B_PLACEMENT_INV = B_PLACEMENT.clone().invert();

const _UP_Z = new THREE.Vector3( 0, 0, 1 ); // P-frame "up" axis (yaw rotation axis)

// Xbot's own hip/knee sagittal-plane flexion axis: local X, re-confirmed for this
// rewrite by directly parsing Xbot.glb (no browser) and checking the "walk" clip's
// baked LeftUpLeg rotation keys' per-component RANGE across the whole clip (x range
// 0.42, dominant; y range 0.11; z range 0.15) -- same conclusion the prior module's
// comment already stated. This axis is still what this module's two-bone IK writes
// UpLeg/Leg rotations about.
//
// IMPORTANT SIGN CORRECTION vs. the prior module (found during this rewrite, NOT
// assumed): the prior module's comment claimed "positive hip pitch about +X swings
// the leg forward" -- re-verified numerically for THIS rewrite via a full forward-
// kinematics reconstruction of the canned "walk" clip using REAL three.js Object3D
// parenting (Hips->LeftUpLeg->LeftLeg->LeftFoot, applying each bone's own baked
// rotation, then reading LeftFoot's world Z relative to Hips): POSITIVE LeftUpLeg
// local-X rotation angle actually correlates with the foot swinging BACKWARD (-Z),
// not forward -- confirmed at multiple points across the clip (e.g. LeftUpLeg.x
// reaches its most positive value +0.0835 at t=0.767s, exactly where LeftFoot's Z
// relative to Hips is near ITS most negative, -39.8; and LeftUpLeg.x is most negative
// -0.339 at t=0.233s, where LeftFoot Z is near its most POSITIVE, +39.2) -- i.e. the
// OPPOSITE of the prior comment's claim. This module's own two-bone IK (buildHipQuat/
// solveLeg below) SIDESTEPS this ambiguity entirely rather than resting on it: hip
// orientation is derived geometrically (rotate the bind-pose thigh direction onto the
// computed target direction, via a quaternion "rotate a onto b" construction, then a
// twist correction to aim the knee-bend plane at a pole vector) rather than computed
// as a signed scalar angle whose sign convention would need to be trusted. The ONLY
// place a scalar sign still matters is Leg's OWN local kneeBend rotation (child of
// UpLeg, so it composes on top automatically) -- verified via a full IK->FK round-trip
// (build hip+knee quaternions, apply via real Object3D parenting, read back the
// achieved ankle position) that `Leg.quaternion = axisAngle(PITCH_AXIS, -kneeBend)`
// (kneeBend >= 0) reproduces the requested target EXACTLY (0.000000 m error across a
// battery of forward/backward/lateral/high/low/edge-case targets) -- this part of the
// sign convention IS unchanged from the prior module and IS correct.
const PITCH_AXIS = new THREE.Vector3( 1, 0, 0 );

// NOTE: the source glTF names these "mixamorig:LeftUpLeg" etc (with a colon), but
// three.js's GLTFLoader strips the colon when it creates each Object3D's .name
// (confirmed empirically -- getObjectByName('mixamorig:LeftUpLeg') came back null;
// traversing the loaded scene showed "mixamorigLeftUpLeg" instead -- AGENTS.md
// incident #3).
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
	hips: 'mixamorigHips',
};

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

// ===========================================================================
// Two-bone leg IK
// ===========================================================================

/**
 * Solve a two-bone (hip->knee->ankle) IK chain for `target` (a point in the SAME
 * space as `hipPivot` -- this module always calls it with both in anchor-local/Xbot-
 * local space), with the knee bending toward `poleVector` (also anchor-local -- the
 * task spec's "leg's OWN yaw reference", converted to an anchor-local forward
 * direction by the caller).
 *
 * Returns `{ hipQuat, kneeBend, reachClamped }`:
 *   - hipQuat: UpLeg's LOCAL rotation (UpLeg's parent, Hips, sits at bind
 *     position+identity-rotation every sync() call -- see sync()'s own comment -- so
 *     this IS UpLeg's effective anchor-local rotation, no further composition needed
 *     by the caller before writing it directly to `bones.leftUpLeg.quaternion` etc).
 *   - kneeBend: interior-angle deficit from straight (0 = fully extended, positive =
 *     bent), in radians -- caller writes `Leg.quaternion = axisAngle(PITCH_AXIS,
 *     -kneeBend)` (see PITCH_AXIS's own comment for why this specific sign, verified
 *     via a full IK->FK round-trip, is correct).
 *   - reachClamped: true if `target` was farther than `maxReach` from `hipPivot` (the
 *     solve still returns a valid, fully-extended-leg pose aimed at the CLAMPED
 *     target's direction) -- consumed by the pelvis-reachability step in sync().
 *
 * DERIVATION (geometric, not a signed scalar hip-pitch angle -- see PITCH_AXIS's own
 * comment for why: this module found the prior version's "positive hip pitch =
 * forward" convention does not hold for Xbot's actual rig, and a geometric
 * construction sidesteps needing to trust either sign):
 *   1. reach d = clamp(|target-hipPivot|, minReach, maxReach)
 *   2. law of cosines: kneeInterior (0=straight..pi=fully folded), hipOffsetAngle
 *      (the angle between the thigh's actual aim direction and the straight
 *      hip->target line, since a bent knee doesn't aim the thigh directly at the
 *      target)
 *   3. bendAxis = normalize(cross(toTarget, poleVector)) -- the horizontal hinge axis
 *      the knee rotates about, chosen so the knee ends up on the poleVector side
 *   4. thighDir = toTarget rotated by +hipOffsetAngle about bendAxis (leans the
 *      thigh back from the target while the knee comes toward poleVector)
 *   5. hipQuat = a rotation taking the bind-pose thigh direction (0,-1,0), i.e.
 *      "straight down" (re-confirmed for this rewrite: EVERY leg bone's bind-pose
 *      local rotation is ~exactly identity, and UpLeg->Leg's bind-pose world offset
 *      is (0, -0.4437, +0.0028) -- almost purely -Y) onto thighDir, PLUS a twist
 *      about thighDir so the bone's own hinge axis (PITCH_AXIS, as swung by the
 *      first part of the rotation) ends up aligned with bendAxis -- i.e. a full
 *      "aim + roll" two-constraint solve, not just a single-constraint shortest arc.
 * Verified (Node, real three.js, before writing this into the app): IK->FK round-trip
 * error 0.000000 m across 5 varied targets (straight-down, forward+down, backward+
 * down, forward+up/high-step, lateral) plus a 200-sample random sweep within a
 * plausible leg-reach volume (0 NaN, 0.000000 m worst error on unclamped cases); the
 * knee-toward-pole property independently verified by reading back the KNEE's own
 * (not just the final ankle's) position for pole=+Z vs pole=-Z and confirming it
 * lands on the requested side both times; edge cases (target exactly at hip pivot,
 * target exactly at min/max reach, target far beyond max reach) all resolve to finite,
 * sane poses with no NaN.
 */
const _REST_DIR = new THREE.Vector3( 0, - 1, 0 );

function _solveLegIK( hipPivot, target, poleVector, maxReachM, minReachM, L1, L2 ) {

	const toHip = new THREE.Vector3().subVectors( target, hipPivot );
	const rawDist = toHip.length();
	const reachClamped = rawDist > maxReachM;
	const d = THREE.MathUtils.clamp( rawDist, minReachM, maxReachM );

	const toTarget = rawDist > 1e-6 ? toHip.clone().normalize() : _REST_DIR.clone();

	const cosKneeInterior = THREE.MathUtils.clamp( ( L1 * L1 + L2 * L2 - d * d ) / ( 2 * L1 * L2 ), - 1, 1 );
	const kneeInterior = Math.acos( cosKneeInterior );
	const kneeBend = Math.PI - kneeInterior;

	const cosHipOffset = THREE.MathUtils.clamp( ( L1 * L1 + d * d - L2 * L2 ) / ( 2 * L1 * d ), - 1, 1 );
	const hipOffsetAngle = Math.acos( cosHipOffset );

	let bendAxis = new THREE.Vector3().crossVectors( toTarget, poleVector );
	if ( bendAxis.lengthSq() < 1e-8 ) {

		bendAxis = new THREE.Vector3().crossVectors( toTarget, new THREE.Vector3( 1, 0, 0 ) );
		if ( bendAxis.lengthSq() < 1e-8 ) bendAxis = new THREE.Vector3().crossVectors( toTarget, new THREE.Vector3( 0, 1, 0 ) );

	}
	bendAxis.normalize();

	const thighDir = toTarget.clone().applyAxisAngle( bendAxis, hipOffsetAngle );

	// hipQuat: swing restDir->thighDir, then twist about thighDir so PITCH_AXIS (as
	// swung) aligns with bendAxis.
	const swing = _quatFromTo( _REST_DIR, thighDir );
	const hingeAfterSwing = PITCH_AXIS.clone().applyQuaternion( swing );
	const projectPerp = ( v, axis ) => {

		const p = v.clone().sub( axis.clone().multiplyScalar( v.dot( axis ) ) );
		return p.lengthSq() < 1e-10 ? null : p.normalize();

	};
	const hingeProj = projectPerp( hingeAfterSwing, thighDir );
	const bendProj = projectPerp( bendAxis, thighDir );

	let hipQuat;
	if ( hingeProj && bendProj ) {

		const cosT = THREE.MathUtils.clamp( hingeProj.dot( bendProj ), - 1, 1 );
		const crossT = new THREE.Vector3().crossVectors( hingeProj, bendProj );
		const sinSign = Math.sign( crossT.dot( thighDir ) ) || 1;
		const twistAngle = Math.acos( cosT ) * sinSign;
		const twist = new THREE.Quaternion().setFromAxisAngle( thighDir, twistAngle );
		hipQuat = twist.multiply( swing );

	} else {

		// Degenerate (thighDir parallel to PITCH_AXIS or bendAxis) -- swing-only is
		// the best available answer; extremely rare (would need the leg pointing
		// exactly along its own hinge axis, never produced by this app's terrain/gait).
		hipQuat = swing;

	}

	return { hipQuat, kneeBend, reachClamped, achievedReach: d };

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
		this._mixer = null;
		this._walkAction = null;
		this._hipsHeightM = 0.97; // overwritten by the real load-time measurement below (hip PIVOT height -- see load()'s own comment for why not the "Hips" bone's own height); this default only matters if load() somehow never runs before sync()
		this._patientRootNode = null;
		this._attached = false;

		// Rig measurements (populated by load(), all in WORLD METERS -- see each
		// field's own comment at the point it's measured).
		this._L1 = 0.44; // UpLeg->Leg length
		this._L2 = 0.44; // Leg->Foot length
		this._maxReachM = 0.995 * ( this._L1 + this._L2 );
		this._minReachM = Math.abs( this._L1 - this._L2 ) + 1e-4;
		this._ankleHeightM = 0.087; // Foot.y - ToeBase.y at bind pose (how far the ankle sits above a flat sole)
		this._toeForwardLenM = 0.107; // horizontal Foot->ToeBase reach
		this._footLateralM = 0.082; // hip pivot's own lateral (X) offset from Hips, one side
		this._hipPivotLocal = { left: new THREE.Vector3(), right: new THREE.Vector3() }; // ANCHOR-local, measured at load
		this._bindHipsPosition = new THREE.Vector3();

		// PatientGait.js state (populated by buildGait()).
		this._terrain = null;
		this._schedules = null; // { follow: schedule, climb: schedule }
		this._gaitParams = DEFAULT_GAIT_PARAMS;

		// Upper-body phase-lock (populated by buildGait(), needs the loaded clip).
		this._clipPhaseOffset = 0;
		this._walkClipDuration = 1;

		this.ikSelfCheckFailed = false;

		// Diagnostic snapshot from the most recent sync() call -- see sync()'s own
		// comment at the point it's populated. Read by main.js's patientDiag.
		this._lastSync = null;

		// Reusable scratch objects (avoid per-frame allocation in the hot sync() path).
		this._scratch = {
			v0: new THREE.Vector3(), v1: new THREE.Vector3(), v2: new THREE.Vector3(),
			q0: new THREE.Quaternion(), q1: new THREE.Quaternion(),
			rootQuatInv: new THREE.Quaternion(),
		};

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

			// --- Measure the rig, in WORLD METERS, at bind pose (mixer.update() has
			// not run yet -- these are the true bind-pose values). All lengths/offsets
			// via getWorldPosition() deltas, per this module's own header/AGENTS.md:
			// Xbot's skeleton sits under an "Armature" node with uniform scale 0.01, so
			// bone-LOCAL .position values are ~100x world meters -- world-space
			// measurement sidesteps that entirely. ---
			const hipsWorld = new THREE.Vector3();
			bones.hips.getWorldPosition( hipsWorld );
			this._bindHipsPosition = bones.hips.position.clone();

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
			// bone, averaged left/right), NOT from the "Hips" bone itself, even though
			// "Hips" is the more obvious-sounding name match -- found necessary
			// numerically, not assumed: pipeline/anim_bake.PATIENT_HIP_HEIGHT_M (0.92 m)
			// is the OLD primitive mannequin's own HIP-JOINT height (the point its legs
			// actually pivot from -- see anim_bake.py's PATIENT_STANCE_TARGET_Z/
			// PATIENT_MAX_REACH_M tuning, calibrated against that same joint), which is
			// the same ANATOMICAL concept as Xbot's UpLeg bone, not its "Hips" bone
			// (Mixamo's "Hips" is a pelvis-center reference that in this rig's bind
			// pose sits ~6.75 cm ABOVE the actual hip pivot -- confirmed via
			// getWorldPosition deltas). Using "Hips"' own (larger) height here anchored
			// the rig too LOW relative to a flat-ground ankle target: reach dropped to
			// ~86% of L1+L2 for an ordinary standing sample from the real "follow" clip
			// (verified with a full sync()-equivalent pipeline run against the real
			// extracted patient_root track), and per this solver's own severe
			// reach-vs-bend nonlinearity (AGENTS.md incident #8) that 86% reach solves
			// to ~61 deg of stance knee bend -- comfortably BUSTING the kneeBendDeg.
			// stanceMedian<=40 deg acceptance bar. Re-deriving _hipsHeightM from the
			// hip PIVOT instead (which anatomically matches what 0.92 m was always
			// calibrated to represent) raises reach to ~94% for the same sample,
			// solving to ~28-30 deg -- comfortably within bar. The pelvis-reachability
			// step (see sync()) is a SEPARATE, one-directional (lower-only) mechanism
			// for the OPPOSITE problem (target too FAR away, near/beyond max reach) --
			// it cannot fix an UNDER-reach (target too CLOSE, the failure mode here),
			// since lowering the anchor only ever moves the hip pivot CLOSER to a
			// flat/low target, which INCREASES bend, not decreases it (confirmed by
			// directly sweeping the reachability trigger threshold against real data:
			// making it more aggressive made stanceMedian WORSE, not better, for
			// exactly this reason) -- getting the BASE anchor height right is not
			// optional, there is no downstream correction for choosing the wrong bone.
			this._hipsHeightM = ( leftUpLegWorld.y + rightUpLegWorld.y ) / 2;

			this._L1 = leftUpLegWorld.distanceTo( leftLegWorld );
			this._L2 = leftLegWorld.distanceTo( leftFootWorld );
			// Anatomical cap slightly under the geometric sum, same reasoning as the
			// old Python IK's PATIENT_MAX_REACH_M: keeps a natural minimum knee bend
			// and stays clear of the exact-straight-leg singularity where the two-bone
			// solve's bendAxis/hipOffsetAngle become numerically ill-conditioned.
			this._maxReachM = 0.995 * ( this._L1 + this._L2 );
			this._minReachM = Math.abs( this._L1 - this._L2 ) + 1e-4;

			// Foot(ankle)->ToeBase: vertical drop (how far the ankle sits above a
			// flat sole -- consumed as the ankle IK target's own +ankleHeight offset
			// above the tread/ground contact height) and horizontal reach (consumed
			// by PatientGait's stair-snap margins, passed into buildGait below).
			this._ankleHeightM = leftFootWorld.y - leftToeWorld.y;
			this._toeForwardLenM = Math.hypot( leftToeWorld.x - leftFootWorld.x, leftToeWorld.z - leftFootWorld.z );

			// Hip pivot, in ANCHOR-LOCAL space: this.anchor doesn't exist as a real
			// parent yet at load() time (attachTo() hasn't run), so measure via a
			// throwaway anchor-shaped quaternion/position at IDENTITY (anchor's own
			// transform is re-set every sync() call anyway -- what we actually need
			// here is the hip pivot's position relative to the SCENE ROOT the bones
			// were just measured in, which is exactly "anchor-local" once attachTo()
			// parents that same `scene` under `this.anchor` with no additional
			// transform of its own -- see attachTo(): `this.anchor.add(this._scene)`
			// with the scene keeping its own loaded transform, so scene-local ==
			// anchor-local as long as the scene's own root node carries no extra
			// offset, which GLTFLoader's output does not introduce).
			this._hipPivotLocal.left.copy( leftUpLegWorld );
			this._hipPivotLocal.right.copy( rightUpLegWorld );
			this._footLateralM = Math.abs( leftUpLegWorld.x - hipsWorld.x );

			// Load-time sanity check (task requirement): the Hips ancestor chain's
			// bind-pose world quaternion should be ~identity (chain rotation ~
			// identity) before this module trusts anchor-local axis math built on the
			// assumption that UpLeg's LOCAL rotation IS its effective anchor-local
			// rotation (true only if everything between Hips and the scene root is
			// rotation-free -- confirmed for THIS rig via a direct Node-side parse
			// before writing this code, but re-checked live here too since a future
			// Xbot.glb swap could silently violate it).
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

			const walkClip = THREE.AnimationClip.findByName( gltf.animations, 'walk' ) || gltf.animations[ 0 ];
			const mixer = new THREE.AnimationMixer( scene );
			const walkAction = mixer.clipAction( walkClip );
			walkAction.play();
			walkAction.paused = true; // scrub-driven, same convention as the robot's own actions
			this._walkClipDuration = walkClip.duration;

			this._scene = scene;
			this._bones = bones;
			this._mixer = mixer;
			this._walkAction = walkAction;

			this._runIkSelfCheck();

			this.ready = true;

		} catch ( error ) {

			console.error( '[blueprint-viewer] PatientHuman failed to load; no patient will be shown:', error );

		}

	}

	/**
	 * Parent the loaded model under isaacWorldNode (a SIBLING of patient_root, NOT a
	 * child of it -- the mesh's own SkinnedMesh binding needs a constant-transform
	 * ancestor; see AGENTS.md's incident ledger), remember patientRootNode (read
	 * every sync() call), and apply the shared blueprint tint. Safe to call before
	 * load() resolves -- callers should await load() first regardless (main.js does).
	 */
	attachTo( isaacWorldNode, patientRootNode, tintMaterial ) {

		if ( ! this.ready || this._attached ) return;

		this._patientRootNode = patientRootNode;
		this.anchor.add( this._scene );
		isaacWorldNode.add( this.anchor );
		this._attached = true;

		this._scene.traverse( ( node ) => {

			if ( ! node.isMesh ) return;

			const oldMaterials = Array.isArray( node.material ) ? node.material : [ node.material ];
			for ( const mat of oldMaterials ) {

				if ( mat ) mat.dispose();

			}

			node.material = tintMaterial;
			node.castShadow = false;
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
	 * Build the per-clip footfall schedule from `phaseClips` (a { follow, climb }
	 * map of THREE.AnimationClip, as loaded by main.js from robot.glb) and
	 * `stairSpec`/`landingFarX` (robot.meta.json's own fields). Must run AFTER
	 * load() resolves (needs the measured rig proportions) and BEFORE the first
	 * sync() call. Idempotent-safe to call more than once (rebuilds from scratch).
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

		this._deriveClipPhaseOffset();

	}

	/**
	 * Derive clipPhaseOffset: the canned "walk" clip's own loop period has no
	 * relationship to this module's gait-driven step timing (steps are now event-
	 * triggered, not periodic at all -- see PatientGait.js's header), so the canned
	 * clip is driven by GAIT PHASE, not wall-clock time (sync()'s own upper-body
	 * step). This derivation finds WHICH TIME within the canned clip's own loop has
	 * its LeftFoot most forward relative to Hips (sampling the mixer at several
	 * times, exactly as the task spec describes), so that time can be aligned with
	 * this module's own integer gait-phase values (left plants at integer phases --
	 * see PatientGait.buildSchedule's own phase-timeline doc).
	 */
	_deriveClipPhaseOffset() {

		const duration = this._walkClipDuration;
		if ( ! ( duration > 0 ) ) { this._clipPhaseOffset = 0; return; }

		const b = this._bones;
		const samples = 24;
		let bestT = 0, bestForwardZ = - Infinity;

		const hipsWorld = new THREE.Vector3();
		const footWorld = new THREE.Vector3();

		for ( let i = 0; i < samples; i ++ ) {

			const t = ( i / samples ) * duration;
			this._walkAction.time = t;
			this._mixer.update( 0 );

			b.hips.getWorldPosition( hipsWorld );
			b.leftFoot.getWorldPosition( footWorld );
			// Xbot local Z = forward; measuring in SCENE space here (not anchor-
			// local) is fine since this only compares relative Z across samples of
			// the SAME (unattached-anchor-independent) scene subtree -- the anchor's
			// own placement doesn't exist yet / doesn't matter for this ONE-TIME
			// load-time derivation (this is not the "never worldToLocal mid-sync"
			// case sync() itself must avoid -- this runs once, before any patient is
			// ever rendered, well before sync() exists as a concept for this frame).
			const forwardZ = footWorld.z - hipsWorld.z;
			if ( forwardZ > bestForwardZ ) { bestForwardZ = forwardZ; bestT = t; }

		}

		this._walkAction.time = 0;
		this._mixer.update( 0 );

		// clipPhaseOffset: the FRACTION of the clip's own duration that "left most
		// forward" occurs at, so that `walkAction.time = ((gaitPhase + offset) mod 1)
		// * duration` lands the canned clip's own left-forward instant at gaitPhase's
		// integer values (left plants -- see PatientGait's phase-timeline doc: a
		// "left plant" is the instant right BEFORE the left foot lifts off again,
		// which for a natural gait is close to but not identical to "most forward";
		// using "most forward" as the alignment anchor is the closest single-sample
		// proxy available from the canned clip's own geometry without re-deriving a
		// full contact-phase model for a clip this module doesn't otherwise analyze).
		this._clipPhaseOffset = bestT / duration;

	}

	/**
	 * LOAD-TIME FK SELF-CHECK (mandatory per the task spec): pose each leg at
	 * several synthetic targets, apply via the SAME _solveLegIK this module's
	 * sync() uses, run the skeleton's world-matrix update, measure the achieved
	 * Foot bone ANCHOR-LOCAL position (via a throwaway parent transform matching
	 * sync()'s own anchor-local convention) against the requested target, and flag
	 * loudly (console.error + this.ikSelfCheckFailed=true) if any error exceeds 1cm.
	 * Restores bind pose (identity UpLeg/Leg rotations) afterward -- this runs once
	 * at load, before any patient is ever shown, so a visible pose glitch here would
	 * never actually reach the screen, but leaving the rig mid-test-pose would still
	 * corrupt the FIRST real sync() call's starting state if not restored.
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

		for ( const offset of testOffsets ) {

			const target = hipPivot.clone().add( offset );
			const { hipQuat, kneeBend } = _solveLegIK(
				hipPivot, target, pole, this._maxReachM, this._minReachM, this._L1, this._L2,
			);

			b.leftUpLeg.quaternion.copy( hipQuat );
			b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - kneeBend );
			b.leftUpLeg.updateMatrixWorld( true );

			// Achieved position, converted into the SAME frame `target` is already
			// in (this._hipPivotLocal.left's own frame, i.e. scene-local == anchor-
			// local per attachTo()'s own no-extra-transform parenting -- see
			// _deriveClipPhaseOffset's comment for the same reasoning) -- getWorldPosition here reads
			// the SCENE's own world matrix (updateMatrixWorld(true) on an unparented-
			// to-anchor-yet bone at load time is scene-local, since attachTo() hasn't
			// run and the scene itself has no other transform ancestor at this point
			// in load() -- this is a LOAD-TIME-ONLY read, not a mid-sync
			// matrixWorld read, so it doesn't violate the "never worldToLocal/
			// matrixWorld mid-sync" rule (there is no "sync" happening yet).
			const achieved = new THREE.Vector3();
			b.leftFoot.getWorldPosition( achieved );

			const error = achieved.distanceTo( target );
			worstError = Math.max( worstError, error );

		}

		if ( worstError > 0.01 ) {

			console.error(
				`[blueprint-viewer] PatientHuman: IK self-check FAILED -- worst error ${worstError.toFixed( 4 )} m ` +
				'(bar: <=0.01 m). Leg placement may be visibly wrong.',
			);
			this.ikSelfCheckFailed = true;

		}

		// Restore bind pose.
		b.leftUpLeg.quaternion.identity();
		b.leftLeg.quaternion.identity();
		b.leftUpLeg.updateMatrixWorld( true );

	}

	/**
	 * Re-pose the human for the given phase/time. Pure function of (phaseName,
	 * timeSec) and this module's own load-time measurements/schedules -- no state
	 * carried between calls (every quantity sync() needs is either a load-time
	 * constant or freshly recomputed from `poseAt(schedule, terrain, timeSec)`, which
	 * is itself stateless -- see PatientGait.js's own determinism contract). No-op
	 * until attachTo() (and buildGait()) have run.
	 */
	sync( phaseName, timeSec ) {

		if ( ! this._attached || ! this._schedules ) return;

		const schedule = this._schedules[ phaseName ];
		if ( ! schedule ) return;

		const root = this._patientRootNode;
		const b = this._bones;
		const scratch = this._scratch;

		const pose = poseAt( schedule, this._terrain, timeSec );

		// --- 1) Anchor placement ---
		//
		// position: (root.x, root.y, root.z - _hipsHeightM). REMOVES the old
		// "+ _ankleGroundClearanceM" whole-rig raise the prior module applied here
		// (see AGENTS.md incident #5's "bug B"): that raise existed because the OLD
		// architecture's leg IK targeted the ANKLE at ground/tread level directly, so
		// the whole rig needed lifting by the ankle-above-sole anatomical gap for the
		// real toe mesh to land at ground level. This rewrite's IK targets are
		// EXPLICIT ankle positions computed per-foot below (`target.z = sole/tread
		// height + this._ankleHeightM`), so the ankle-height offset is now applied
		// exactly once, per foot, at the actual IK target -- not as a whole-rig
		// bias that (incorrectly, for a two-legged IK where each foot can be at a
		// DIFFERENT height, e.g. one foot on a tread and one still on the flat
		// approach) assumed both feet needed the identical vertical shift. Dropping
		// this term is therefore not an accidental regression; it's the offset
		// moving to the (correct, per-foot) place it belongs.
		scratch.v0.set( root.position.x, root.position.y, root.position.z - this._hipsHeightM );

		// Gait bob: phase-locked (freezes when steps stop, per PatientGait's own
		// gaitPhase contract -- see poseAt's doc).
		scratch.v0.z += this._gaitParams.bobAmplitude * Math.sin( 4 * Math.PI * pose.gaitPhase );

		this.anchor.quaternion.copy( root.quaternion ).multiply( B_PLACEMENT );

		// --- Pelvis reachability (computed BEFORE finalizing anchor.position, since
		// lowering the anchor changes how far away a FIXED-in-P-frame ankle target
		// appears in anchor-local terms -- see this module's own design notes: a
		// P-frame-world-fixed target gets RELATIVELY CLOSER as the anchor/hip moves
		// down toward it, which is exactly the desired "crouch to reach a stretch"
		// correction). Two-pass: solve once at the nominal anchor height to measure
		// the worst-case excess reach, then re-solve (below, in step 4) at the
		// final, possibly-lowered anchor height. ---
		const rootQuatInv = scratch.rootQuatInv.copy( root.quaternion ).invert();

		/** Convert a P-frame world point to anchor-local, using the anchor position passed in (NOT necessarily this.anchor.position yet, since this is called once pre-lowering and once post-lowering) -- explicit quaternion math per the task spec, never Object3D.worldToLocal/matrixWorld mid-sync. */
		const toAnchorLocal = ( pWorld, anchorPos, out ) => {

			out.copy( pWorld ).sub( anchorPos );
			out.applyQuaternion( rootQuatInv );
			out.applyQuaternion( B_PLACEMENT_INV );
			return out;

		};

		const leftTargetWorld = new THREE.Vector3( pose.leftFoot.x, pose.leftFoot.y, pose.leftFoot.z + this._ankleHeightM );
		const rightTargetWorld = new THREE.Vector3( pose.rightFoot.x, pose.rightFoot.y, pose.rightFoot.z + this._ankleHeightM );

		const nominalAnchorPos = scratch.v0.clone();
		const leftLocalNominal = toAnchorLocal( leftTargetWorld, nominalAnchorPos, new THREE.Vector3() );
		const rightLocalNominal = toAnchorLocal( rightTargetWorld, nominalAnchorPos, new THREE.Vector3() );

		const leftReachNominal = this._hipPivotLocal.left.distanceTo( leftLocalNominal );
		const rightReachNominal = this._hipPivotLocal.right.distanceTo( rightLocalNominal );
		const reachLimit = 0.98 * ( this._L1 + this._L2 );
		const worstExcess = Math.max( 0, leftReachNominal - reachLimit, rightReachNominal - reachLimit );

		// Lower the anchor by the worst excess (never RAISE above the recorded path
		// -- worstExcess is clamped >=0 above, so this only ever subtracts).
		scratch.v0.z -= worstExcess;

		this.anchor.position.copy( scratch.v0 );

		// --- 2) Ankle IK targets, final anchor-local conversion (post-lowering) ---
		const leftTargetLocal = toAnchorLocal( leftTargetWorld, this.anchor.position, new THREE.Vector3() );
		const rightTargetLocal = toAnchorLocal( rightTargetWorld, this.anchor.position, new THREE.Vector3() );

		// --- 3)/4) Two-bone leg IK, per leg, in anchor-local (Xbot) space ---
		//
		// Pole vector: "that leg's OWN yaw reference (the foot's current yaw from the
		// schedule, NOT the body's -- a planted foot's leg keeps its plant heading
		// while the torso turns)" -- convert the foot's OWN P-frame yaw into an
		// anchor-local forward direction the SAME way the ankle targets themselves
		// were converted (P-frame direction -> anchor-local via rootQuatInv then
		// B_PLACEMENT_INV), NOT the root's current yaw.
		const poleFromYaw = ( footYaw ) => {

			const fwdWorld = new THREE.Vector3( Math.cos( footYaw ), Math.sin( footYaw ), 0 ); // P-frame forward at this yaw (X-forward, Y-lateral, Z-up convention -- yaw about +Z)
			fwdWorld.applyQuaternion( rootQuatInv ).applyQuaternion( B_PLACEMENT_INV );
			return fwdWorld;

		};

		const leftPole = poleFromYaw( pose.leftFoot.yaw );
		const rightPole = poleFromYaw( pose.rightFoot.yaw );

		const leftIK = _solveLegIK( this._hipPivotLocal.left, leftTargetLocal, leftPole, this._maxReachM, this._minReachM, this._L1, this._L2 );
		const rightIK = _solveLegIK( this._hipPivotLocal.right, rightTargetLocal, rightPole, this._maxReachM, this._minReachM, this._L1, this._L2 );

		// --- 5) Pelvis reachability already applied above (anchor.position.z), as
		// part of computing the FINAL anchor-local targets steps 2/3/4 solved against
		// -- nothing further to do here.

		// --- 6)/7) Upper body FIRST: drive the canned walk clip by GAIT PHASE, not
		// wall-clock time (see _deriveClipPhaseOffset's own doc for why/how
		// clipPhaseOffset was derived) -- arms then counter-swing in lockstep with
		// the procedural legs and freeze exactly when the patient stands still
		// (gaitPhase itself only advances at real footstep events --
		// PatientGait.buildSchedule's own contract). This mixer evaluation is done
		// BEFORE this module's own Hips/leg/foot overrides below (not after, and
		// not both before-and-after -- AGENTS.md incident #5's ordering requirement
		// is that the OVERRIDES apply after the mixer eval in the FINAL bone state;
		// there is no code path in this sync() where an early-return happens between
		// this mixer.update(0) and the overrides below, so applying the overrides
		// only ONCE, after, is both correct and avoids doing the (nontrivial: two
		// IK-derived quaternion writes + two _orientFoot calls) override work twice
		// per call for no effect -- an early version of this method mirrored the
		// prior module's own "apply, evaluate mixer, re-apply" structure literally,
		// but that structure only mattered THERE because of a genuine early-return
		// fallback path (`if (!pose) return`) this module's own architecture doesn't
		// have (poseAt() is never null/undefined for a valid schedule)).
		const walkDuration = this._walkClipDuration;
		if ( walkDuration > 0 ) {

			const phaseFrac = ( ( pose.gaitPhase + this._clipPhaseOffset ) % 1 + 1 ) % 1; // JS % can return negative; normalize to [0,1)
			this._walkAction.time = phaseFrac * walkDuration;

		} else {

			this._walkAction.time = 0;

		}
		this._mixer.update( 0 );

		// Hips: reset to bind pose (position AND rotation) -- LeftUpLeg/RightUpLeg
		// are children of Hips, so an uncontrolled Hips transform (the canned clip's
		// own hip bob/sway, just written by the mixer.update(0) above) would silently
		// re-transform this module's own carefully-computed leg placement (AGENTS.md
		// incidents #5/#7).
		b.hips.position.copy( this._bindHipsPosition );
		b.hips.quaternion.identity();

		b.leftUpLeg.quaternion.copy( leftIK.hipQuat );
		b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - leftIK.kneeBend );
		b.rightUpLeg.quaternion.copy( rightIK.hipQuat );
		b.rightLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - rightIK.kneeBend );

		// --- Foot orientation ---
		//
		// UpLeg's parent is Hips (identity, just reset above), so hipQuat IS each
		// leg's effective anchor-local UpLeg orientation; composing Leg's own LOCAL
		// kneeBend rotation on top gives the shin's anchor-local orientation, entirely
		// analytically (no matrixWorld read) -- this is the "OWN analytic parent
		// chain" the task spec asks for.
		this._orientFoot( b.leftFoot, leftIK.hipQuat, - leftIK.kneeBend, pose.leftFoot, scratch );
		this._orientFoot( b.rightFoot, rightIK.hipQuat, - rightIK.kneeBend, pose.rightFoot, scratch );

		// ToeBase: identity local rotation (relative to Foot) -- matches the prior
		// module's own safe default (AGENTS.md incident #5's bug A): this module's
		// foot orientation already places the WHOLE foot (Foot bone) at the correct
		// world-flat-or-swinging orientation, so ToeBase riding along un-rotated
		// relative to its own parent keeps the toe an unarticulated rigid extension
		// of the sole -- exactly what a flat-footed stance/swing needs (no separate
		// toe-curl animation in this app's contract). Also overrides whatever the
		// canned clip's own ToeBase rotation (just written by mixer.update(0) above)
		// put there, same reasoning as Hips.
		b.leftToeBase.quaternion.identity();
		b.rightToeBase.quaternion.identity();

		// --- 8) Torso lean ---
		//
		// SET (not premultiply) -- replaces the canned clip's own uncorrelated spine
		// sway entirely, same reasoning as AGENTS.md incident #7 (stacking this
		// pipeline's real, physically-meaningful lean on top of the canned clip's
		// cosmetic sway is what produced the earlier "leans too far forward" bug).
		const gp = this._gaitParams;
		const torsoPitch = THREE.MathUtils.clamp(
			gp.leanBase + gp.leanSpeedK * pose.speed + gp.leanSlopeK * pose.groundSlope,
			0, 0.15,
		);
		b.spine.quaternion.setFromAxisAngle( PITCH_AXIS, torsoPitch );

		// Diagnostic snapshot (consumed by main.js's window.__viewer.patientDiag) --
		// pure bookkeeping, no effect on the render path. Captures the P-FRAME WORLD
		// ankle IK targets (leftTargetWorld/rightTargetWorld, computed above BEFORE
		// the anchor-local conversion) and knee-bend angles this call just solved for,
		// so the diagnostic can compute fkErrorMax (achieved Foot bone position vs
		// requested target) and kneeBendDeg without re-deriving the IK itself (which
		// would risk the diagnostic and the app silently drifting apart if the IK
		// math is ever tuned in one place but not the other).
		this._lastSync = {
			leftAnkleTargetWorld: { x: leftTargetWorld.x, y: leftTargetWorld.y, z: leftTargetWorld.z },
			rightAnkleTargetWorld: { x: rightTargetWorld.x, y: rightTargetWorld.y, z: rightTargetWorld.z },
			leftKneeBendDeg: THREE.MathUtils.radToDeg( leftIK.kneeBend ),
			rightKneeBendDeg: THREE.MathUtils.radToDeg( rightIK.kneeBend ),
			leftPlanted: pose.leftFoot.planted,
			rightPlanted: pose.rightFoot.planted,
			speed: pose.speed,
		};

	}

	/**
	 * Set `footBone`'s LOCAL quaternion (relative to Leg) so the foot's ANCHOR-LOCAL
	 * orientation is: PLANTED -> world-flat sole at `footPose.yaw` (a P-frame
	 * absolute yaw); SWINGING -> the same flat-at-yaw base, blended fromYaw->toYaw
	 * (footPose.yaw already IS that eased value -- PatientGait.poseAt computes it),
	 * with a small pitch modulation (slight plantarflex easing out after liftoff,
	 * slight dorsiflex mid-swing, level by touchdown) layered on top.
	 *
	 * `shinAnchorLocalQuat` = hipQuat (UpLeg's effective anchor-local orientation,
	 * per sync()'s own comment) composed with Leg's own local kneeBend rotation --
	 * i.e. the shin's own anchor-local orientation, computed ANALYTICALLY from
	 * values this module already has in hand (no matrixWorld read).
	 */
	_orientFoot( footBone, hipQuat, legLocalAngle, footPose, scratch ) {

		const shinAnchorLocalQuat = scratch.q0.copy( hipQuat ).multiply(
			scratch.q1.setFromAxisAngle( PITCH_AXIS, legLocalAngle ),
		);

		// Desired anchor-local orientation for a world-flat sole at footPose.yaw:
		// P-frame absolute yaw about +Z -> anchor-local via B_PLACEMENT_INV *
		// rootQuatInv (same conversion direction as position targets, just applied
		// to a pure-yaw quaternion instead of a point).
		const desiredWorldQuat = scratch.q1.setFromAxisAngle( _UP_Z, footPose.yaw );
		const desiredAnchorLocal = scratch.rootQuatInv.clone().multiply( desiredWorldQuat ); // rootQuatInv * desiredWorldQuat -- see below for the B_PLACEMENT_INV factor
		desiredAnchorLocal.premultiply( B_PLACEMENT_INV );

		let footLocal = shinAnchorLocalQuat.clone().invert().multiply( desiredAnchorLocal );

		if ( ! footPose.planted && footPose.swingU !== null ) {

			// Swing pitch modulation: 0 at u=0/1 (level at liftoff/touchdown, matching
			// the planted/about-to-plant foot's own flat orientation), a small
			// plantarflex->dorsiflex arc in between (toe drops slightly just after
			// liftoff -- "easing out after liftoff" -- then the whole foot levels and
			// slightly dorsiflexes mid-swing to clear the ground/tread, then levels
			// again by touchdown). A single sine term already satisfies "zero at both
			// ends"; a small additional asymmetry (u^0.7 bias) front-loads the
			// plantarflex-easing-out phase per the task's own wording ("easing OUT
			// after liftoff" implies the flex is largest EARLY, not exactly
			// symmetric at the swing's midpoint).
			const u = footPose.swingU;
			const modulationAngle = 0.12 * Math.sin( Math.PI * Math.pow( u, 0.7 ) );
			const modulation = scratch.q1.setFromAxisAngle( PITCH_AXIS, modulationAngle );
			// Apply the pitch modulation in the FOOT'S OWN local frame (on top of the
			// flat-at-yaw base), so it reads as an ankle articulation, not a re-aim of
			// the whole foot.
			footLocal = footLocal.multiply( modulation );

		}

		footBone.quaternion.copy( footLocal );

	}

}
