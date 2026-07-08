// PatientHuman.js
//
// Loads the patient as a real imported+rigged human model (models/vendor/Xbot.glb --
// Mixamo's "X Bot" mannequin, bundled by three.js's own examples repo -- see
// models/vendor/NOTICE.md) instead of this pipeline's old hand-built primitive
// mannequin, per user feedback that the primitive body "looked bad" and clipped at
// the knee/elbow. A real rigged mesh looks far better than anything buildable from
// primitives -- but its own canned "walk" AnimationClip is a generic flat-ground loop
// that has no idea where OUR stairs' risers are, so playing it straight through the
// climb would clip through or float above the treads. This module resolves that by
// layering: the canned "walk" clip drives the arm swing + spine sway, and every
// sync() call then OVERRIDES the hip position, leg bones, foot/toe bones, and torso
// lean from this pipeline's own data-driven angles (pipeline/anim_bake.py's
// patient_pose output, computed by the same leg IK that used to drive the old
// primitive rig) -- see sync()'s own comment for why the canned clip's hip bob and
// ankle articulation specifically can't be left in (they put the feet through the
// floor/tread).
//
// Coordinate systems: Xbot ships in its own local convention (lateral=local X,
// up=local Y, forward=local Z -- a standard glTF/Mixamo humanoid rig, confirmed by
// inspecting its mesh bounding box and its "walk" clip's dominant rotation axis).
// This viewer's world (everything under gltf_export.py's "isaac_world" node) uses
// forward=X, lateral=Y, up=Z. B_PLACEMENT is the fixed rotation reconciling the two;
// see its own comment below for the derivation.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const XBOT_URL = './models/vendor/Xbot.glb';
const POSE_URL = './models/patient_pose.json';

// Kept in sync with anim_bake.PATIENT_HIP_HEIGHT_M. Used two ways: (1) `_hipsHeightM`
// below defaults to this constant but is normally overwritten with Xbot's OWN
// (different, ~1.04m) bind-pose Hips-bone height once loaded -- this module doesn't
// force Xbot's hip to match the Python mannequin's stylized height; (2) `sync()`'s
// ground-clearance math uses this constant directly to convert `patient_root`'s
// world position (which anim_bake.py's own convention always sets to
// `ground_z + PATIENT_HIP_HEIGHT_M`) back into a ground-relative height -- see
// AGENTS.md incident #14.
const PATIENT_HIP_HEIGHT_M = 0.92;

// Xbot's own local axes, as image vectors in THIS viewer's (forward=X, lateral=Y,
// up=Z) convention: Xbot's local X (lateral) -> our Y, Xbot's local Y (up) -> our Z,
// Xbot's local Z (forward) -> our X. This is a proper (det=+1) rotation -- a cyclic
// axis permutation, not a mirror -- so it preserves rotation handedness/sign, which
// is what lets _retargetAngle below reuse this pipeline's angles UNCHANGED (same
// sign, same magnitude) just aimed at Xbot's own lateral axis instead of ours.
const _basisMatrix = new THREE.Matrix4().makeBasis(
	new THREE.Vector3( 0, 1, 0 ),
	new THREE.Vector3( 0, 0, 1 ),
	new THREE.Vector3( 1, 0, 0 ),
);
const B_PLACEMENT = new THREE.Quaternion().setFromRotationMatrix( _basisMatrix );

// Xbot's own hip/knee sagittal-plane flexion axis: empirically confirmed (not
// guessed) by inspecting its "walk" clip's baked LeftUpLeg rotation keys, whose
// dominant component is the local-X term -- i.e. rotating about local X swings the
// leg forward/back, exactly like this pipeline's own (now-removed) primitive rig
// rotated about local Y for the same motion. See B_PLACEMENT's comment for why the
// SAME scalar angle (no sign flip) is correct on this axis.
const PITCH_AXIS = new THREE.Vector3( 1, 0, 0 );

// NOTE: the source glTF names these "mixamorig:LeftUpLeg" etc (with a colon), but
// three.js's GLTFLoader strips the colon when it creates each Object3D's .name
// (confirmed empirically -- getObjectByName('mixamorig:LeftUpLeg') came back null;
// traversing the loaded scene showed "mixamorigLeftUpLeg" instead).
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

/**
 * Analytic 2-link-plus-toe FK: the WORLD-Y (Xbot local-up) drop from Hips down to
 * ToeBase, for ONE leg, given that leg's current hip_pitch/knee_bend -- see AGENTS.md
 * incident #14 for the full derivation/measurements this replaces (a single
 * bind-pose-measured "ankle clearance" constant, which only holds at hip_pitch=0).
 * Mirrors the exact rotation composition sync() applies to the live bones
 * (leftUpLeg/leftLeg/leftFoot/leftToeBase quaternions below) -- kept as a pure
 * function of each bone's own BIND-POSE local `.position` (never modified elsewhere,
 * only `.quaternion` is) so it can be evaluated for the anchor BEFORE those bones are
 * actually posed this frame, and for both legs, without a scene-graph round trip.
 * Returns the drop in the bones' own raw local units (NOT yet meters -- see
 * `this._localToM` for the conversion applied at the call site).
 */
function _legToeDropLocal( bonesLocal, hipPitch, kneeBend ) {

	const cosP = Math.cos( hipPitch ), sinP = Math.sin( hipPitch );
	const kneeAngle = hipPitch - kneeBend;
	const cosK = Math.cos( kneeAngle ), sinK = Math.sin( kneeAngle );

	// rotate a (y,z) pair about the shared PITCH_AXIS (local X) by the given angle,
	// keep only the resulting y (local "up") component -- x never contributes to y.
	const legY = bonesLocal.upLeg.y + ( bonesLocal.leg.y * cosP - bonesLocal.leg.z * sinP );
	const footY = legY + ( bonesLocal.foot.y * cosK - bonesLocal.foot.z * sinK );
	const toeY = footY + ( bonesLocal.toe.y * cosP - bonesLocal.toe.z * sinP );
	return - toeY; // positive = below Hips

}

/** Linear-interpolated lookup into a (times, values) pair sampled at an arbitrary t (clamped to the array's own range at either end). */
function sampleAt( times, values, t ) {

	if ( ! times || ! times.length ) return 0;
	if ( t <= times[ 0 ] ) return values[ 0 ];
	const last = times.length - 1;
	if ( t >= times[ last ] ) return values[ last ];

	for ( let i = 1; i <= last; i ++ ) {

		if ( times[ i ] >= t ) {

			const t0 = times[ i - 1 ], t1 = times[ i ];
			const frac = t1 > t0 ? ( t - t0 ) / ( t1 - t0 ) : 0;
			return values[ i - 1 ] + frac * ( values[ i ] - values[ i - 1 ] );

		}

	}

	return values[ last ];

}

export class PatientHuman {

	constructor() {

		this.ready = false;
		this.anchor = new THREE.Group();
		this.anchor.name = 'patient_human_anchor';

		this._bones = null;
		this._mixer = null;
		this._walkAction = null;
		this._poseData = null;
		this._hipsHeightM = PATIENT_HIP_HEIGHT_M;
		this._patientRootNode = null;
		this._attached = false;

	}

	/**
	 * Kick off the GLTFLoader + JSON fetch in parallel. Returns a Promise that
	 * resolves once both are ready (or logs + leaves this.ready=false on failure --
	 * the viewer degrades to "no patient shown" rather than inventing a new
	 * placeholder, matching how loadPlaceholder() already handles a total
	 * robot.glb load failure elsewhere in this app).
	 */
	async load() {

		try {

			const loader = new GLTFLoader();
			const [ gltf, poseData ] = await Promise.all( [
				new Promise( ( resolve, reject ) => loader.load( XBOT_URL, resolve, undefined, reject ) ),
				fetch( POSE_URL ).then( ( r ) => r.json() ),
			] );

			const scene = gltf.scene || gltf.scenes[ 0 ];
			scene.updateMatrixWorld( true );

			const bones = {};
			for ( const [ key, name ] of Object.entries( BONE_NAMES ) ) {

				bones[ key ] = scene.getObjectByName( name );
				if ( ! bones[ key ] ) throw new Error( `Xbot.glb: bone ${name} not found` );

			}

			const hipsWorld = new THREE.Vector3();
			bones.hips.getWorldPosition( hipsWorld );
			this._hipsHeightM = hipsWorld.y;
			// Captured BEFORE the walk clip is ever evaluated (mixer.update() hasn't
			// run yet), so this is the true bind-pose local translation -- restored
			// every sync() call to cancel the canned clip's own hip bob (see sync()'s
			// comment for why: leaving it in put the character's whole body, and
			// therefore its feet, up to ~6cm below where _hipsHeightM assumes).
			this._bindHipsPosition = bones.hips.position.clone();

			// anim_bake.py's leg IK targets the ANKLE at ground/tread level (the old
			// hand-built rig's "foot" was a thin box centered right at the ankle
			// joint, so "ankle at ground" WAS "sole at ground" for that rig). Xbot is
			// a real anatomical skeleton: "Foot" is the actual ankle bone, which sits
			// well above the ground, with the sole reached only via LeftToeBase's own
			// translation further down+forward. Landing the ankle exactly at ground
			// (as the old rig's calibration assumes) therefore buries the real toe
			// mesh in the floor by that same anatomical gap. Measure it once from the
			// bind pose (feet resting on Xbot's own y=0 ground) as a SAFE-DEFAULT
			// fallback (used only if a phase has no pose data at all -- see sync()'s
			// early-out) -- the real per-frame compensation is `_legToeDropLocal`
			// below, which depends on the CURRENT hip_pitch, not just this bind value
			// (see AGENTS.md incident #14: a hip_pitch=0-only constant under-corrects
			// at every other hip_pitch, since pitching the foot+toe assembly forward
			// rotates more of ToeBase's forward offset into "downward").
			const footWorld = new THREE.Vector3();
			bones.leftFoot.getWorldPosition( footWorld );
			const toeWorld = new THREE.Vector3();
			bones.leftToeBase.getWorldPosition( toeWorld );
			this._ankleGroundClearanceM = footWorld.y - toeWorld.y;

			// Per-leg bind-pose local offsets (each bone's own `.position`, never
			// touched by sync() -- only `.quaternion` is) for `_legToeDropLocal`,
			// plus the raw-local-unit -> meters scale factor: Xbot's skeleton sits
			// under an "Armature" node with its own uniform scale (0.01, confirmed
			// empirically by walking the bone->parent chain in the browser console),
			// so bone-local `.position` values are ~100x world meters. At bind pose
			// (hip_pitch=knee_bend=0 for both legs), `_legToeDropLocal` evaluates to
			// exactly the Hips->ToeBase drop in raw local units, and `_hipsHeightM -
			// toeWorld.y` is that SAME drop already measured in true world meters --
			// their ratio is the scale factor, derived rather than hard-coded so this
			// module doesn't depend on that implementation detail of the vendored
			// asset.
			const bindDropLocalL = _legToeDropLocal( {
				upLeg: bones.leftUpLeg.position, leg: bones.leftLeg.position,
				foot: bones.leftFoot.position, toe: bones.leftToeBase.position,
			}, 0, 0 );
			this._localToM = ( this._hipsHeightM - toeWorld.y ) / bindDropLocalL;
			this._legBonesLocal = {
				l: {
					upLeg: bones.leftUpLeg.position.clone(), leg: bones.leftLeg.position.clone(),
					foot: bones.leftFoot.position.clone(), toe: bones.leftToeBase.position.clone(),
				},
				r: {
					upLeg: bones.rightUpLeg.position.clone(), leg: bones.rightLeg.position.clone(),
					foot: bones.rightFoot.position.clone(), toe: bones.rightToeBase.position.clone(),
				},
			};

			const walkClip = THREE.AnimationClip.findByName( gltf.animations, 'walk' ) || gltf.animations[ 0 ];
			const mixer = new THREE.AnimationMixer( scene );
			const walkAction = mixer.clipAction( walkClip );
			walkAction.play();
			walkAction.paused = true; // scrub-driven, same convention as the robot's own actions

			this._scene = scene;
			this._bones = bones;
			this._mixer = mixer;
			this._walkAction = walkAction;
			this._poseData = poseData;
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
	 * Re-pose the human for the given phase/time (called every time the robot's own
	 * mixer.update(0) is called -- scrub, phase switch, playback step -- so the two
	 * stay in lockstep). No-op until attachTo() has run.
	 */
	sync( phaseName, timeSec ) {

		if ( ! this._attached ) return;

		const root = this._patientRootNode;
		const pose = this._poseData[ phaseName ];

		// Ground clearance (how far ABOVE GROUND -- not above `root`/the hip -- Xbot's
		// Hips bone must sit so the STANCE foot's TOE, not its ankle, lands on the
		// ground/tread): a fixed bind-pose constant under-corrects whenever
		// hip_pitch != 0 (see AGENTS.md incident #14 -- pitching the foot+toe assembly
		// forward rotates more of ToeBase's forward offset into "downward", so the
		// true ankle-to-toe drop GROWS with hip_pitch). Recompute it per frame from
		// `_legToeDropLocal` (a pure function of each leg's bind-pose local bone
		// offsets, evaluated BEFORE the bones are actually posed below -- this must
		// run first, not after, exactly the ordering hazard flagged in the repo-root
		// CLAUDE.md's "debug_info populated in call order" incident, just in this
		// module instead of that dict), using whichever leg has the SMALLER knee_bend
		// (closer to full leg extension -- the calibrated stance bend, see incident
		// #8) as the one actually bearing weight on the ground; the other (swinging)
		// leg is airborne anyway, so its own toe height doesn't need to be exact.
		// Falls back to the old fixed `_ankleGroundClearanceM` when a phase has no
		// pose data at all (matches the `if ( ! pose ) return` early-out below, which
		// never reaches the per-frame leg override this clearance is calibrated
		// against) -- in that fallback case groundClearanceAboveRootM is added to
		// `root.position.z` directly (the pre-existing, already-correct behavior),
		// same as before this fix.
		let groundClearanceAboveRootM = this._ankleGroundClearanceM;
		let hipPitchL, kneeBendL, hipPitchR, kneeBendR;
		if ( pose ) {

			hipPitchL = sampleAt( pose.times, pose.hip_pitch_l, timeSec );
			kneeBendL = sampleAt( pose.times, pose.knee_bend_l, timeSec );
			hipPitchR = sampleAt( pose.times, pose.hip_pitch_r, timeSec );
			kneeBendR = sampleAt( pose.times, pose.knee_bend_r, timeSec );
			const stanceSide = kneeBendL <= kneeBendR ? 'l' : 'r';
			const stanceHipPitch = stanceSide === 'l' ? hipPitchL : hipPitchR;
			const stanceKneeBend = stanceSide === 'l' ? kneeBendL : kneeBendR;
			const dropLocal = _legToeDropLocal(
				this._legBonesLocal[ stanceSide ], stanceHipPitch, stanceKneeBend,
			);
			// dropLocal*_localToM is the Hips->Toe drop in METERS, ABOVE GROUND (not
			// above root/hip -- `root.position` is itself already ground+
			// PATIENT_HIP_HEIGHT_M, per anim_bake.py's own patient_root convention,
			// so that fixed offset has to come back OUT here before re-adding the
			// dynamic drop, or the anchor ends up floating a further
			// PATIENT_HIP_HEIGHT_M too high -- confirmed by an initial numeric check
			// after first writing this fix: the toe landed flush with the HIP height
			// instead of the ground, off by almost exactly 0.92 m).
			groundClearanceAboveRootM = ( dropLocal * this._localToM ) - PATIENT_HIP_HEIGHT_M;

		}

		this.anchor.position.set(
			root.position.x, root.position.y,
			root.position.z - this._hipsHeightM + groundClearanceAboveRootM,
		);
		this.anchor.quaternion.copy( root.quaternion ).multiply( B_PLACEMENT );

		// 1) Pose the WHOLE skeleton from Xbot's own canned "walk" clip (arm swing,
		// spine sway) -- looped independently of the active phase clip's own (much
		// longer) duration.
		const walkDuration = this._walkAction.getClip().duration;
		this._walkAction.time = walkDuration > 0 ? ( timeSec % walkDuration ) : 0;
		this._mixer.update( 0 );

		const b = this._bones;

		// The canned clip also animates Hips' TRANSLATION (a vertical bob that dips
		// well below the bind-pose height _hipsHeightM was measured from -- as low
		// as ~6cm below at points in the cycle), Hips' ROTATION (a sway/tilt keyed to
		// the canned clip's OWN leg swing timing), and independently animates the
		// Foot/ToeBase ROTATION (ankle articulation tuned for the canned clip's OWN
		// hip/knee angles, not ours). Left alone, ALL of these put the feet through
		// the floor/tread or in the air: the hip bob sinks the whole leg chain, the
		// mismatched ankle rotation swings the toe tip by tens of centimeters (a
		// 20-30 degree error at the end of a ~0.44m shin is ~0.2m of travel), and --
		// this was the hardest one to catch, since it doesn't clip/float in a fixed
		// way but drifts -- the Hips ROTATION is the DIRECT PARENT transform of
		// LeftUpLeg/RightUpLeg, so leaving it un-reset silently re-rotates this
		// module's own carefully-computed leg angles by whatever the canned clip's
		// hip sway happens to be at that instant. Since the canned clip's own loop
		// period (~0.97s, this._walkAction's own clip duration) has NO relationship
		// to this pipeline's physically-computed gait period (~1.1-1.2s,
		// PATIENT_GAIT_FLAT/PATIENT_GAIT_CLIMB in synthetic_motion.py), the two drift
		// in and out of phase continuously -- confirmed by a live numeric sweep
		// (scrubbing both clips and measuring toe-height-above-ground and a
		// spine-up-vector lean angle every 10%): toe deviation above ground/tread
		// varied 0.01-0.24m (never negative, i.e. never actually planted) and the
		// spine lean angle varied 2.7-9.9 degrees with NO correlation to walk phase
		// -- exactly what an uncorrelated second rotation source riding on the same
		// bones looks like. Restore Hips to full bind pose (position AND rotation)
		// and hold the foot/toe at bind pose (~identity) as a SAFE DEFAULT so the
		// only thing steering the leg chain is this pipeline's own hip/knee angles
		// below -- Foot gets overridden again once pose data is available (see the
		// kneeBend compensation below); this identity assignment only matters as the
		// fallback for the `if ( ! pose ) return;` early-out a few lines down.
		b.hips.position.copy( this._bindHipsPosition );
		b.hips.quaternion.identity();
		b.leftFoot.quaternion.identity();
		b.rightFoot.quaternion.identity();
		b.leftToeBase.quaternion.identity();
		b.rightToeBase.quaternion.identity();

		// 2) Override the legs (+ set torso lean) from this pipeline's own
		// data-driven angles -- see this module's header comment for why the canned
		// clip can't be trusted with the stairs. (hipPitchL/R, kneeBendL/R already
		// sampled above, for the ground-clearance calculation -- reused here so
		// they're computed exactly once per frame.)
		if ( ! pose ) return;

		const torsoPitch = sampleAt( pose.times, pose.torso_pitch, timeSec );

		b.leftUpLeg.quaternion.setFromAxisAngle( PITCH_AXIS, hipPitchL );
		b.leftLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - kneeBendL );
		b.rightUpLeg.quaternion.setFromAxisAngle( PITCH_AXIS, hipPitchR );
		b.rightLeg.quaternion.setFromAxisAngle( PITCH_AXIS, - kneeBendR );
		// Foot LOCAL rotation = +kneeBend, cancelling Leg's own -kneeBend local
		// rotation (both about the SAME axis, PITCH_AXIS, so composition is exact
		// angle addition): Foot's WORLD rotation becomes hip_pitch alone -- i.e. the
		// foot tracks the THIGH's angle, not the (knee-bend-swung) shin's. Plain
		// identity (the original fix, see below) is only a good approximation while
		// knee_bend stays small -- at a MODERATE flat-ground bend (~25-50 degrees)
		// identity looks fine, but the stair-climbing gait swings the knee up to
		// ~80+ degrees to clear a riser (a real, correct angle -- see AGENTS.md
		// incident #6/#9), and at that bend identity rigidly points the foot wherever
		// the heavily-rotated SHIN points -- nearly straight up/back, reading as a
		// visibly twisted/broken ankle (user-reported: "the leg looks way too
		// wired" on stairs specifically, not on flat ground, exactly matching where
		// knee_bend actually gets large). This compensation is what a real ankle
		// does (dorsiflexing to keep the foot roughly aligned with the leg's own
		// swing direction as the knee folds to clear an obstacle), and it reduces to
		// the old ~identity behavior for small knee_bend, so flat-ground walking is
		// unaffected.
		b.leftFoot.quaternion.setFromAxisAngle( PITCH_AXIS, kneeBendL );
		b.rightFoot.quaternion.setFromAxisAngle( PITCH_AXIS, kneeBendR );
		// SET (not premultiply/add) Spine's rotation directly from this pipeline's
		// own torso_pitch, in Spine's parent (Hips-relative) frame -- same reasoning
		// as Hips above: premultiplying onto the canned clip's OWN spine-sway
		// quaternion stacked our real, physically-meaningful lean angle on top of an
		// uncorrelated cosmetic sway, which is what actually produced the reported
		// "leans too far forward" (the two would occasionally add constructively).
		// Spine is a sibling of the leg bones (both children of Hips), so this
		// doesn't affect foot placement -- only the torso's own visible lean, which
		// should be driven by torso_pitch alone.
		b.spine.quaternion.setFromAxisAngle( PITCH_AXIS, torsoPitch );

	}

}
