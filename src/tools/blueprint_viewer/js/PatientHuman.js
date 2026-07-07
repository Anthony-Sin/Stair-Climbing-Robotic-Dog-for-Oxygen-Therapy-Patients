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

// Kept in sync with anim_bake.PATIENT_HIP_HEIGHT_M -- NOT used directly (see
// _hipsHeightM below, which measures Xbot's OWN Hips-bone height instead of assuming
// it matches this pipeline's mannequin height), but documents the quantity this
// module is reconciling against.
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
			// bind pose (feet resting on Xbot's own y=0 ground) and raise the whole
			// rig by that amount so the TOE, not the ankle, ends up at ground/tread.
			const footWorld = new THREE.Vector3();
			bones.leftFoot.getWorldPosition( footWorld );
			const toeWorld = new THREE.Vector3();
			bones.leftToeBase.getWorldPosition( toeWorld );
			this._ankleGroundClearanceM = footWorld.y - toeWorld.y;

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
		this.anchor.position.set(
			root.position.x, root.position.y,
			root.position.z - this._hipsHeightM + this._ankleGroundClearanceM,
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
		// clip can't be trusted with the stairs.
		const pose = this._poseData[ phaseName ];
		if ( ! pose ) return;

		const hipPitchL = sampleAt( pose.times, pose.hip_pitch_l, timeSec );
		const kneeBendL = sampleAt( pose.times, pose.knee_bend_l, timeSec );
		const hipPitchR = sampleAt( pose.times, pose.hip_pitch_r, timeSec );
		const kneeBendR = sampleAt( pose.times, pose.knee_bend_r, timeSec );
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
