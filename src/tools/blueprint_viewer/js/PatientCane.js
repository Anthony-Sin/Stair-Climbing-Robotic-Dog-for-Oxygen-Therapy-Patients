// PatientCane.js
//
// The patient's walking cane: a small, self-contained THREE prop (mesh build + a
// pure pose helper), owned by the RIG side of the IK overhaul (IK_OVERHAUL_SPEC.md
// S5/S6.7). PatientHuman.js builds one of these at load time and attaches it as a
// SIBLING of its own anchor under isaacWorldNode (never parented to the hand -- the
// HAND is IK'd to the CANE, so the cane stays terrain-exact regardless of arm reach;
// see PatientHuman's cane-unreachable fallback). This file itself never touches a
// bone, never imports PatientHuman.js (no circular dependency -- PatientHuman is the
// only caller), and does no DOM work (matches spec S2's "pure THREE, no DOM").
//
// Coordinate convention: exactly like PatientHuman's anchor, this Group is a direct
// child of isaacWorldNode, so its .position/.quaternion are already in the P-frame
// (isaac_world's own local frame: X forward, Y lateral, Z up -- see PatientGait.js's
// header). Nothing here needs PatientHuman's B_PLACEMENT reconciliation at all --
// that constant exists ONLY because Xbot's imported skeleton ships in glTF/Mixamo's
// own axis convention; this module builds its OWN mesh from scratch, so it is free to
// pick its build convention to match the P-frame directly (see _LOCAL_UP below).

import * as THREE from 'three';

// ===========================================================================
// Tunables (spec S5/S9: named, unit-suffixed, no magic numbers inline)
// ===========================================================================

export const CANE_PARAMS = {
	caneLengthM: 0.90, // m, tip-to-handle overall length (spec S5: handle sits near greater-trochanter/hip height ~0.92 m)
	caneHandForwardM: 0.05, // m, how far ahead of the tip the handle sits while PLANTED -- consumed analytically (see _plantedLeanRad), not as an extra geometric offset
	caneSwingLeanMaxRad: THREE.MathUtils.degToRad( 18 ), // spec S5 "lean forward up to ~18 deg" mid-swing (pendulum peak)
	shaftRadiusM: 0.011, // spec S5 "~11 mm radius"
	handleRadiusM: 0.022, // knob half-width; see buildCaneGroup's own comment for why an axisymmetric knob replaces a literal offset-T/derby silhouette
	handleLengthM: 0.05, // knob's own long-axis extent (capsule "length" param, excludes the two end caps)
	tipRadiusM: 0.02, // spec S5 "~2 cm" rubber ferrule
	tipLengthM: 0.03,
};

// Colors matched to the RETIRED decorative cane (main.js, pre-overhaul lines
// ~1229-1257, deleted by the VERIFY agent as part of this same overhaul) for visual
// continuity -- 0x6b4a32 was that cane's own wood tone; 0x232629 is main.js's
// `robotBlackMaterial` color (the linear-black baked into the robot's own COLOR_0
// meshes), reused here for the rubber tip so it reads as the same "near-black rubber"
// elsewhere in the scene.
const _WOOD_COLOR = 0x6b4a32;
const _TIP_COLOR = 0x232629;

/** Small NearestFilter 1D gradient texture for MeshToonMaterial's `gradientMap` --
 *  the SAME technique main.js's own `makeToonGradientMap`/`makeBlueprintMaterial`
 *  use (banded toon shading is this viewer's whole-scene visual language), but
 *  duplicated locally in a few lines rather than imported: RIG owns this file and
 *  cannot edit/import from main.js (VERIFY owns it), and this module is meant to be
 *  fully self-contained (buildable/postable with nothing but `tip`+`handle` points).
 *  This intentionally does NOT replicate makeBlueprintMaterial's onBeforeCompile
 *  fresnel-rim injection -- that's a scene-wide polish detail, not load-bearing for
 *  "the cane reads as a toon-shaded prop consistent with the rest of the set". */
function _makeToonGradientMap( levels ) {

	const data = new Uint8Array( levels.length );
	for ( let i = 0; i < levels.length; i ++ ) data[ i ] = Math.round( THREE.MathUtils.clamp( levels[ i ], 0, 1 ) * 255 );

	const texture = new THREE.DataTexture( data, levels.length, 1, THREE.RedFormat );
	texture.minFilter = THREE.NearestFilter;
	texture.magFilter = THREE.NearestFilter;
	texture.wrapS = THREE.ClampToEdgeWrapping;
	texture.wrapT = THREE.ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;
	return texture;

}

// Close to (not identical to) main.js's own CEL_GRADIENT_MAP ([0.30, 0.60, 1.0]) --
// same 3-band toon style, independently owned so this file has zero import coupling
// to main.js's module-level state.
const _CANE_GRADIENT_MAP = _makeToonGradientMap( [ 0.35, 0.65, 1.0 ] );

// This Group's own BUILD axis: local +Y runs tip -> handle (a plain THREE.Cylinder's
// natural axis, matching the retired cane's own Y-up construction style). Posing the
// group is then just "aim local +Y at the desired world direction" (applyCanePose).
const _LOCAL_UP = new THREE.Vector3( 0, 1, 0 );

/** Shortest-arc quaternion rotating unit vector `a` onto unit vector `b`. Standard
 *  cross/dot construction (antiparallel case picks an arbitrary perpendicular axis
 *  for the 180deg rotation -- never hit in practice here, the cane axis never points
 *  exactly opposite its own build axis). Deliberately a SEPARATE small copy of
 *  PatientHuman.js's own `_quatFromTo` (same few lines) rather than a cross-file
 *  import -- keeps this module's only dependency as `three` itself, matching its own
 *  "pure THREE, no DOM, no PatientHuman coupling" header contract. */
function _quatFromTo( a, b, out = new THREE.Quaternion() ) {

	const d = a.dot( b );

	if ( d > 1 - 1e-9 ) return out.identity();

	if ( d < - 1 + 1e-9 ) {

		let perp = new THREE.Vector3().crossVectors( a, new THREE.Vector3( 1, 0, 0 ) );
		if ( perp.lengthSq() < 1e-6 ) perp = new THREE.Vector3().crossVectors( a, new THREE.Vector3( 0, 1, 0 ) );
		perp.normalize();
		return out.setFromAxisAngle( perp, Math.PI );

	}

	const axis = new THREE.Vector3().crossVectors( a, b );
	return out.set( axis.x, axis.y, axis.z, 1 + d ).normalize();

}

/**
 * Build the cane's THREE.Group (shaft + handle knob + rubber tip), local +Y = tip->
 * handle (see _LOCAL_UP). Self-contained materials (see _WOOD_COLOR/_TIP_COLOR/
 * _CANE_GRADIENT_MAP above) -- no material is accepted as a parameter because
 * PatientHuman.attachTo()'s existing signature (`attachTo(isaacWorldNode,
 * patientRootNode, tintMaterial)`) cannot change (main.js calls it positionally and
 * RIG cannot edit main.js to pass a 4th, cane-specific material) and `tintMaterial`
 * itself is unsuitable here regardless (it's a white MeshToonMaterial with
 * `vertexColors:true`, built for the patient's own per-vertex-painted skin/clothes --
 * see PatientHuman.js's paintPatientRegionColors -- not a plain-colored rigid prop).
 *
 * Handle shape: a real cane's "derby"/offset-T handle is NOT rotationally symmetric
 * about the shaft axis, but this module poses the whole Group with a plain aim-only
 * shortest-arc rotation (applyCanePose/_quatFromTo) that leaves ROLL (twist about the
 * tip->handle axis) unconstrained/arbitrary -- an asymmetric handle would therefore
 * appear to spin/twist unpredictably as the cane sways through its planted-lean ->
 * swing-lean range, with no numeric bar and no browser/visual verification available
 * in RIG's pass to tune a roll reference against. An axisymmetric knob (capsule,
 * aligned with the shaft) sidesteps the problem entirely -- correct-looking from
 * every roll angle -- at the cost of a less literally "derby" silhouette.
 */
export function buildCaneGroup( params = CANE_PARAMS ) {

	const group = new THREE.Group();
	group.name = 'patient_cane';

	const woodMat = new THREE.MeshToonMaterial( { color: _WOOD_COLOR, gradientMap: _CANE_GRADIENT_MAP } );
	const tipMat = new THREE.MeshToonMaterial( { color: _TIP_COLOR, gradientMap: _CANE_GRADIENT_MAP } );

	const tipTopY = params.tipLengthM * 0.85; // tip mesh sinks slightly below y=0 (ground-contact ferrule) -- see tip.position below
	const handleBottomY = params.caneLengthM - params.handleRadiusM * 0.6;
	const shaftLenM = Math.max( 0.05, handleBottomY - tipTopY );

	const shaft = new THREE.Mesh(
		new THREE.CylinderGeometry( params.shaftRadiusM, params.shaftRadiusM * 1.15, shaftLenM, 10 ),
		woodMat,
	);
	shaft.position.y = tipTopY + shaftLenM / 2;
	group.add( shaft );

	const handle = new THREE.Mesh(
		new THREE.CapsuleGeometry( params.handleRadiusM, params.handleLengthM, 6, 12 ),
		woodMat,
	);
	handle.position.y = params.caneLengthM;
	group.add( handle );

	const tip = new THREE.Mesh(
		new THREE.CylinderGeometry( params.tipRadiusM * 0.7, params.tipRadiusM, params.tipLengthM, 8 ),
		tipMat,
	);
	// Centered slightly below y=0 so the visible ferrule's BOTTOM face sits at the
	// cane's own y=0 (the posed group's origin, which applyCanePose plants exactly at
	// pose.cane's terrain-exact tip point) -- a cylinder mesh is centered on its own
	// midpoint by construction, so this offset keeps the true ground-contact point at
	// the group origin rather than at the tip mesh's vertical center.
	tip.position.y = params.tipLengthM * 0.5 - params.tipLengthM * 0.15;
	group.add( tip );

	group.traverse( ( o ) => { if ( o.isMesh ) o.castShadow = true; } );

	return group;

}

/** Planted lean ANGLE (radians, from the group's local +Y "straight up the shaft"),
 *  derived analytically from caneHandForwardM/caneLengthM so that
 *  `handle = tip + axis*caneLengthM` places the handle EXACTLY caneHandForwardM ahead
 *  of the tip at rest (spec S5: "handle sits caneHandForwardM (0.05) ahead of tip") --
 *  not a hardcoded angle/slope constant (spec S9: no magic numbers). At the spec
 *  defaults (0.05 / 0.90) this is ~3.2 deg, close to (not identical to -- see this
 *  module's own header) the spec's illustrative axis vector "(0.06, 0, 1) normalized"
 *  (~3.4 deg); deriving it from the two NAMED tunables instead of a third magic
 *  "0.06" was judged the better trade (spec S9 "No magic numbers inline"). */
function _plantedLeanRad( params ) {

	const l = params.caneLengthM;
	const f = Math.min( params.caneHandForwardM, l - 1e-6 );
	return Math.atan2( f, Math.sqrt( Math.max( 1e-9, l * l - f * f ) ) );

}

/** Smoothstep, 0 at u=0, 1 at u=1, zero slope at both ends (same primitive
 *  PatientGait.js's own `_smoothstep` uses -- duplicated here rather than imported,
 *  see this file's "zero PatientHuman/PatientGait coupling" header note). */
function _smoothstep( u ) { return u * u * ( 3.0 - 2.0 * u ); }

/** 3-point profile: exactly `startVal` at u=0, `midVal` at u=0.5, `endVal` at u=1,
 *  each half an independent smoothstep glued at u=0.5 -- zero slope at ALL three
 *  control points (in particular at u=0.5, from BOTH sides), so this is C1-continuous
 *  everywhere, not just C0. Used to animate the cane's swing lean between the planted
 *  angle (matching on both ends -- no pop at liftoff/touchdown) and the mid-swing
 *  peak lean. */
function _smoothstep3( u, startVal, midVal, endVal ) {

	if ( u <= 0.5 ) return startVal + ( midVal - startVal ) * _smoothstep( u / 0.5 );
	return midVal + ( endVal - midVal ) * _smoothstep( ( u - 0.5 ) / 0.5 );

}

/**
 * Pure pose computation (NO Object3D writes -- see applyCanePose for that). Given
 * PatientGait v2's `pose.cane` (poseAt's return, `{x,y,z,planted,swingU,...}`; caller
 * must not invoke this with a null/undefined pose -- see PatientHuman's own
 * caneEnabled===false / integration-race handling, spec I10) and the CURRENT sync's
 * `rootQuat` (the SAME tail-resolved quaternion PatientHuman's anchor itself uses),
 * returns `{tip, axisWorld, handle, leanRad}` -- all in P-frame (isaac_world local)
 * meters, the frame this Group's .position/.quaternion already live in (see this
 * file's header) so the result can be handed straight to applyCanePose with zero
 * further conversion.
 *
 * `rootQuat` reconciliation: patient_root's baked rotation is yaw-only (see
 * PatientGait.js's header/extractPathSamples's own contract), so applying it to a
 * "facing-frame" vector built from (forward-component, 0 lateral, up-component) IS
 * exactly the P-frame conversion (a pure-Z rotation leaves the up/Z component
 * untouched and mixes forward/lateral by the current yaw) -- no separate yaw
 * extraction/trig needed, and no cross-file B_PLACEMENT-style basis change either
 * (this module never touches Xbot's own local convention at all -- see header).
 */
export function computeCanePose( pose, rootQuat, params = CANE_PARAMS, out = { tip: new THREE.Vector3(), axisWorld: new THREE.Vector3(), handle: new THREE.Vector3(), leanRad: 0 } ) {

	out.tip.set( pose.x, pose.y, pose.z );

	const plantedLean = _plantedLeanRad( params );
	let lean = plantedLean;
	if ( ! pose.planted && pose.swingU !== null && pose.swingU !== undefined ) {

		// Pendulum-forward during swing (spec S5/S6.7): peaks at caneSwingLeanMaxRad
		// near mid-swing, EXACTLY matching the planted lean at both u=0 (just lifted)
		// and u=1 (about to plant) -- no pop against the planted formula above at
		// either boundary.
		lean = _smoothstep3( pose.swingU, plantedLean, params.caneSwingLeanMaxRad, plantedLean );

	}
	out.leanRad = lean;

	// Facing-frame axis (forward-component, 0 lateral, up-component), rotated into
	// P-frame by the current yaw -- see this function's own doc comment above.
	out.axisWorld.set( Math.sin( lean ), 0, Math.cos( lean ) ).applyQuaternion( rootQuat );

	out.handle.copy( out.tip ).addScaledVector( out.axisWorld, params.caneLengthM );

	return out;

}

/**
 * Write `group`'s position/quaternion so it sits with its tip at `tip` and its local
 * +Y axis (see _LOCAL_UP) aimed at `handle` (direction only -- magnitude of
 * `handle-tip` is NOT assumed to equal caneLengthM here, so a caller may pass an
 * arm-reach-adjusted handle point, per spec S5's "If the arm IK can't reach ... the
 * cane tilts toward the hand rather than the arm hyper-extending": PatientHuman
 * computes the nominal handle via computeCanePose, solves the right-arm two-bone IK
 * against it, and on `reachClamped` recomputes an ADJUSTED handle -- still exactly
 * caneLengthM from `tip`, just re-aimed toward the achieved hand direction -- before
 * calling this). Degenerate (tip==handle) safely falls back to identity rather than
 * a NaN quaternion.
 */
export function applyCanePose( group, tip, handle ) {

	group.position.copy( tip );

	const axis = handle.clone().sub( tip );
	if ( axis.lengthSq() < 1e-10 ) { group.quaternion.identity(); return; }
	axis.normalize();

	group.quaternion.copy( _quatFromTo( _LOCAL_UP, axis ) );

}
