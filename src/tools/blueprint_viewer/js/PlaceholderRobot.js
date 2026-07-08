// PlaceholderRobot.js
//
// Builds an in-code placeholder "robot dog" (trunk + 4 two-segment legs +
// an oxygen tank box) plus two procedurally-keyframed AnimationClips named
// "follow" (a trot cycle) and "climb" (a stair-ish climb), so the full UI
// — phase buttons, scrubber, part labels, follow-cam — stays fully
// functional even before/without models/robot.glb.
//
// The returned root uses the SAME node-naming convention the real bake is
// expected to use (robot_base, FL_thigh, FR_calf, FR_foot, oxygen_tank,
// cradle_rails, head, patient_root) so PartLabels.js and the camera-fit
// logic exercise the identical code path for both the real model and the
// placeholder.

import {
	AnimationClip,
	BoxGeometry,
	Group,
	Mesh,
	QuaternionKeyframeTrack,
	Quaternion,
	Vector3,
	VectorKeyframeTrack,
} from 'three';

const LEG_LAYOUT = [
	{ name: 'FL', x: 0.18, z: 0.32 },
	{ name: 'FR', x: -0.18, z: 0.32 },
	{ name: 'RL', x: 0.18, z: -0.32 },
	{ name: 'RR', x: -0.18, z: -0.32 },
];

/**
 * Build the placeholder robot hierarchy.
 * @param {THREE.Material} material shared body material (already palette-colored)
 * @returns {{ root: THREE.Group, clips: THREE.AnimationClip[], legNodes: object, robotBase: THREE.Group }}
 */
export function buildPlaceholderRobot( material ) {

	const root = new Group();
	root.name = 'robot_placeholder_root';

	const robotBase = new Group();
	robotBase.name = 'robot_base';
	root.add( robotBase );

	// --- Trunk ---
	const trunkGeom = new BoxGeometry( 0.5, 0.18, 0.7 );
	const trunk = new Mesh( trunkGeom, material );
	trunk.name = 'trunk';
	trunk.position.set( 0, 0.55, 0 );
	robotBase.add( trunk );

	// --- Head / depth camera stub, front of trunk ---
	const head = new Group();
	head.name = 'head';
	head.position.set( 0, 0.58, 0.4 );
	robotBase.add( head );
	const headBox = new Mesh( new BoxGeometry( 0.12, 0.1, 0.12 ), material );
	head.add( headBox );

	// --- Oxygen tank, mounted on the trunk's back/top like a saddle payload ---
	const oxygenTank = new Group();
	oxygenTank.name = 'oxygen_tank';
	oxygenTank.position.set( 0, 0.75, -0.05 );
	robotBase.add( oxygenTank );
	const tankMesh = new Mesh( new BoxGeometry( 0.22, 0.22, 0.42 ), material );
	oxygenTank.add( tankMesh );

	// --- Cradle rails (patient cradle mount, alongside the tank) ---
	const cradleRails = new Group();
	cradleRails.name = 'cradle_rails';
	cradleRails.position.set( 0, 0.68, -0.32 );
	robotBase.add( cradleRails );
	const railGeom = new BoxGeometry( 0.46, 0.03, 0.03 );
	const railL = new Mesh( railGeom, material );
	railL.position.set( 0.18, 0, 0 );
	const railR = new Mesh( railGeom, material );
	railR.position.set( -0.18, 0, 0 );
	cradleRails.add( railL, railR );

	// --- Patient root: a simple reclined patient block on the cradle ---
	const patientRoot = new Group();
	patientRoot.name = 'patient_root';
	patientRoot.position.set( 0, 0.86, -0.32 );
	robotBase.add( patientRoot );
	const patientMesh = new Mesh( new BoxGeometry( 0.28, 0.16, 0.58 ), material );
	patientRoot.add( patientMesh );

	// --- Legs: hip pivot -> thigh (upper segment) -> calf pivot -> calf (lower) -> foot ---
	const legNodes = {};
	const thighGeom = new BoxGeometry( 0.08, 0.32, 0.08 );
	const calfGeom = new BoxGeometry( 0.06, 0.3, 0.06 );
	const footGeom = new BoxGeometry( 0.09, 0.05, 0.12 );

	for ( const leg of LEG_LAYOUT ) {

		const hip = new Group();
		hip.name = `${leg.name}_hip`;
		hip.position.set( leg.x, 0.55, leg.z );
		robotBase.add( hip );

		const thigh = new Group();
		thigh.name = `${leg.name}_thigh`;
		hip.add( thigh );
		const thighMesh = new Mesh( thighGeom, material );
		thighMesh.position.set( 0, -0.16, 0 );
		thigh.add( thighMesh );

		const calfPivot = new Group();
		calfPivot.name = `${leg.name}_calf_pivot`;
		calfPivot.position.set( 0, -0.32, 0 );
		thigh.add( calfPivot );

		const calf = new Group();
		calf.name = `${leg.name}_calf`;
		calfPivot.add( calf );
		const calfMesh = new Mesh( calfGeom, material );
		calfMesh.position.set( 0, -0.15, 0 );
		calf.add( calfMesh );

		const foot = new Group();
		foot.name = `${leg.name}_foot`;
		foot.position.set( 0, -0.3, 0 );
		calf.add( foot );
		const footMesh = new Mesh( footGeom, material );
		foot.add( footMesh );

		legNodes[ leg.name ] = { hip, thigh, calfPivot };

	}

	// --- Stairs prop (static, for the "climb" clip + stairs label) ---
	const stairs = new Group();
	stairs.name = 'stairs';
	stairs.position.set( 0, 0, -1.1 );
	root.add( stairs );
	const stepCount = 5;
	const stepRise = 0.13;
	const stepRun = 0.28;
	for ( let i = 0; i < stepCount; i ++ ) {

		const stepGeom = new BoxGeometry( 0.9, stepRise, stepRun );
		const step = new Mesh( stepGeom, material );
		step.position.set( 0, stepRise * ( i + 0.5 ), -stepRun * i - 0.2 );
		stairs.add( step );

	}

	const clips = [
		buildFollowClip( legNodes, robotBase ),
		buildClimbClip( legNodes, robotBase ),
	];

	return { root, clips, legNodes, robotBase };

}

// ---------------------------------------------------------------------
// Animation builders
// ---------------------------------------------------------------------

/**
 * A short trot cycle: opposite-corner leg pairs swing together, trunk bobs
 * slightly, whole robot_base translates forward along +Z.
 */
function buildFollowClip( legNodes, robotBase ) {

	const duration = 4.0;
	const tracks = [];

	// robot_base travels forward (+Z) over the clip, with a faint sinusoidal
	// vertical bob so the trot reads as locomotion, not a slide.
	const baseTimes = [ 0, 1, 2, 3, 4 ];
	const basePos = [];
	for ( const t of baseTimes ) {

		const frac = t / duration;
		basePos.push( 0, 0.55 + Math.sin( frac * Math.PI * 8 ) * 0.012, frac * 2.4 );

	}
	tracks.push( new VectorKeyframeTrack( 'robot_base.position', baseTimes, basePos ) );

	// Diagonal trot pairs: FL+RR swing together, FR+RL swing opposite phase.
	const swing = 0.5; // radians
	const lift = -0.35;

	const pairA = [ 'FL', 'RR' ];
	const pairB = [ 'FR', 'RL' ];

	const cycleTimes = [ 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 ];

	for ( const legName of pairA.concat( pairB ) ) {

		const phaseOffset = pairA.includes( legName ) ? 0 : Math.PI;
		const hip = legNodes[ legName ].hip;
		const calfPivot = legNodes[ legName ].calfPivot;

		const hipQuats = [];
		const calfQuats = [];

		for ( const t of cycleTimes ) {

			const phase = ( t / 1.0 ) * Math.PI * 2 + phaseOffset;
			const hipAngle = Math.sin( phase ) * swing * 0.5;
			const calfAngle = lift + Math.max( 0, Math.sin( phase ) ) * 0.6;

			const hq = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), hipAngle );
			hipQuats.push( hq.x, hq.y, hq.z, hq.w );

			const cq = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), calfAngle );
			calfQuats.push( cq.x, cq.y, cq.z, cq.w );

		}

		tracks.push( new QuaternionKeyframeTrack( `${hip.name}.quaternion`, cycleTimes, hipQuats ) );
		tracks.push( new QuaternionKeyframeTrack( `${calfPivot.name}.quaternion`, cycleTimes, calfQuats ) );

	}

	return new AnimationClip( 'follow', duration, tracks );

}

/**
 * A slower "climb" clip: robot_base rises in risers of stepRise while
 * translating -Z into the stairs prop, legs lift higher per step, and the
 * trunk pitches slightly nose-up on each riser (a deliberately different,
 * more careful gait than the trot).
 */
function buildClimbClip( legNodes, robotBase ) {

	const duration = 6.0;
	const stepRise = 0.13;
	const stepCount = 5;
	const tracks = [];

	const baseTimes = [];
	const basePos = [];
	const baseQuats = [];

	for ( let i = 0; i <= stepCount; i ++ ) {

		const t = ( i / stepCount ) * duration;
		baseTimes.push( t );
		basePos.push( 0, 0.55 + stepRise * i, -0.28 * i );

		const pitch = i < stepCount ? -0.12 : 0; // slight nose-up while climbing, level at top
		const bq = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), pitch );
		baseQuats.push( bq.x, bq.y, bq.z, bq.w );

	}

	tracks.push( new VectorKeyframeTrack( 'robot_base.position', baseTimes, basePos ) );
	tracks.push( new QuaternionKeyframeTrack( 'robot_base.quaternion', baseTimes, baseQuats ) );

	// Legs: higher, slower lifts than the trot, still diagonal pairs.
	const pairA = [ 'FL', 'RR' ];
	const pairB = [ 'FR', 'RL' ];
	const cycleTimes = [];
	for ( let i = 0; i <= stepCount * 2; i ++ ) cycleTimes.push( ( i / ( stepCount * 2 ) ) * duration );

	for ( const legName of pairA.concat( pairB ) ) {

		const phaseOffset = pairA.includes( legName ) ? 0 : Math.PI;
		const hip = legNodes[ legName ].hip;
		const calfPivot = legNodes[ legName ].calfPivot;

		const hipQuats = [];
		const calfQuats = [];

		for ( const t of cycleTimes ) {

			const phase = ( t / duration ) * stepCount * Math.PI * 2 + phaseOffset;
			const hipAngle = Math.sin( phase ) * 0.7 * 0.5;
			const calfAngle = -0.55 + Math.max( 0, Math.sin( phase ) ) * 0.95;

			const hq = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), hipAngle );
			hipQuats.push( hq.x, hq.y, hq.z, hq.w );

			const cq = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), calfAngle );
			calfQuats.push( cq.x, cq.y, cq.z, cq.w );

		}

		tracks.push( new QuaternionKeyframeTrack( `${hip.name}.quaternion`, cycleTimes, hipQuats ) );
		tracks.push( new QuaternionKeyframeTrack( `${calfPivot.name}.quaternion`, cycleTimes, calfQuats ) );

	}

	return new AnimationClip( 'climb', duration, tracks );

}
