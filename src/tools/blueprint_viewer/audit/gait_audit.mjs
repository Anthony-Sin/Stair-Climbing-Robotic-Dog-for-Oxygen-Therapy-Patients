#!/usr/bin/env node
// gait_audit.mjs
//
// Node-tier headless replay harness for js/PatientGait.js (spec IK_OVERHAUL_SPEC.md
// section 8, "Node tier"). PatientGait.js is deliberately ZERO-IMPORT pure math (no
// THREE, no DOM -- see that file's own header), so it can be `import()`-ed directly
// in plain Node and driven through buildSchedule()/poseAt() exactly as the browser
// does, without a browser at all. This is the "don't trust screenshots" numeric
// verification layer: every metric below is a real computed number compared against
// a documented bar, not a visual judgment call.
//
// Usage (Windows PowerShell, from this directory or anywhere -- paths resolve
// relative to THIS file unless overridden):
//   node gait_audit.mjs
//   node gait_audit.mjs --module ../js/PatientGait.js --tracks out/tracks.json --out out/report_live.json --label live
//   node gait_audit.mjs --module out/PatientGait_baseline.mjs --out out/report_baseline.json --label baseline
//
// Runs TWO kinds of cases against whichever PatientGait module is loaded:
//   1. "follow" / "climb": the REAL recorded clips, if `--tracks` points at a
//      tracks.json produced by extract_tracks.py. Skipped (with a warning, not a
//      crash) if that file is missing -- this script must also work standalone.
//   2. "constant" / "stopgo" / "zigzag": synthetic unit-style fixtures built in
//      this file (flat terrain), always run regardless of --tracks, so the audit
//      never depends on the GLB/extraction pipeline having succeeded.
//
// Every metric is computed via the SAME reusable auditSchedule() function against
// whichever schedule/terrain it's handed -- there is exactly one implementation of
// each check, exercised against 5 different cases, not 5 copies that could drift.
//
// Forward-compatibility (GAIT/RIG agents are landing PatientGait.js v2 IN PARALLEL,
// per IK_OVERHAUL_SPEC.md section 3, while this file was written): this script
// feature-detects v2-only pose fields (phaseC, support, cane) with one probe
// poseAt() call per schedule and reports the metrics that need them as
// `pass: 'na(pending-gait-v2)'` instead of crashing when they're absent. Re-run
// this script once GAIT/RIG land to get real numbers for those rows.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const __dirname = dirname( fileURLToPath( import.meta.url ) );

// ===========================================================================
// CLI
// ===========================================================================

function printHelp() {

	console.log( `Usage: node gait_audit.mjs [--module <path>] [--tracks <path>] [--out <path>] [--label <name>]

  --module   Path to a PatientGait.js-shaped ES module to audit (default: ../js/PatientGait.js)
  --tracks   Path to a tracks.json produced by extract_tracks.py (default: out/tracks.json;
             if missing, "follow"/"climb" cases are skipped with a warning -- synthetic cases still run)
  --out      Report output path (default: out/report_<label>.json)
  --label    Free-text label stored in the report and used for the default --out name (default: live)
` );

}

function parseArgs( argv ) {

	const args = {
		module: resolve( __dirname, '../js/PatientGait.js' ),
		tracks: resolve( __dirname, 'out/tracks.json' ),
		out: null,
		label: 'live',
	};

	for ( let i = 0; i < argv.length; i ++ ) {

		const a = argv[ i ];
		if ( a === '--module' ) args.module = argv[ ++ i ];
		else if ( a === '--tracks' ) args.tracks = argv[ ++ i ];
		else if ( a === '--out' ) args.out = argv[ ++ i ];
		else if ( a === '--label' ) args.label = argv[ ++ i ];
		else if ( a === '--help' || a === '-h' ) { printHelp(); process.exit( 0 ); }
		else { console.error( `[gait_audit] unknown argument: ${ a }\n` ); printHelp(); process.exit( 1 ); }

	}

	args.module = resolve( args.module );
	args.tracks = resolve( args.tracks );
	if ( ! args.out ) args.out = resolve( __dirname, `out/report_${ args.label }.json` );
	else args.out = resolve( args.out );

	return args;

}

// ===========================================================================
// Synthetic fixtures (unit-style: no tracks.json needed)
//
// All on FLAT terrain (a stairSpec whose start_x_m is far beyond any synthetic
// path's range), 30 fps to match the real baked clips' own rate, samples shaped
// exactly like PatientGait.extractPathSamples's own output ({t,x,y,zRoot,yaw,
// groundRef}) so buildSchedule can't tell the difference from a real clip.
// ===========================================================================

const PATIENT_HIP_HEIGHT_M = 0.92; // kept in lockstep with PatientGait.js's own copy
const SYNTH_FPS = 30;

function buildFlatTerrain( gait ) {

	// start_x_m far away -> heightAt()/treadIndexAt() return 0 / -1 everywhere
	// a synthetic path below ever visits (max synthetic path length is a few
	// tens of metres at most).
	return gait.buildTerrain(
		{ start_x_m: 1e5, step_height_m: 0.145, step_depth_m: 0.305, step_count: 14 },
		1e5 + 14 * 0.305,
	);

}

/** Straight-line constant-velocity walk (default: the real follow clip's own
 *  recorded ~0.26 m/s). Exercises the base gait cleanly (no stops, no turns). */
function buildConstantVelocitySamples( { speed = 0.26, durationSec = 20, yaw = 0 } = {} ) {

	const dt = 1 / SYNTH_FPS;
	const n = Math.round( durationSec / dt ) + 1;
	const samples = new Array( n );
	const fx = Math.cos( yaw ), fy = Math.sin( yaw );

	for ( let i = 0; i < n; i ++ ) {

		const t = i * dt;
		const x = fx * speed * t, y = fy * speed * t;
		samples[ i ] = { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 };

	}

	return samples;

}

/** Alternating moving/fully-stopped segments -- the fixture this module's own
 *  header + AGENTS.md incident #6 exist to defend against ("moonwalking" through
 *  a stop). Exercises the idle gate (buildSchedule's rootIsIdle) directly: a
 *  correct scheduler must produce ZERO foot motion during the stopped spans. */
function buildStopAndGoSamples( { moveSpeed = 0.30, moveSec = 3, stopSec = 2, cycles = 5, yaw = 0 } = {} ) {

	const dt = 1 / SYNTH_FPS;
	const fx = Math.cos( yaw ), fy = Math.sin( yaw );
	const samples = [];
	let t = 0, x = 0, y = 0;
	samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } );

	for ( let c = 0; c < cycles; c ++ ) {

		const moveSteps = Math.round( moveSec / dt );
		for ( let i = 0; i < moveSteps; i ++ ) {

			t += dt; x += fx * moveSpeed * dt; y += fy * moveSpeed * dt;
			samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } );

		}

		const stopSteps = Math.round( stopSec / dt );
		for ( let i = 0; i < stopSteps; i ++ ) {

			t += dt;
			samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } );

		}

	}

	return samples;

}

/** Bonus fixture (spec section 8 mentions "zig-zag" alongside constant/stop-and-go;
 *  not explicitly required by this harness's own task brief, but cheap to add given
 *  the reusable auditSchedule() below, and it's the only fixture that exercises
 *  continuous yaw-driven step "need" / in-place-turn adjustment steps at all). A
 *  forward walk whose heading oscillates (a real S-curve path, not just a wobble in
 *  place), so lateral drift genuinely accumulates. */
function buildZigzagSamples( { speed = 0.25, durationSec = 24, yawAmplitude = 0.4, yawPeriodSec = 5 } = {} ) {

	const dt = 1 / SYNTH_FPS;
	const n = Math.round( durationSec / dt ) + 1;
	const samples = new Array( n );
	let x = 0, y = 0;

	for ( let i = 0; i < n; i ++ ) {

		const t = i * dt;
		const yaw = yawAmplitude * Math.sin( 2 * Math.PI * t / yawPeriodSec );
		if ( i > 0 ) {

			x += Math.cos( yaw ) * speed * dt;
			y += Math.sin( yaw ) * speed * dt;

		}
		samples[ i ] = { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 };

	}

	return samples;

}

// ===========================================================================
// Metric helpers
// ===========================================================================

function clamp( v, lo, hi ) { return Math.max( lo, Math.min( hi, v ) ); }

function median( arr ) {

	if ( ! arr.length ) return null;
	const s = [ ...arr ].sort( ( a, b ) => a - b );
	const mid = Math.floor( s.length / 2 );
	return s.length % 2 ? s[ mid ] : ( s[ mid - 1 ] + s[ mid ] ) / 2;

}

function metric( value, bar, pass ) { return { value, bar, pass }; }
function naMetric( bar, reason = 'na' ) { return { value: null, bar, pass: reason }; }

// ===========================================================================
// The metric suite: spec section 8 Node-tier M1-M7, M9 (cane), M10.
// Runs against ANY built {schedule, terrain} pair -- follow/climb (real data) or
// a synthetic fixture, v1 (baseline) or v2 (overhauled) PatientGait.js alike.
// ===========================================================================

function auditSchedule( gait, schedule, terrain, label ) {

	const { poseAt } = gait;
	const params = schedule.params;
	const samples = schedule.samples;
	const duration = samples[ samples.length - 1 ].t;

	// --- feature detection: v1 (pre-overhaul) vs v2 PatientGait.js -----------
	const probe = poseAt( schedule, terrain, samples[ 0 ].t );
	const hasPhaseC = typeof probe.phaseC === 'number';
	const hasSupport = typeof probe.support === 'number';
	const hasCaneField = Object.prototype.hasOwnProperty.call( probe, 'cane' );
	const hasFootTiming = !! ( probe.leftFoot && ( 'liftAt' in probe.leftFoot ) && ( 'nextLiftAt' in probe.leftFoot ) && ( 'strideLen' in probe.leftFoot ) );

	const allEvents = [
		...schedule.events.left.map( ( e ) => ( { ...e, foot: 'left' } ) ),
		...schedule.events.right.map( ( e ) => ( { ...e, foot: 'right' } ) ),
	].sort( ( a, b ) => a.tLift - b.tLift );

	const metrics = {};

	// --- M1: step length distribution -----------------------------------------
	const stepLengths = allEvents.map( ( e ) => Math.hypot( e.to.x - e.from.x, e.to.y - e.from.y ) );
	const m1med = median( stepLengths );
	metrics.M1_stepLengthMedian = stepLengths.length
		? metric( m1med, '[0.15, 0.45] m', m1med >= 0.15 && m1med <= 0.45 )
		: naMetric( '[0.15, 0.45] m', 'na(no steps)' );

	// --- M2: cadence + L/R strict-alternation ratio ----------------------------
	const cadenceStepsPerMin = duration > 0 ? allEvents.length / ( duration / 60 ) : 0;
	let alternating = 0;
	for ( let i = 1; i < allEvents.length; i ++ ) if ( allEvents[ i ].foot !== allEvents[ i - 1 ].foot ) alternating ++;
	const alternationRatio = allEvents.length > 1 ? alternating / ( allEvents.length - 1 ) : null;
	metrics.M2_cadenceStepsPerMin = metric( cadenceStepsPerMin, 'informational (no bar specified)', true );
	metrics.M2_alternationRatio = alternationRatio === null
		? naMetric( '>= 0.9', 'na(<2 steps)' )
		: metric( alternationRatio, '>= 0.9', alternationRatio >= 0.9 );

	// --- M3: duty factor per foot ------------------------------------------------
	const dutyFactorOf = ( foot ) => {

		const evs = schedule.events[ foot ];
		if ( ! evs.length || duration <= 0 ) return null;
		const swingTime = evs.reduce( ( sum, e ) => sum + Math.max( 0, Math.min( e.tLand, duration ) - Math.max( 0, e.tLift ) ), 0 );
		return clamp( 1 - swingTime / duration, 0, 1 );

	};
	const dutyL = dutyFactorOf( 'left' ), dutyR = dutyFactorOf( 'right' );
	const dutyOk = ( v ) => v !== null && v >= 0.55 && v <= 0.8;
	metrics.M3_dutyFactor = ( dutyL === null && dutyR === null )
		? naMetric( '[0.55, 0.8] per foot', 'na(no steps)' )
		: metric( { left: dutyL, right: dutyR }, '[0.55, 0.8] per foot', dutyOk( dutyL ) && dutyOk( dutyR ) );

	// --- 60 Hz sweep: M4, M5, M6, M7, M9, M10 ------------------------------------
	const HZ = 60;
	const dtSample = 1 / HZ;
	const nSamples = Math.max( 1, Math.round( duration / dtSample ) );
	const idleDtProbe = 0.02; // matches PatientGait.js's own internal _speedAt probe magnitude

	let maxPenetration = 0; // M4
	let nonIdleCount = 0, doubleSupportNonIdleCount = 0; // M5
	let maxRootTravelPlanted = 0, dsStartPos = null, wasDsNonIdle = false; // M6
	let maxPhaseCDelta = 0, phaseCMonotoneViol = 0, phaseCIdleViol = 0, prevPhaseC = null; // M7
	// M9 tip-range bar, CORRECTED 2026-07-10 (orchestrator + IK_OVERHAUL_SPEC.md section 8):
	// the original check measured 3D tip-to-hip distance against the physical cane
	// length (caneLengthM, ~0.90 m) -- geometrically impossible for a ground-planted
	// tip under a hip at z ~= PATIENT_HIP_HEIGHT_M (0.92 m), since even a tip directly
	// underfoot is >= hip height alone in 3D. Replaced with a HORIZONTAL (XY)
	// tip-to-root range bar -- a fixed audit constant (like M6's 0.20 m literal below),
	// not tied to the RIG's physical caneLengthM.
	const caneTipHorizontalRangeM = 0.55;
	let caneMaxPenetration = 0, caneIdleSamples = 0, canePlantedIdleSamples = 0, caneHorizontalRangeViolations = 0, caneSampleCount = 0; // M9
	let maxIdleFootMotion = 0; // M10
	let prevIdleLeft = null, prevIdleRight = null, prevIdleCane = null, wasIdle = false;

	for ( let i = 0; i <= nSamples; i ++ ) {

		const t = Math.min( duration, i * dtSample );
		const pose = poseAt( schedule, terrain, t );

		// Idle detection: pose.speed comes straight from the module's own
		// central-difference _speedAt (exposed on every pose); yaw rate isn't
		// exposed, so approximate it the same way (central difference of
		// rootYaw via two extra poseAt calls) -- this is NOT bit-identical to
		// buildSchedule's own internal idleSustainSamples-windowed gate (that
		// state is private to the build pass), just a close, honest proxy
		// good enough to bucket samples into "idle" vs "not" for M5/M6/M10.
		const ta = Math.max( 0, t - idleDtProbe ), tb = Math.min( duration, t + idleDtProbe );
		const yawRate = tb > ta ? Math.abs( poseAt( schedule, terrain, tb ).rootYaw - poseAt( schedule, terrain, ta ).rootYaw ) / ( tb - ta ) : 0;
		const idle = pose.speed < params.idleSpeedThreshold && yawRate < params.idleYawRateThreshold;

		// M4: foot penetration below terrain (positive = below/bad).
		const lh = terrain.heightAt( pose.leftFoot.x ), rh = terrain.heightAt( pose.rightFoot.x );
		maxPenetration = Math.max( maxPenetration, lh - pose.leftFoot.z, rh - pose.rightFoot.z );

		// M5 / M6: double support while non-idle.
		const bothPlanted = pose.leftFoot.planted && pose.rightFoot.planted;
		if ( ! idle ) {

			nonIdleCount ++;
			if ( bothPlanted ) doubleSupportNonIdleCount ++;

		}
		const dsNonIdleNow = ! idle && bothPlanted;
		if ( dsNonIdleNow && ! wasDsNonIdle ) dsStartPos = { x: pose.rootX, y: pose.rootY };
		if ( dsNonIdleNow && dsStartPos ) {

			maxRootTravelPlanted = Math.max( maxRootTravelPlanted, Math.hypot( pose.rootX - dsStartPos.x, pose.rootY - dsStartPos.y ) );

		}
		wasDsNonIdle = dsNonIdleNow;

		// M7: phaseC (v2 only).
		if ( hasPhaseC ) {

			if ( prevPhaseC !== null ) {

				const d = pose.phaseC - prevPhaseC;
				if ( d < - 1e-9 ) phaseCMonotoneViol ++;
				maxPhaseCDelta = Math.max( maxPhaseCDelta, Math.abs( d ) );
				if ( idle && Math.abs( d ) > 1e-9 ) phaseCIdleViol ++;

			}
			prevPhaseC = pose.phaseC;

		}

		// M9: cane (v2 only, and only while caneEnabled -- pose.cane is null otherwise).
		if ( hasCaneField && pose.cane ) {

			caneSampleCount ++;
			caneMaxPenetration = Math.max( caneMaxPenetration, terrain.heightAt( pose.cane.x ) - pose.cane.z );
			if ( idle ) {

				caneIdleSamples ++;
				if ( pose.cane.planted ) canePlantedIdleSamples ++;

			}
			// "recorded hip position" proxy: pose.root{X,Y} (root IS the hip
			// reference point per PATIENT_HIP_HEIGHT_M's own convention -- see
			// PatientGait.js's header). HORIZONTAL (XY) ONLY -- see
			// caneTipHorizontalRangeM's own comment above for why 3D was wrong.
			const horizontalReach = Math.hypot( pose.cane.x - pose.rootX, pose.cane.y - pose.rootY );
			if ( horizontalReach > caneTipHorizontalRangeM + 1e-6 ) caneHorizontalRangeViolations ++;

		}

		// M10: zero foot/cane motion between two CONSECUTIVE idle samples.
		if ( idle && wasIdle && prevIdleLeft ) {

			const mL = Math.hypot( pose.leftFoot.x - prevIdleLeft.x, pose.leftFoot.y - prevIdleLeft.y, pose.leftFoot.z - prevIdleLeft.z );
			const mR = Math.hypot( pose.rightFoot.x - prevIdleRight.x, pose.rightFoot.y - prevIdleRight.y, pose.rightFoot.z - prevIdleRight.z );
			maxIdleFootMotion = Math.max( maxIdleFootMotion, mL, mR );
			if ( hasCaneField && pose.cane && prevIdleCane ) {

				maxIdleFootMotion = Math.max( maxIdleFootMotion, Math.hypot( pose.cane.x - prevIdleCane.x, pose.cane.y - prevIdleCane.y, pose.cane.z - prevIdleCane.z ) );

			}

		}
		if ( idle ) {

			prevIdleLeft = { x: pose.leftFoot.x, y: pose.leftFoot.y, z: pose.leftFoot.z };
			prevIdleRight = { x: pose.rightFoot.x, y: pose.rightFoot.y, z: pose.rightFoot.z };
			prevIdleCane = ( hasCaneField && pose.cane ) ? { x: pose.cane.x, y: pose.cane.y, z: pose.cane.z } : null;

		} else {

			prevIdleLeft = prevIdleRight = prevIdleCane = null;

		}
		wasIdle = idle;

	}

	metrics.M4_footPenetrationMax = metric( maxPenetration, '<= 1e-6 m', maxPenetration <= 1e-6 );

	metrics.M5_doubleSupportFraction = nonIdleCount > 0
		? metric( doubleSupportNonIdleCount / nonIdleCount, '[0.2, 0.5]', ( doubleSupportNonIdleCount / nonIdleCount ) >= 0.2 && ( doubleSupportNonIdleCount / nonIdleCount ) <= 0.5 )
		: naMetric( '[0.2, 0.5]', 'na(no non-idle samples)' );

	metrics.M6_maxRootTravelWhilePlanted = metric( maxRootTravelPlanted, '<= 0.20 m', maxRootTravelPlanted <= 0.20 );

	if ( hasPhaseC ) {

		// Boundary sub-check, CORRECTED 2026-07-10 by the orchestrator: the spec's
		// original "phaseC == gaitPhase at event boundaries" requirement is
		// SELF-CONTRADICTORY against real data — the legacy gaitPhase staircase is
		// NOT monotone (measured on the real follow clip: 0.5, 0.0, 1.5, 1.0, 2.5,
		// 2.0... because _fillPhaseTimeline assigns left events integer phases by
		// PER-FOOT order, and this clip's RIGHT foot steps first), so no monotone
		// phaseC can equal it at every boundary. The corrected v2 contract: phaseC
		// is its OWN clean counter — for the k-th event (0-based) in merged
		// tLift-sorted order, phaseC(e.tLift) == 0.5*k and
		// phaseC(e.tLand) == 0.5*(k+1), ramping linearly in between and frozen
		// outside swings. Events whose window extends past the sampled duration
		// are skipped (counted) — the sweep can't evaluate them.
		let boundaryErrMax = 0, boundarySkipped = 0;
		for ( let k = 0; k < allEvents.length; k ++ ) {

			const e = allEvents[ k ];
			if ( e.tLift > duration ) { boundarySkipped ++; continue; }
			const atLift = poseAt( schedule, terrain, e.tLift );
			boundaryErrMax = Math.max( boundaryErrMax, Math.abs( atLift.phaseC - 0.5 * k ) );
			if ( e.tLand <= duration ) {

				const atLand = poseAt( schedule, terrain, e.tLand );
				boundaryErrMax = Math.max( boundaryErrMax, Math.abs( atLand.phaseC - 0.5 * ( k + 1 ) ) );

			} else boundarySkipped ++;

		}
		const m7pass = maxPhaseCDelta <= 0.04 && phaseCMonotoneViol === 0 && phaseCIdleViol === 0 && boundaryErrMax <= 1e-6;
		metrics.M7_phaseC = metric(
			{ maxDeltaPerSample: maxPhaseCDelta, monotoneViolations: phaseCMonotoneViol, idleViolations: phaseCIdleViol, boundaryErrMax, boundarySkipped },
			'monotone; maxDelta<=0.04; frozen(delta=0) at idle; phaseC(kth event lift)=0.5k, (land)=0.5(k+1)',
			m7pass,
		);

	} else {

		metrics.M7_phaseC = naMetric( 'monotone; maxDelta<=0.04; frozen(delta=0) at idle; phaseC(kth event lift)=0.5k, (land)=0.5(k+1)', 'na(pending-gait-v2)' );

	}

	if ( hasCaneField && caneSampleCount > 0 ) {

		const canePlantedIdleFraction = caneIdleSamples > 0 ? canePlantedIdleSamples / caneIdleSamples : null;

		// Timing correlation: each LEFT-foot event should be paired with a cane
		// event that fires at or before it (spec section 5: cane "advances WITH
		// (slightly leading) the CONTRALATERAL (LEFT) foot's swing").
		const caneEvents = schedule.caneEvents || [];
		let timingOk = true, timingSamples = 0, worstLeadSec = Infinity;
		for ( const le of schedule.events.left ) {

			let nearest = null, nearestD = Infinity;
			for ( const ce of caneEvents ) {

				const d = Math.abs( ce.tLift - le.tLift );
				if ( d < nearestD ) { nearestD = d; nearest = ce; }

			}
			if ( nearest ) {

				timingSamples ++;
				const lead = le.tLift - nearest.tLift; // >=0: cane fired at/before the paired left step (correct)
				worstLeadSec = Math.min( worstLeadSec, lead );
				if ( lead < - 1e-6 ) timingOk = false;

			}

		}

		metrics.M9_caneTipClearance = metric( caneMaxPenetration, '<= 1e-6 m', caneMaxPenetration <= 1e-6 );
		metrics.M9_canePlantedWhileIdle = canePlantedIdleFraction === null
			? naMetric( '== 1.0', 'na(no idle samples)' )
			: metric( canePlantedIdleFraction, '== 1.0', canePlantedIdleFraction >= 0.999 );
		metrics.M9_caneTipHorizontalRange = metric( caneHorizontalRangeViolations, `0 samples beyond ${ caneTipHorizontalRangeM.toFixed( 2 ) } m HORIZONTAL (XY) range of the root`, caneHorizontalRangeViolations === 0 );
		metrics.M9_caneLeadsLeftFoot = timingSamples > 0
			? metric( { worstLeadSec: worstLeadSec === Infinity ? null : worstLeadSec, pairedEvents: timingSamples }, 'cane tLift <= its paired left-foot tLift', timingOk )
			: naMetric( 'cane tLift <= its paired left-foot tLift', 'na(no left-foot events)' );

	} else {

		const naReason = hasCaneField ? 'na(cane disabled or zero cane events this case)' : 'na(pending-gait-v2)';
		metrics.M9_caneTipClearance = naMetric( '<= 1e-6 m', naReason );
		metrics.M9_canePlantedWhileIdle = naMetric( '== 1.0', naReason );
		metrics.M9_caneTipHorizontalRange = naMetric( '0 samples beyond 0.55 m HORIZONTAL (XY) range of the root', naReason );
		metrics.M9_caneLeadsLeftFoot = naMetric( 'cane tLift <= its paired left-foot tLift', naReason );

	}

	metrics.M10_idleMotionMax = metric( maxIdleFootMotion, '<= 0.002 m', maxIdleFootMotion <= 0.002 );

	const pass = Object.values( metrics ).every( ( m ) => m.pass === true || ( typeof m.pass === 'string' && m.pass.startsWith( 'na' ) ) );

	return {
		label,
		duration,
		sampleCount: samples.length,
		stepCount: allEvents.length,
		features: { hasPhaseC, hasSupport, hasCaneField, hasFootTiming },
		pass,
		metrics,
	};

}

// ===========================================================================
// Main
// ===========================================================================

function buildParams( gait ) {

	// Mirrors PatientHuman.buildGait() EXACTLY: DEFAULT_GAIT_PARAMS with
	// footLateral/toeForwardLen overridden by the real measured Xbot rig
	// values (see PatientHuman.js's constructor defaults for these two numbers
	// -- they're pre-load defaults there too, but happen to equal the actual
	// measured bind-pose geometry, per this task's own instruction).
	return { ...gait.DEFAULT_GAIT_PARAMS, footLateral: 0.082, toeForwardLen: 0.107 };

}

function loadTracks( path ) {

	try {

		return JSON.parse( readFileSync( path, 'utf-8' ) );

	} catch ( err ) {

		console.warn( `[gait_audit] could not read tracks file ${ path } (${ err.message }) -- "follow"/"climb" cases will be SKIPPED; synthetic cases still run.` );
		return null;

	}

}

function fmtValue( v ) {

	if ( v === null || v === undefined ) return 'null';
	if ( typeof v === 'number' ) return Number.isFinite( v ) ? v.toFixed( 4 ) : String( v );
	if ( typeof v === 'object' ) return JSON.stringify( v );
	return String( v );

}

function printSummaryTable( report ) {

	console.log( `\n=== gait_audit report: label="${ report.label }" module=${ report.modulePath } ===` );
	for ( const [ caseName, c ] of Object.entries( report.cases ) ) {

		console.log( `\n-- ${ caseName } -- duration=${ c.duration.toFixed( 2 ) }s steps=${ c.stepCount } pass=${ c.pass } features=${ JSON.stringify( c.features ) }` );
		for ( const [ name, m ] of Object.entries( c.metrics ) ) {

			const status = m.pass === true ? 'PASS' : ( m.pass === false ? 'FAIL' : String( m.pass ).toUpperCase() );
			console.log( `   ${ status.padEnd( 24 ) } ${ name.padEnd( 28 ) } value=${ fmtValue( m.value ) }  bar=${ m.bar }` );

		}

	}
	console.log( `\noverallPass: ${ report.overallPass }\n` );

}

async function main() {

	const args = parseArgs( process.argv.slice( 2 ) );

	console.log( `[gait_audit] importing module ${ args.module }` );
	const gait = await import( pathToFileURL( args.module ).href );
	for ( const need of [ 'DEFAULT_GAIT_PARAMS', 'buildTerrain', 'extractPathSamples', 'buildSchedule', 'poseAt' ] ) {

		if ( typeof gait[ need ] === 'undefined' ) {

			console.error( `[gait_audit] FATAL: ${ args.module } does not export "${ need }" -- is this really a PatientGait.js-shaped module?` );
			process.exit( 2 );

		}

	}

	const cases = {};

	const tracksData = loadTracks( args.tracks );
	if ( tracksData ) {

		const terrain = gait.buildTerrain( tracksData.stair_spec, tracksData.landing_far_x_m );
		for ( const clipName of [ 'follow', 'climb' ] ) {

			const clip = tracksData.clips && tracksData.clips[ clipName ];
			if ( ! clip ) { console.warn( `[gait_audit] tracks.json has no "${ clipName }" clip -- skipping` ); continue; }

			const samples = gait.extractPathSamples( clip.posTimes, clip.posValues, clip.quatTimes, clip.quatValues );
			const schedule = gait.buildSchedule( samples, terrain, buildParams( gait ) );
			cases[ clipName ] = auditSchedule( gait, schedule, terrain, clipName );

		}

	}

	const flatTerrain = buildFlatTerrain( gait );

	const constantSamples = buildConstantVelocitySamples( { speed: 0.26 } );
	cases.constant = auditSchedule( gait, gait.buildSchedule( constantSamples, flatTerrain, buildParams( gait ) ), flatTerrain, 'constant' );

	const stopgoSamples = buildStopAndGoSamples( {} );
	cases.stopgo = auditSchedule( gait, gait.buildSchedule( stopgoSamples, flatTerrain, buildParams( gait ) ), flatTerrain, 'stopgo' );

	const zigzagSamples = buildZigzagSamples( {} );
	cases.zigzag = auditSchedule( gait, gait.buildSchedule( zigzagSamples, flatTerrain, buildParams( gait ) ), flatTerrain, 'zigzag' );

	const overallPass = Object.values( cases ).every( ( c ) => c.pass );

	const report = {
		generatedAt: new Date().toISOString(),
		label: args.label,
		modulePath: args.module,
		tracksPath: tracksData ? args.tracks : null,
		tracksAvailable: !! tracksData,
		overallPass,
		cases,
	};

	mkdirSync( dirname( args.out ), { recursive: true } );
	writeFileSync( args.out, JSON.stringify( report, null, 2 ) + '\n', 'utf-8' );
	console.log( `[gait_audit] wrote ${ args.out }` );

	printSummaryTable( report );

	process.exitCode = overallPass ? 0 : 1;

}

main().catch( ( err ) => {

	console.error( '[gait_audit] FATAL:', err );
	process.exitCode = 2;

} );
