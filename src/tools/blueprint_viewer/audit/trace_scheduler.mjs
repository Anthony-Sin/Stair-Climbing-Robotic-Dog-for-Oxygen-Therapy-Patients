#!/usr/bin/env node
// trace_scheduler.mjs
//
// SCHED-ANALYST round-2 tool (IK_OVERHAUL_SPEC.md, "round 2: pinpoint numerically WHAT
// is unnatural" -- this is the raw-data half of that pass; audit/analyze_scheduler.py
// is the metrics half). Unlike gait_audit.mjs (which reduces a sweep straight down to
// pass/fail metrics), this script keeps the FULL per-frame time series: every poseAt()
// sample at 60 Hz, for the two REAL recorded clips plus a parametric synthetic fixture
// bench (constant-speed sweep, constant-curvature arcs, stop-and-go, speed ramp), so
// analyze_scheduler.py can compute curve-shape metrics (cadence-vs-speed, turning
// asymmetry, swing vertical-profile shape, FFT/autocorrelation, ...) that a single
// aggregate number can't capture. Reuses gait_audit.mjs's idioms (module loading via
// dynamic import() of a PatientGait.js-shaped ES module, buildParams' footLateral/
// toeForwardLen override, buildFlatTerrain's far-away stairSpec trick, the same 30 fps
// synthetic-sample cadence as the real baked clips) -- see that file + audit/README.md
// for the underlying conventions this one extends rather than forks. gait_audit.mjs
// itself is NOT modified by this script (its own fixture generators are unexported
// plain functions, not importable, so the synthetic generators below are written fresh
// in the same shape/style rather than reusing its unexported code directly).
//
// Usage (Windows PowerShell, from this directory or anywhere -- paths resolve relative
// to THIS file unless overridden):
//   node trace_scheduler.mjs
//   node trace_scheduler.mjs --module ../js/PatientGait.js --tracks out/tracks.json --outDir out --hz 60
//   node trace_scheduler.mjs --only follow,climb,arc_r1.50   (comma-separated case names, for a fast partial re-run)
//
// ===========================================================================
// Output schema (one file per case: audit/out/strace_<case>.json)
// ===========================================================================
//
// {
//   "case": "follow",                    // case name (also the file's <case> suffix)
//   "kind": "real" | "synthetic",
//   "meta": { ... },                     // case-specific parameters (speed, radius, etc -- see each builder)
//   "generatedAt": "<ISO8601>",
//   "hz": 60,                            // sweep rate poseAt() was queried at
//   "durationSec": 23.6,                 // schedule's own last-sample time (== samples[-1].t)
//   "sampleCount": 1417,                 // series array length (== round(durationSec*hz)+1)
//   "params": { ...schedule.params },    // EFFECTIVE (already-merged, stanceWidenM-applied) DEFAULT_GAIT_PARAMS
//                                        // used to build this schedule -- cite these keys in fix-lever recommendations
//   "events": {                          // RAW scheduled footfall/cane events (buildSchedule's own event objects,
//                                        // numbers rounded to 1e-6) -- ground truth for step timing, independent
//                                        // of the 60 Hz series below (useful for exact step-count spot-checks)
//     "left":  [ { tLift, tLand, from:{x,y,z}, to:{x,y,z}, fromYaw, toYaw }, ... ],
//     "right": [ ... same shape ... ],
//     "cane":  [ ... same shape ... ] | null   // null when params.caneEnabled is false for this schedule
//   },
//   "tail": { startT, freezeT, rootStart:{x,y,zRoot,yaw}, rootEnd:{x,y,zRoot,yaw} }, // buildSchedule's walk-on descriptor
//   "series": {                          // COLUMNAR (parallel arrays, all the same length == sampleCount) --
//                                        // NOT array-of-objects, to keep file size sane per the task brief.
//                                        // Numbers rounded to 1e-6 (position/z, radians) or 1e-5 (time) on write.
//     "t":         [...],                // sample time (s), min(duration, i/hz)
//     "rootX":     [...], "rootY": [...], "rootZ": [...], "rootYaw": [...],
//     "speed":     [...],                // poseAt's own central-difference root speed (m/s)
//     "groundSlope":[...],
//     "gaitPhase": [...],                // legacy staircase phase (kept for cross-reference; non-monotone by design)
//     "phaseC":    [...],                // v2 continuous phase (monotone within a schedule, frozen at idle)
//     "support":   [...],                // v2 lateral weight signal in [-1, +1]
//     "leftFoot":  { x,y,z,yaw:[...], planted:[0|1...], swingU:[num|null...],
//                    liftAt:[num|null...], landedAt:[num|null...], nextLiftAt:[num|null...], strideLen:[...] },
//     "rightFoot": { ... same shape as leftFoot ... },
//     "cane":      { x,y,z:[...], planted:[0|1...], swingU:[num|null...],
//                    liftAt:[num|null...], landedAt:[num|null...], nextLiftAt:[num|null...] } | absent (key omitted)
//                    when this schedule's poseAt().cane is null (caneEnabled=false) -- check with 'cane' in series.
//   }
// }
//
// A companion audit/out/strace_index.json lists every case written this run (name,
// file, kind, meta, durationSec, stepCount) so analyze_scheduler.py (or a human) can
// enumerate cases without hardcoding the list or re-deriving it from a directory scan.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const __dirname = dirname( fileURLToPath( import.meta.url ) );

// ===========================================================================
// CLI
// ===========================================================================

function printHelp() {

	console.log( `Usage: node trace_scheduler.mjs [--module <path>] [--tracks <path>] [--outDir <path>] [--hz <n>] [--only <names>]

  --module   Path to a PatientGait.js-shaped ES module to trace (default: ../js/PatientGait.js)
  --tracks   Path to a tracks.json produced by extract_tracks.py (default: out/tracks.json;
             if missing, "follow"/"climb" cases are SKIPPED with a warning -- synthetic cases still run)
  --outDir   Directory to write strace_<case>.json + strace_index.json into (default: out)
  --hz       Sweep rate for poseAt() queries (default: 60, matches gait_audit.mjs's own 60 Hz sweep)
  --only     Comma-separated case names -- only build/trace/write these (default: all cases)
` );

}

function parseArgs( argv ) {

	const args = {
		module: resolve( __dirname, '../js/PatientGait.js' ),
		tracks: resolve( __dirname, 'out/tracks.json' ),
		outDir: resolve( __dirname, 'out' ),
		hz: 60,
		only: null,
	};

	for ( let i = 0; i < argv.length; i ++ ) {

		const a = argv[ i ];
		if ( a === '--module' ) args.module = argv[ ++ i ];
		else if ( a === '--tracks' ) args.tracks = argv[ ++ i ];
		else if ( a === '--outDir' ) args.outDir = argv[ ++ i ];
		else if ( a === '--hz' ) args.hz = Number( argv[ ++ i ] );
		else if ( a === '--only' ) args.only = argv[ ++ i ].split( ',' ).map( ( s ) => s.trim() ).filter( Boolean );
		else if ( a === '--help' || a === '-h' ) { printHelp(); process.exit( 0 ); }
		else { console.error( `[trace_scheduler] unknown argument: ${ a }\n` ); printHelp(); process.exit( 1 ); }

	}

	args.module = resolve( args.module );
	args.tracks = resolve( args.tracks );
	args.outDir = resolve( args.outDir );

	return args;

}

// ===========================================================================
// Synthetic fixture generators (same {t,x,y,zRoot,yaw,groundRef} sample shape as
// PatientGait.extractPathSamples' own output / gait_audit.mjs's fixtures -- see this
// file's header for why these are written fresh rather than imported from there).
// 30 fps to match the real baked clips' own rate (gait_audit.mjs's SYNTH_FPS) --
// buildSchedule's predictive-trigger/touchdown-resolution logic "walks the sample
// array forward" (see PatientGait.js's own buildSchedule docstring), so the sample
// DENSITY matters, not just the path shape.
// ===========================================================================

const PATIENT_HIP_HEIGHT_M = 0.92; // kept in lockstep with PatientGait.js's own copy
const SYNTH_FPS = 30;

/** Straight-line constant-velocity walk. */
function buildConstSpeedSamples( { speed, durationSec, yaw = 0 } ) {

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

/**
 * Constant-curvature arc: closed-form circular path (no numerical integration, so no
 * accumulated drift over a long sweep) -- yaw(t) = omega*t, omega = turnSign*speed/radius,
 * x(t) = (speed/omega)*sin(omega t), y(t) = (speed/omega)*(1-cos(omega t)). This traces a
 * circle of exactly `radius` metres, turning LEFT (CCW, +yaw) for turnSign=+1, matching
 * this module's own +Y=left convention (PatientGait.js's _nominalAt/_caneTargetAt
 * comments). Over a long-enough durationSec the tighter radii wrap around more than one
 * full revolution (e.g. r=0.75 m at 0.26 m/s over 24 s covers ~478 degrees) -- this is
 * fine/deliberate: curvature stays constant regardless of total revolutions, and more
 * steps under identical steady-state turning conditions is MORE data for the turning-
 * asymmetry metrics (per-step yaw delta, inside/outside step-length split), not less.
 */
function buildArcSamples( { speed, radius, durationSec, turnSign = 1 } ) {

	const dt = 1 / SYNTH_FPS;
	const n = Math.round( durationSec / dt ) + 1;
	const samples = new Array( n );
	const omega = turnSign * speed / radius;

	for ( let i = 0; i < n; i ++ ) {

		const t = i * dt;
		const yaw = omega * t;
		let x, y;
		if ( Math.abs( omega ) > 1e-9 ) {

			x = ( speed / omega ) * Math.sin( omega * t );
			y = ( speed / omega ) * ( 1 - Math.cos( omega * t ) );

		} else {

			x = speed * t; y = 0;

		}

		samples[ i ] = { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 };

	}

	return samples;

}

/**
 * Single stop-go cycle with a HARD (instantaneous, not decelerating) stop: walk at
 * moveSpeed for walkSec, freeze for stopSec (root literally stationary sample-to-
 * sample), resume at the SAME moveSpeed for walkSec more. Deliberately a single cycle
 * (unlike gait_audit.mjs's own buildStopAndGoSamples, which repeats 5 short cycles at a
 * different speed) so the pre-stop/post-stop step sequences are long enough to isolate
 * "how many steps does deceleration/re-acceleration actually take" cleanly, per the
 * task brief's stop/start-dynamics metric.
 */
function buildStopGoHardSamples( { moveSpeed, walkSec, stopSec, yaw = 0 } ) {

	const dt = 1 / SYNTH_FPS;
	const fx = Math.cos( yaw ), fy = Math.sin( yaw );
	const samples = [];
	let t = 0, x = 0, y = 0;
	samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } );

	const walkSteps = Math.round( walkSec / dt );
	for ( let i = 0; i < walkSteps; i ++ ) { t += dt; x += fx * moveSpeed * dt; y += fy * moveSpeed * dt; samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } ); }

	const stopSteps = Math.round( stopSec / dt );
	for ( let i = 0; i < stopSteps; i ++ ) { t += dt; samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } ); }

	const walkSteps2 = Math.round( walkSec / dt );
	for ( let i = 0; i < walkSteps2; i ++ ) { t += dt; x += fx * moveSpeed * dt; y += fy * moveSpeed * dt; samples.push( { t, x, y, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 } ); }

	return samples;

}

/** Linear speed ramp speedStart->speedEnd over durationSec, straight path (x(t) = speedStart*t + 0.5*((speedEnd-speedStart)/durationSec)*t^2, the exact integral of the linear v(t)). */
function buildRampSamples( { speedStart, speedEnd, durationSec, yaw = 0 } ) {

	const dt = 1 / SYNTH_FPS;
	const n = Math.round( durationSec / dt ) + 1;
	const samples = new Array( n );
	const fx = Math.cos( yaw ), fy = Math.sin( yaw );
	const accel = ( speedEnd - speedStart ) / durationSec;

	for ( let i = 0; i < n; i ++ ) {

		const t = i * dt;
		const dist = speedStart * t + 0.5 * accel * t * t;
		samples[ i ] = { t, x: fx * dist, y: fy * dist, zRoot: PATIENT_HIP_HEIGHT_M, yaw, groundRef: 0 };

	}

	return samples;

}

function buildFlatTerrain( gait ) {

	// start_x_m far away -> heightAt()/treadIndexAt() return 0/-1 everywhere any
	// synthetic path here ever visits (same trick as gait_audit.mjs's own buildFlatTerrain).
	return gait.buildTerrain(
		{ start_x_m: 1e5, step_height_m: 0.145, step_depth_m: 0.305, step_count: 14 },
		1e5 + 14 * 0.305,
	);

}

function buildParams( gait ) {

	// Mirrors gait_audit.mjs's own buildParams() EXACTLY (see its comment): DEFAULT_GAIT_PARAMS
	// with footLateral/toeForwardLen overridden to the real measured Xbot rig values, so
	// this trace reflects the same effective schedule the live app / gait_audit both use.
	return { ...gait.DEFAULT_GAIT_PARAMS, footLateral: 0.082, toeForwardLen: 0.107 };

}

// ===========================================================================
// Rounding (keeps strace_*.json sizes sane -- 1e-6 m/rad, 1e-5 s -- while staying far
// below any visually/mechanically meaningful precision loss).
// ===========================================================================

function r6( v ) { return ( v === null || v === undefined ) ? null : Math.round( v * 1e6 ) / 1e6; }
function r5( v ) { return ( v === null || v === undefined ) ? null : Math.round( v * 1e5 ) / 1e5; }

// ===========================================================================
// The full-series tracer: sweeps poseAt() at `hz` and returns the schema documented in
// this file's header. Runs against ANY built {schedule, terrain} pair, real or synthetic.
// ===========================================================================

function traceSchedule( gait, schedule, terrain, caseName, kind, meta, hz ) {

	const { poseAt } = gait;
	const samples = schedule.samples;
	const duration = samples[ samples.length - 1 ].t;
	const dt = 1 / hz;
	const nSamples = Math.max( 1, Math.round( duration / dt ) );

	const probe = poseAt( schedule, terrain, samples[ 0 ].t );
	const hasCane = !! probe.cane;

	const mkFootSeries = () => ( { x: [], y: [], z: [], yaw: [], planted: [], swingU: [], liftAt: [], landedAt: [], nextLiftAt: [], strideLen: [] } );

	const series = {
		t: [], rootX: [], rootY: [], rootZ: [], rootYaw: [], speed: [], groundSlope: [],
		gaitPhase: [], phaseC: [], support: [],
		leftFoot: mkFootSeries(), rightFoot: mkFootSeries(),
	};
	if ( hasCane ) series.cane = { x: [], y: [], z: [], planted: [], swingU: [], liftAt: [], landedAt: [], nextLiftAt: [] };

	const pushFoot = ( dst, f ) => {

		dst.x.push( r6( f.x ) ); dst.y.push( r6( f.y ) ); dst.z.push( r6( f.z ) ); dst.yaw.push( r6( f.yaw ) );
		dst.planted.push( f.planted ? 1 : 0 );
		dst.swingU.push( r6( f.swingU ) );
		dst.liftAt.push( r5( f.liftAt ) ); dst.landedAt.push( r5( f.landedAt ) ); dst.nextLiftAt.push( r5( f.nextLiftAt ) );
		dst.strideLen.push( r6( f.strideLen ) );

	};

	for ( let i = 0; i <= nSamples; i ++ ) {

		const t = Math.min( duration, i * dt );
		const pose = poseAt( schedule, terrain, t );

		series.t.push( r5( t ) );
		series.rootX.push( r6( pose.rootX ) ); series.rootY.push( r6( pose.rootY ) );
		series.rootZ.push( r6( pose.rootZ ) ); series.rootYaw.push( r6( pose.rootYaw ) );
		series.speed.push( r6( pose.speed ) );
		series.groundSlope.push( r6( pose.groundSlope ) );
		series.gaitPhase.push( r6( pose.gaitPhase ) );
		series.phaseC.push( r6( pose.phaseC ) );
		series.support.push( r6( pose.support ) );

		pushFoot( series.leftFoot, pose.leftFoot );
		pushFoot( series.rightFoot, pose.rightFoot );

		if ( hasCane && pose.cane ) {

			series.cane.x.push( r6( pose.cane.x ) ); series.cane.y.push( r6( pose.cane.y ) ); series.cane.z.push( r6( pose.cane.z ) );
			series.cane.planted.push( pose.cane.planted ? 1 : 0 );
			series.cane.swingU.push( r6( pose.cane.swingU ) );
			series.cane.liftAt.push( r5( pose.cane.liftAt ) ); series.cane.landedAt.push( r5( pose.cane.landedAt ) ); series.cane.nextLiftAt.push( r5( pose.cane.nextLiftAt ) );

		}

	}

	const eventsOut = ( evs ) => evs.map( ( e ) => ( {
		tLift: r5( e.tLift ), tLand: r5( e.tLand ),
		from: { x: r6( e.from.x ), y: r6( e.from.y ), z: r6( e.from.z ) },
		to: { x: r6( e.to.x ), y: r6( e.to.y ), z: r6( e.to.z ) },
		fromYaw: r6( e.fromYaw ), toYaw: r6( e.toYaw ),
	} ) );

	const stepCount = schedule.events.left.length + schedule.events.right.length;

	return {
		trace: {
			case: caseName, kind, meta,
			generatedAt: new Date().toISOString(),
			hz, durationSec: r5( duration ), sampleCount: nSamples + 1,
			params: schedule.params,
			events: {
				left: eventsOut( schedule.events.left ),
				right: eventsOut( schedule.events.right ),
				cane: schedule.caneEvents ? eventsOut( schedule.caneEvents ) : null,
			},
			tail: schedule.tail ? {
				startT: r5( schedule.tail.startT ), freezeT: r5( schedule.tail.freezeT ),
				rootStart: { x: r6( schedule.tail.rootStart.x ), y: r6( schedule.tail.rootStart.y ), zRoot: r6( schedule.tail.rootStart.zRoot ), yaw: r6( schedule.tail.rootStart.yaw ) },
				rootEnd: { x: r6( schedule.tail.rootEnd.x ), y: r6( schedule.tail.rootEnd.y ), zRoot: r6( schedule.tail.rootEnd.zRoot ), yaw: r6( schedule.tail.rootEnd.yaw ) },
			} : null,
			series,
		},
		stepCount,
		duration,
	};

}

// ===========================================================================
// Main
// ===========================================================================

function loadTracks( path ) {

	try {

		return JSON.parse( readFileSync( path, 'utf-8' ) );

	} catch ( err ) {

		console.warn( `[trace_scheduler] could not read tracks file ${ path } (${ err.message }) -- "follow"/"climb" cases will be SKIPPED; synthetic cases still run.` );
		return null;

	}

}

async function main() {

	const args = parseArgs( process.argv.slice( 2 ) );

	console.log( `[trace_scheduler] importing module ${ args.module }` );
	const gait = await import( pathToFileURL( args.module ).href );
	for ( const need of [ 'DEFAULT_GAIT_PARAMS', 'buildTerrain', 'extractPathSamples', 'buildSchedule', 'poseAt' ] ) {

		if ( typeof gait[ need ] === 'undefined' ) {

			console.error( `[trace_scheduler] FATAL: ${ args.module } does not export "${ need }" -- is this really a PatientGait.js-shaped module?` );
			process.exit( 2 );

		}

	}

	mkdirSync( args.outDir, { recursive: true } );

	// Registry of every case this tool can build: { name, kind, meta, buildSamples(gait), buildTerrainFn(gait) }.
	// buildTerrainFn defaults to the flat synthetic terrain; real cases override it below.
	const registry = [];

	const tracksData = loadTracks( args.tracks );
	if ( tracksData ) {

		const realTerrain = gait.buildTerrain( tracksData.stair_spec, tracksData.landing_far_x_m );
		for ( const clipName of [ 'follow', 'climb' ] ) {

			const clip = tracksData.clips && tracksData.clips[ clipName ];
			if ( ! clip ) { console.warn( `[trace_scheduler] tracks.json has no "${ clipName }" clip -- skipping` ); continue; }
			registry.push( {
				name: clipName, kind: 'real', meta: { source: 'robot.glb patient_root track' },
				samples: gait.extractPathSamples( clip.posTimes, clip.posValues, clip.quatTimes, clip.quatValues ),
				terrain: realTerrain,
			} );

		}

	}

	const flatTerrain = buildFlatTerrain( gait );

	// Constant-speed sweep (30 s each) -- cadence/step-length-vs-speed curve (metric 1).
	for ( const speed of [ 0.08, 0.13, 0.20, 0.26, 0.35, 0.45, 0.60 ] ) {

		registry.push( {
			name: `const_${ speed.toFixed( 2 ) }`, kind: 'synthetic', meta: { speed, durationSec: 30 },
			samples: buildConstSpeedSamples( { speed, durationSec: 30 } ),
			terrain: flatTerrain,
		} );

	}

	// Constant-curvature arcs at the follow clip's own recorded speed (0.26 m/s) -- turning naturalness (metric 4).
	for ( const radius of [ 0.75, 1.5, 3.0 ] ) {

		registry.push( {
			name: `arc_r${ radius.toFixed( 2 ) }`, kind: 'synthetic', meta: { speed: 0.26, radius, durationSec: 24, turnSign: 1 },
			samples: buildArcSamples( { speed: 0.26, radius, durationSec: 24, turnSign: 1 } ),
			terrain: flatTerrain,
		} );

	}

	// Stop-and-go (hard stop) -- stop/start dynamics (metric 5).
	registry.push( {
		name: 'stopgo_hard', kind: 'synthetic', meta: { moveSpeed: 0.26, walkSec: 8, stopSec: 3 },
		samples: buildStopGoHardSamples( { moveSpeed: 0.26, walkSec: 8, stopSec: 3 } ),
		terrain: flatTerrain,
	} );

	// Speed ramp -- stop/start dynamics + walk-ratio curve cross-check (metric 5 / metric 1).
	registry.push( {
		name: 'ramp', kind: 'synthetic', meta: { speedStart: 0.1, speedEnd: 0.5, durationSec: 20 },
		samples: buildRampSamples( { speedStart: 0.1, speedEnd: 0.5, durationSec: 20 } ),
		terrain: flatTerrain,
	} );

	const only = args.only ? new Set( args.only ) : null;
	const params = buildParams( gait );
	const index = [];

	for ( const c of registry ) {

		if ( only && ! only.has( c.name ) ) continue;

		const schedule = gait.buildSchedule( c.samples, c.terrain, params );
		const { trace, stepCount, duration } = traceSchedule( gait, schedule, c.terrain, c.name, c.kind, c.meta, args.hz );

		const outPath = resolve( args.outDir, `strace_${ c.name }.json` );
		writeFileSync( outPath, JSON.stringify( trace ) + '\n', 'utf-8' );

		const hasCane = !! trace.series.cane;
		console.log( `[trace_scheduler] wrote ${ outPath } (duration=${ duration.toFixed( 2 ) }s steps=${ stepCount } cane=${ hasCane } samples=${ trace.sampleCount })` );

		index.push( { name: c.name, kind: c.kind, meta: c.meta, file: `strace_${ c.name }.json`, durationSec: r5( duration ), stepCount, hasCane } );

	}

	const indexPath = resolve( args.outDir, 'strace_index.json' );
	writeFileSync( indexPath, JSON.stringify( { generatedAt: new Date().toISOString(), module: args.module, hz: args.hz, cases: index }, null, 2 ) + '\n', 'utf-8' );
	console.log( `[trace_scheduler] wrote ${ indexPath } (${ index.length } cases)` );

}

main().catch( ( err ) => {

	console.error( '[trace_scheduler] FATAL:', err );
	process.exitCode = 2;

} );
