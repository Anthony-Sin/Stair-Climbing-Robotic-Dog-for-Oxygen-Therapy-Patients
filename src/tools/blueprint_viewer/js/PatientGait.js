// PatientGait.js
//
// Pure-math, ZERO-import (no THREE, no DOM) procedural gait scheduler for the patient
// mannequin. Replaces the old Python-baked scalar pose (patient_pose.json /
// pipeline/anim_bake.py's per-frame hip_pitch/knee_bend) with a viewer-side schedule
// computed once per clip at load time, then sampled statelessly during scrub/playback.
//
// WHY this exists (see js/PatientHuman.js's header + AGENTS.md for the full history):
// the old pipeline baked leg angles frame-by-frame in Python from a TIME-driven gait
// (a fixed cycle_period_s ticking regardless of whether the root was actually moving).
// That produced the reported "moonwalking": the gait kept cycling through the
// recorded path's stop-and-go pauses, so feet visibly slid/stepped in place while the
// body stood still. This module fixes that at the root: steps are triggered by
// ACTUAL ROOT DISPLACEMENT/YAW NEED (see buildSchedule), never by elapsed time, so a
// stopped root structurally cannot produce a moving foot -- there is no "cycle phase
// clock" here at all, only discrete liftoff/touchdown EVENTS gated on real movement.
//
// Coordinate convention: everything in this module is expressed in the "P-frame" --
// isaac_world's own local frame (the pipeline's native convention: X forward, Y
// lateral, Z up; see pipeline/gltf_export.py / AGENTS.md incident #4's "Coordinate
// reconciliation" note). patient_root's baked position/quaternion tracks are already
// in this frame (they're literally isaac_world's child), so no basis conversion
// happens in this module at all -- PatientHuman.js is the ONLY place that reconciles
// P-frame directions against Xbot's own local axes (via B_PLACEMENT), exactly as it
// already does for the anchor's own placement.
//
// Determinism / scrub-safety contract: buildSchedule() is the ONE place allowed to
// carry mutable, order-dependent state (it marches forward through the sample array
// once, "deciding" each footstep event as it goes -- the same "decide once at a
// discrete event, never recompute live" discipline as AGENTS.md incident #6, applied
// here at the SCHEDULE-BUILDING level instead of per-frame). poseAt() is the ONLY
// entry point the viewer calls every sync()/scrub, and it is a pure function of
// (schedule, terrain, t): no closures over mutable state, no memo of "last t queried" --
// querying t=27.3 then t=5.0 then t=27.3 again must return bit-identical results the
// third time as the first (the viewer is scrub-driven, i.e. genuinely random-access).

// ===========================================================================
// Tunables
// ===========================================================================

export const DEFAULT_GAIT_PARAMS = {
	stepTrigger: 0.16, // m -- flat-ground horizontal "need" (foot drift from nominal) that triggers a step
	stepTriggerClimb: 0.12, // m -- lower on stairs: treads are narrow, so a foot must react sooner or it runs out of tread to plant on
	stepLead: 0.08, // m -- touchdown target leads the root's facing direction by this much (a real stride lands slightly ahead of "under the hip")
	swingDur: 0.32, // s -- flat-ground swing duration (liftoff -> touchdown)
	swingDurClimb: 0.45, // s -- longer on stairs: clearing a riser needs a slower, more deliberate swing
	swingClearance: 0.07, // m -- flat-ground vertical margin added over the highest terrain sample along a swing's path
	swingClearanceClimb: 0.10, // m -- taller margin on stairs so the swinging foot clears a riser nosing, not just the tread top
	minEventGap: 0.15, // s -- minimum time between one foot's swing ENDING and the SAME foot starting another (prevents rapid double-triggers)
	yawErrorWeight: 0.30, // m/rad -- converts a plant-vs-current yaw error into an equivalent "need" distance; tuned so ~25 deg (0.44 rad) of yaw error alone crosses stepTrigger (0.44*0.30=0.132, just under 0.16 -- combines with even a little translational need to trigger, matching "yaw alone eventually triggers, not instantly")
	idleSpeedThreshold: 0.02, // m/s -- SAME value the browser diagnostic (main.js patientDiag) uses to define "idle" for idleFootMotionMax; a swing may only START at a sample where root translational speed OR yaw rate clears its own idle floor (see buildSchedule) -- deliberately shared so "does a step trigger" and "does the diagnostic call this idle" can never disagree
	idleYawRateThreshold: 0.05, // rad/s -- companion to idleSpeedThreshold: an in-place turn (near-zero translational speed, real yaw rate) must still be able to trigger an adjustment step, so idleness requires BOTH speed and yaw-rate to be below their floors, not just speed alone
	idleSustainSamples: 3, // count -- the idle gate requires this many CONSECUTIVE trailing samples to all clear the idle floor (not just the trigger sample itself), so a trigger can't fire on the single leading-edge sample of a resume-from-stop, whose swing would otherwise still span mostly-idle samples just before it
	heelMargin: 0.05, // m -- keep the foot's heel/back edge this far from a tread's near (riser) edge
	nosingMargin: 0.03, // m -- keep the foot's toe this far from a tread's far (nosing) edge
	bobAmplitude: 0.015, // m -- vertical anchor bob amplitude, phase-locked to gaitPhase (freezes when steps stop)
	footLateral: 0.09 * 0.75, // m -- half-stance-width (nominal foot lateral offset from the root). This default matches the OLD (now-retired) Python pipeline's anim_bake._PATIENT_LEG_HIP_OFFSET magnitude, kept only so this module stays usable standalone (Node tests, this file's own header) -- PatientHuman.buildGait() ALWAYS overrides this with the REAL measured hip-pivot lateral offset from Xbot's own bind pose (~0.082 m, close but not identical to this default) before building a schedule for the live app, exactly like toeForwardLen below
	toeForwardLen: 0.107, // m -- horizontal Foot->ToeBase reach, measured from Xbot's own bind pose (see PatientHuman.js's load-time measurement) -- default here is that measured value, duplicated so this module stays load-order-independent (PatientHuman passes the REAL measured value in at buildGait() time; this default only matters for standalone/Node testing)
	// Torso lean gains (consumed by PatientHuman.js, not this module -- kept here so
	// every gait-related tunable lives in one place): torsoPitch = clamp(leanBase +
	// leanSpeedK*speed + leanSlopeK*groundSlope, 0, 0.15) rad.
	leanBase: 0.02,
	leanSpeedK: 0.05,
	leanSlopeK: 0.55,
};

// ===========================================================================
// Terrain
// ===========================================================================

/**
 * Discrete stair-terrain height function, matching pipeline/synthetic_motion.py's
 * `terrain_height` EXACTLY (generate_climb_frames, ~line 435-443):
 *   x <  start_x            -> 0
 *   x >= top_x               -> top_h (= step_count * step_h)
 *   otherwise                -> min(top_h, (floor((x-start_x)/step_d)+1) * step_h)
 * i.e. tread i (0-indexed) spans x in [start_x + i*step_d, start_x + (i+1)*step_d)
 * and its top sits at (i+1)*step_h. `landingFarX` is accepted for API symmetry with
 * the meta.json's own `landing_far_x_m` field but does not change the height function
 * (the analytic model already clamps flat at top_h for any x >= top_x, which covers
 * the full landing depth including landingFarX) -- kept as a parameter rather than a
 * hardcoded landing extent purely so callers don't need to special-case "how far does
 * the flat landing go" separately from the terrain query.
 */
export function buildTerrain( stairSpec, landingFarX ) {

	const startX = stairSpec.start_x_m;
	const stepH = stairSpec.step_height_m;
	const stepD = stairSpec.step_depth_m;
	const stepCount = stairSpec.step_count;
	const topX = startX + stepCount * stepD;
	const topH = stepCount * stepH;

	function heightAt( x ) {

		if ( x < startX ) return 0.0;
		if ( x >= topX ) return topH;
		const stepIdx = Math.floor( ( x - startX ) / stepD );
		return Math.min( topH, ( stepIdx + 1 ) * stepH );

	}

	/** Which tread index (0-based) x falls on, -1 if before the stairs, stepCount if at/past the top landing. */
	function treadIndexAt( x ) {

		if ( x < startX ) return - 1;
		if ( x >= topX ) return stepCount;
		return Math.floor( ( x - startX ) / stepD );

	}

	/** [xStart, xEnd) world-X span of tread index `idx` (0-based). */
	function treadSpan( idx ) {

		return { xStart: startX + idx * stepD, xEnd: startX + ( idx + 1 ) * stepD };

	}

	return {
		heightAt, treadIndexAt, treadSpan,
		startX, topX, topH, stepH, stepD, stepCount, landingFarX,
	};

}

// ===========================================================================
// Path sample extraction
// ===========================================================================

/**
 * Convert THREE.AnimationClip-style flat (times, flattened-values) keyframe track
 * pairs for patient_root's position and quaternion into a dense per-keyframe sample
 * array `[{t, x, y, zRoot, yaw}]`.
 *
 * `posValues`/`quatValues` are FLAT typed arrays (3 floats per position key, 4 per
 * quaternion key, matching THREE.KeyframeTrack.values / a raw glTF accessor dump --
 * see PatientHuman.buildGait for how these are pulled from clip.tracks). Quaternion
 * component order is three.js's own (x, y, z, w).
 *
 * Sampling strategy: walk the POSITION keys as the master timeline (position and
 * rotation tracks are baked at the same fps by pipeline/gltf_export.py, so they share
 * the same key count/times in practice) and linearly interpolate the quaternion track
 * at each position key's time. This keeps the returned sample array in one-sample-
 * per-baked-frame lockstep with the source data (no resampling/aliasing), which
 * matters for buildSchedule's forward march (it needs to "walk the actual sample
 * array forward", not synthesize intermediate samples -- see its own docs).
 *
 * yaw: extracted as the rotation-about-+Z angle from a quaternion that (per the
 * pipeline's own contract, anim_bake.bake_clip: "rotation ... (yaw only)") has zero
 * roll/pitch, i.e. x=y=0 and yaw = 2*atan2(z, w). Unwrapped for continuity (no +-pi
 * seam jumps) since a raw atan2 output wraps at +-pi and this module needs a
 * continuous facing-direction error against a remembered plant yaw.
 *
 * ground reference: zRoot - 0.92 (PATIENT_HIP_HEIGHT_M, pipeline/anim_bake.py) --
 * recovers the raw recorded ground/terrain height under the patient, matching
 * anim_bake.bake_clip's own `ppos = (x, y, ground_z + PATIENT_HIP_HEIGHT_M)` (~L276).
 * Exposed per-sample as `groundRef` for callers that want it (not currently consumed
 * by buildSchedule, which uses the ANALYTIC terrain function instead of this recorded
 * value -- see buildSchedule's own docs for why: the recorded ground_z is a SMOOTH
 * RAMP on real climb data (AGENTS.md incident #12), not the discrete per-tread shape
 * this module's feet must snap to, so it is not a substitute for buildTerrain()).
 */
const PATIENT_HIP_HEIGHT_M = 0.92; // kept in lockstep with pipeline/anim_bake.PATIENT_HIP_HEIGHT_M

export function extractPathSamples( posTimes, posValues, quatTimes, quatValues ) {

	const n = posTimes.length;
	const samples = new Array( n );

	let unwrapOffset = 0.0;
	let prevRawYaw = null;

	for ( let i = 0; i < n; i ++ ) {

		const t = posTimes[ i ];
		const x = posValues[ i * 3 + 0 ];
		const y = posValues[ i * 3 + 1 ];
		const zRoot = posValues[ i * 3 + 2 ];

		const q = _sampleQuatAt( quatTimes, quatValues, t );
		// Pure-Z rotation contract (anim_bake.bake_clip: patient_root rotation is
		// "yaw only"): yaw = 2*atan2(z, w) reads the rotation angle about +Z directly
		// off the quaternion's own z/w components, exact for any x=y=0 quaternion
		// (no need to build a full matrix/Euler decomposition for a single-axis case).
		let rawYaw = 2.0 * Math.atan2( q[ 2 ], q[ 3 ] );

		if ( prevRawYaw !== null ) {

			// Unwrap: if this key's raw yaw jumped by more than pi from the previous
			// raw yaw, it's the atan2 branch cut, not a real >180 deg single-frame
			// turn -- add/subtract 2*pi to keep the running "continuous" yaw close to
			// its predecessor. Accumulated in unwrapOffset so multiple wraps compound
			// correctly across the whole track.
			let delta = rawYaw - prevRawYaw;
			while ( delta > Math.PI ) { delta -= 2 * Math.PI; unwrapOffset -= 2 * Math.PI; }
			while ( delta < - Math.PI ) { delta += 2 * Math.PI; unwrapOffset += 2 * Math.PI; }

		}

		prevRawYaw = rawYaw;
		const yaw = rawYaw + unwrapOffset;

		samples[ i ] = { t, x, y, zRoot, yaw, groundRef: zRoot - PATIENT_HIP_HEIGHT_M };

	}

	return samples;

}

/** Linear-interpolated quaternion (x,y,z,w) lookup at an arbitrary t, clamped at either end of the track. Component-wise lerp (not slerp) is sufficient here: patient_root's baked rotation is yaw-only and the source keys are dense (30fps), so the shortest-arc error from a linear x/y/z/w blend between adjacent keys is negligible -- and this only feeds yaw extraction (2*atan2), not a rendered orientation. */
function _sampleQuatAt( times, values, t ) {

	const n = times.length;
	if ( n === 0 ) return [ 0, 0, 0, 1 ];
	if ( t <= times[ 0 ] ) return [ values[ 0 ], values[ 1 ], values[ 2 ], values[ 3 ] ];
	if ( t >= times[ n - 1 ] ) {

		const j = ( n - 1 ) * 4;
		return [ values[ j ], values[ j + 1 ], values[ j + 2 ], values[ j + 3 ] ];

	}

	for ( let i = 1; i < n; i ++ ) {

		if ( times[ i ] >= t ) {

			const t0 = times[ i - 1 ], t1 = times[ i ];
			const frac = t1 > t0 ? ( t - t0 ) / ( t1 - t0 ) : 0;
			const j0 = ( i - 1 ) * 4, j1 = i * 4;
			const out = [ 0, 0, 0, 0 ];
			for ( let c = 0; c < 4; c ++ ) out[ c ] = values[ j0 + c ] + frac * ( values[ j1 + c ] - values[ j0 + c ] );
			return out;

		}

	}

	const j = ( n - 1 ) * 4;
	return [ values[ j ], values[ j + 1 ], values[ j + 2 ], values[ j + 3 ] ];

}

// ===========================================================================
// Small vector helpers (plain {x,y,z} objects -- no THREE)
// ===========================================================================

function _hyp2( dx, dy ) { return Math.sqrt( dx * dx + dy * dy ); }

/** Nominal (un-swept) foot plant point for `side` ('left'|+1 lateral, 'right'|-1 lateral) at sample `s`: root XY + yaw-rotated lateral offset, terrain height at that X. This is where a foot "wants" to be when the body isn't demanding a step -- the reference buildSchedule measures drift/need against. */
function _nominalAt( s, sign, footLateral, terrain ) {

	const cy = Math.cos( s.yaw ), sy = Math.sin( s.yaw );
	// Lateral offset (0, sign*footLateral, 0) rotated by yaw about +Z, added to root XY.
	const x = s.x + ( - sy * ( sign * footLateral ) );
	const y = s.y + ( cy * ( sign * footLateral ) );
	return { x, y, z: terrain.heightAt( x ), yaw: s.yaw };

}

/** Smoothstep ease, 0 at u=0, 1 at u=1, zero slope at both ends. */
function _smoothstep( u ) { return u * u * ( 3.0 - 2.0 * u ); }

/** Shortest-path angle difference a-b, wrapped to [-pi, pi]. */
function _angleDiff( a, b ) {

	let d = a - b;
	while ( d > Math.PI ) d -= 2 * Math.PI;
	while ( d < - Math.PI ) d += 2 * Math.PI;
	return d;

}

// ===========================================================================
// Schedule builder
// ===========================================================================

/**
 * Build a per-clip footfall schedule by marching FORWARD ONCE through `samples`
 * (stateful during this pass only -- see this module's header for the determinism
 * contract poseAt() upholds afterward).
 *
 * Feet start planted at their snapped nominal (root XY + lateral offset, at t=samples[0].t).
 * A foot may begin a swing only if the OTHER foot is not currently swinging and at
 * least `minEventGap` seconds have passed since any event ended. The candidate foot is
 * whichever has larger "need": planar (XY) distance of its current planted position
 * from its LIVE nominal (recomputed each sample -- this is fine/required here since
 * "need" is explicitly a measure of how far the world has moved out from under a
 * still-planted foot, not a committed event value) plus `yawErrorWeight * |yaw error
 * vs the yaw the foot was planted at|`. A step triggers once that need exceeds
 * `stepTrigger` (`stepTriggerClimb` while the touchdown nominal is on the staircase).
 *
 * On trigger:
 *   - liftoff = the swinging foot's CURRENT planted position (a snapshot -- AGENTS.md
 *     incident #6's "decide once, hold until the next event" discipline applied to
 *     step scheduling itself, not just intra-swing height/position blending).
 *   - touchdown time = triggerT + swingDur (swingDurClimb if the eventual touchdown
 *     lands on stairs -- resolved AFTER finding the raw touchdown nominal, see below).
 *   - touchdown target: found by WALKING the sample array forward from the trigger
 *     sample to the sample nearest touchdown time (never velocity-extrapolated -- the
 *     recorded path can accelerate/decelerate/turn during the swing window, and using
 *     the actual future sample is exact where extrapolation would drift), taking that
 *     future sample's nominal foot point, offset by `stepLead` along ITS OWN facing
 *     direction (a real stride reaches slightly ahead of "directly under the hip").
 *   - stair snap: if the touchdown nominal's X falls within the staircase's world-X
 *     span, clamp X into the tread it would land on (nearest valid position within
 *     [treadStart+heelMargin, treadEnd-toeForwardLen-nosingMargin], so the WHOLE foot
 *     footprint -- heel to toe -- fits on one tread) and set Z to
 *     terrain.heightAt(snappedX) EXACTLY (flat ground already gives z=0 via the same
 *     terrain query, so no separate flat-ground case is needed).
 *   - swing clearance: sample the terrain every <=2cm along the liftoff->touchdown
 *     segment, take the max, add swingClearance (swingClearanceClimb if the event
 *     lands on stairs) -- this is the height poseAt's endpoint-fading clamp arcs up to
 *     at the swing's midpoint (see poseAt).
 *
 * Also produces a monotone gait-PHASE timeline: phase advances by 0.5 at each event
 * (left events land on integer phases, right on half-integer), held frozen between
 * events -- consumed by PatientHuman.js to phase-lock the anchor bob and the canned
 * upper-body walk clip, so both freeze exactly when steps stop (never a live/time
 * clock -- see this module's header).
 */
export function buildSchedule( samples, terrain, params = DEFAULT_GAIT_PARAMS ) {

	const p = { ...DEFAULT_GAIT_PARAMS, ...params };
	const n = samples.length;
	if ( n === 0 ) throw new Error( 'buildSchedule: empty samples array' );

	const s0 = samples[ 0 ];
	const sides = {
		left: { sign: + 1 },
		right: { sign: - 1 },
	};

	// Per-foot running state during the forward march.
	const state = {
		left: { plantedPos: _nominalAt( s0, + 1, p.footLateral, terrain ), plantYaw: s0.yaw, swinging: false, lastEventEndT: - Infinity },
		right: { plantedPos: _nominalAt( s0, - 1, p.footLateral, terrain ), plantYaw: s0.yaw, swinging: false, lastEventEndT: - Infinity },
	};

	const events = { left: [], right: [] };
	// Phase timeline: one entry per sample index, monotone non-decreasing, advances by
	// 0.5 exactly at each event's LIFTOFF sample and holds constant otherwise (matches
	// poseAt's own "frozen between events, interpolates only during swings" contract).
	// Derived in a SEPARATE pass after the event march completes (_fillPhaseTimeline,
	// below) rather than incrementally during the march itself: the march's own event
	// order already fully determines each foot's phase-parity sequence (left events
	// land on integer phases 0,1,2,..., right on half-integer 0.5,1.5,2.5,... -- a
	// standard 2-beat gait's contralateral phase convention), so re-deriving it from
	// the finished `events` lists is simpler and unambiguous compared to threading
	// running counters through the march loop below.
	const phaseAtSampleIdx = new Float64Array( n );

	// Active-swing bookkeeping (at most one event per foot in flight at a time).
	const active = { left: null, right: null };

	for ( let i = 0; i < n; i ++ ) {

		const s = samples[ i ];

		// Resolve any swing whose touchdown time has arrived (process before evaluating
		// new triggers this sample, so a foot that lands and immediately needs another
		// step -- e.g. a sharp turn -- is eligible this same pass, gated by minEventGap
		// same as any other trigger).
		for ( const foot of [ 'left', 'right' ] ) {

			const ev = active[ foot ];
			if ( ev && s.t >= ev.tLand ) {

				state[ foot ].plantedPos = { x: ev.to.x, y: ev.to.y, z: ev.to.z, yaw: ev.toYaw };
				state[ foot ].plantYaw = ev.toYaw;
				state[ foot ].swinging = false;
				state[ foot ].lastEventEndT = ev.tLand;
				active[ foot ] = null;

			}

		}

		// Idle gate: a swing may only START at a sample where the root itself has
		// genuine, SUSTAINED ongoing motion -- translational speed OR yaw rate above
		// their own idle floors (see DEFAULT_GAIT_PARAMS.idleSpeedThreshold/
		// idleYawRateThreshold's comments for why BOTH, not just speed alone, matter:
		// an in-place turn has near-zero translational speed but real yaw rate, and
		// per this module's header/the product spec, in-place adjustment steps during
		// a zigzag turn must still be possible).
		//
		// "Sustained" = non-idle at EVERY sample across a short trailing window
		// (idleSustainSamples, checked below), not just the current instant. A single
		// central-difference sample only "sees" a very narrow (~2 samples) window --
		// right at an idle->moving TRANSITION, the FIRST sample or two after a resume
		// already reads as "moving" (the derivative spans the transition), even though
		// the root has only genuinely been in motion for a fraction of a second. A
		// trigger firing on that very first post-resume sample still leaves its swing
		// spanning mostly-idle samples immediately BEFORE it (the mirror image of the
		// "mostly-idle AFTER" case the separate windowMotion gate above already
		// handles by looking forward from the trigger to the touchdown). Requiring the
		// trailing window to be UNANIMOUSLY non-idle filters out that single-sample
		// "just caught the leading edge of a resume" case while still admitting any
		// trigger sample that occurs after the root has genuinely been moving for that
		// whole window -- which any real sustained walk/turn satisfies trivially.
		// Without EITHER half of this gate (the instantaneous check the loop below
		// still performs, or this sustain requirement), a foot whose NEED had been
		// legitimately accumulating during real upstream motion can cross its
		// threshold at a sample that reads "moving" only by a hair's-breadth right at
		// a stop or a resume, animating a swing while the root reads idle for most of
		// it either way -- observed directly on a synthetic stop-and-go probe.
		let rootIsIdle = false;
		for ( let back = 0; back < p.idleSustainSamples; back ++ ) {

			const j = i - back;
			if ( j < 0 ) break;
			const iPrev = Math.max( 0, j - 1 );
			const iNext = Math.min( n - 1, j + 1 );
			const sp = samples[ iPrev ], sn = samples[ iNext ];
			const dt = sn.t - sp.t;
			if ( dt <= 1e-6 ) continue;
			const spd = _hyp2( sn.x - sp.x, sn.y - sp.y ) / dt;
			const yr = Math.abs( sn.yaw - sp.yaw ) / dt;
			if ( spd < p.idleSpeedThreshold && yr < p.idleYawRateThreshold ) { rootIsIdle = true; break; }

		}
		if ( rootIsIdle ) continue;

		// Evaluate trigger candidates: only feet that are NOT swinging and have cleared
		// minEventGap since their own last event may be considered this sample. "need"
		// is a LIVE measurement (recomputed fresh every sample, deliberately -- it's
		// asking "how far has the world moved out from under this still-planted foot
		// RIGHT NOW", not a value that should be snapshotted/held; contrast with the
		// touchdown target resolved just below, which per AGENTS.md incident #6 IS
		// decided once and held).
		//
		// Threshold gate: a candidate must clear a MINIMUM need before it's even
		// considered -- using stepTriggerClimb (the SMALLER of the two thresholds,
		// since stairs demand a quicker reaction) as this first-pass filter can never
		// wrongly exclude a legitimate flat-ground trigger (whose own, larger,
		// threshold is checked precisely once the eventual touchdown context is known,
		// a few lines below) -- it only ever admits a possibly-too-small-for-flat-
		// ground candidate, which the later re-check then correctly rejects. Without
		// this gate, `bestFoot` would always be non-null once minEventGap clears (ANY
		// nonzero drift/yawErr, however microscopic, "wins" the argmax against the
		// other foot's disqualified state) -- firing a swing almost every sample, the
		// exact mechanism behind an earlier observed idleFootMotionMax violation
		// (~0.048 m against a 0.002 m bar) traced to this gate being missing entirely.
		let bestFoot = null, bestNeed = - Infinity;

		for ( const foot of [ 'left', 'right' ] ) {

			if ( state[ foot ].swinging ) continue;
			if ( s.t - state[ foot ].lastEventEndT < p.minEventGap ) continue;
			// The OTHER foot must not currently be swinging (never both feet in the
			// air at once -- this is a walking gait, not a run).
			const other = foot === 'left' ? 'right' : 'left';
			if ( state[ other ].swinging ) continue;

			const nominal = _nominalAt( s, sides[ foot ].sign, p.footLateral, terrain );
			const planted = state[ foot ].plantedPos;
			const drift = _hyp2( nominal.x - planted.x, nominal.y - planted.y );
			const yawErr = Math.abs( _angleDiff( s.yaw, state[ foot ].plantYaw ) );
			const need = drift + p.yawErrorWeight * yawErr;

			if ( need < p.stepTriggerClimb ) continue; // first-pass filter, see comment above
			if ( need > bestNeed ) { bestNeed = need; bestFoot = foot; }

		}

		if ( bestFoot === null ) continue;

		// Resolve the eventual touchdown target by WALKING the sample array forward
		// (never velocity-extrapolating) from a provisional flat-ground swingDur, so
		// the stair-vs-flat trigger threshold/duration/clearance choice is based on
		// where the foot is ACTUALLY going to land, not a chicken-and-egg guess. If
		// that provisional target lands on stairs, extend the search window to
		// swingDurClimb and re-resolve (the touchdown TIME itself also switches to
		// swingDurClimb in that case) -- at most one extra forward-walk, since a
		// target found within the shorter window can only move FURTHER forward (later
		// sample) when re-searched with the longer window, never behind the staircase
		// it already reached.
		let swingDur = p.swingDur;
		let touchdownSampleIdx = _findSampleAtOrAfter( samples, i, s.t + swingDur );
		let touchdownSample = samples[ touchdownSampleIdx ];
		let touchdownNominal = _nominalAt( touchdownSample, sides[ bestFoot ].sign, p.footLateral, terrain );

		let touchdownOnStairs = terrain.treadIndexAt( touchdownNominal.x ) >= 0 && terrain.treadIndexAt( touchdownNominal.x ) < terrain.stepCount;
		if ( touchdownOnStairs ) {

			swingDur = p.swingDurClimb;
			touchdownSampleIdx = _findSampleAtOrAfter( samples, i, s.t + swingDur );
			touchdownSample = samples[ touchdownSampleIdx ];
			touchdownNominal = _nominalAt( touchdownSample, sides[ bestFoot ].sign, p.footLateral, terrain );
			touchdownOnStairs = terrain.treadIndexAt( touchdownNominal.x ) >= 0 && terrain.treadIndexAt( touchdownNominal.x ) < terrain.stepCount;

		}

		// Final threshold re-check, now that the resolved context (flat vs stairs) is
		// known: the first-pass filter above only guaranteed `need >= stepTriggerClimb`
		// (the SMALLER threshold) -- if this touchdown resolved to FLAT ground, the
		// correct (larger) `stepTrigger` may not actually be cleared yet, in which case
		// this sample does not fire (the still-growing need is simply re-evaluated next
		// sample, exactly as if this candidate had never been found -- no state was
		// mutated above, so this `continue` is entirely safe).
		const requiredTrigger = touchdownOnStairs ? p.stepTriggerClimb : p.stepTrigger;
		if ( bestNeed < requiredTrigger ) continue;

		// "Won't-actually-go-anywhere" gate: the instantaneous idle check just above
		// (rootIsIdle) only looks at a single sample's local derivative, which is too
		// short-sighted to see a stop that is still a few samples away -- a REAL
		// recorded path can decelerate gradually over several samples (unlike this
		// module's own synthetic stress-test, which deliberately used an unrealistic
		// instant stop) and still leave a swing that lifts off while the instantaneous
		// gate reads "moving" but lands well into a now-fully-stopped root. Since the
		// touchdown sample was JUST resolved by walking forward anyway, use it
		// directly: require the root to cover a non-trivial distance (or turn a
		// non-trivial amount) over the WHOLE prospective [liftoff, touchdown] window,
		// not just at the liftoff instant. If it wouldn't, defer -- don't fire this
		// sample; the still-growing need is simply re-evaluated at the next sample
		// (exactly like the threshold re-check above), which naturally waits either
		// for the root to resume moving (giving a window that clears this gate) or,
		// worst case, the swing ends up starting later/shorter rather than orphaned
		// mid-flight through a long stop. Found directly against REAL recorded data
		// (not just the synthetic stress-test): the "climb" clip has a genuine
		// stop-and-go pause starting ~t=33.17s (root x frozen exactly, no yaw change,
		// for ~0.43s) that a right-foot swing lifting off at t=33.10 (need had grown
		// during the preceding stance, crossing threshold right as the root began
		// decelerating) ran through almost entirely -- idleFootMotionMax 0.058 m
		// against the 0.002 m bar before this gate existed.
		const windowDx = touchdownSample.x - s.x, windowDy = touchdownSample.y - s.y;
		const windowDist = _hyp2( windowDx, windowDy );
		const windowYawErr = Math.abs( _angleDiff( touchdownSample.yaw, s.yaw ) );
		const windowMotion = windowDist + p.yawErrorWeight * windowYawErr;
		if ( windowMotion < requiredTrigger * 0.5 ) continue;

		// stepLead: offset the touchdown target forward along the touchdown sample's
		// OWN facing direction (a real stride reaches slightly ahead of "directly
		// under the hip" at the moment of plant).
		const leadCy = Math.cos( touchdownSample.yaw ), leadSy = Math.sin( touchdownSample.yaw );
		let toX = touchdownNominal.x + leadCy * p.stepLead;
		let toY = touchdownNominal.y + leadSy * p.stepLead;
		let toZ = terrain.heightAt( toX );
		const onStairs = terrain.treadIndexAt( toX ) >= 0 && terrain.treadIndexAt( toX ) < terrain.stepCount;

		if ( onStairs ) {

			const idx = terrain.treadIndexAt( toX );
			const span = terrain.treadSpan( idx );
			const lo = span.xStart + p.heelMargin;
			const hi = span.xEnd - p.toeForwardLen - p.nosingMargin;
			// Clamp to the nearest valid position within the tread's footprint margin
			// (if the margins are inverted -- a tread narrower than heelMargin+toe
			// reach+nosingMargin, not the case for this staircase's 0.305 m depth, but
			// guarded generically -- fall back to the tread's own center).
			const clampedLo = Math.min( lo, hi );
			const clampedHi = Math.max( lo, hi );
			toX = Math.min( clampedHi, Math.max( clampedLo, toX ) );
			toZ = terrain.heightAt( toX ); // EXACT tread-top height at the (possibly re-clamped) snapped X

		}

		const from = { x: state[ bestFoot ].plantedPos.x, y: state[ bestFoot ].plantedPos.y, z: state[ bestFoot ].plantedPos.z };
		const to = { x: toX, y: toY, z: toZ };
		const fromYaw = state[ bestFoot ].plantYaw;
		const toYaw = touchdownSample.yaw;

		// Apex/clamp data: precomputed (at build time, "decide once and hold" --
		// AGENTS.md incident #6) along the liftoff->touchdown segment at <=2cm spacing.
		//
		// clampProfile stores the RAW terrain height (running max, so it stays
		// monotone non-decreasing -> smooth to interpolate), sampled at <=2cm spacing
		// along the straight liftoff->touchdown line. poseAt's clamp uses this as a
		// strict non-penetration FLOOR ONLY -- it does NOT separately add the
		// clearance margin (see poseAt: `z = max(zArc, terrainProfile(ease))`, no
		// extra `+ clearance*sin(pi*u)` term on the clamp side). Clearance is already
		// fully provided by zArc's OWN arc bump (its coefficient is
		// `apexZ - max(from.z,to.z)`, and `apexZ = maxTerrainAlongPath + clearance` --
		// i.e. the arc already peaks `clearance` above the highest terrain along the
		// path), so the clamp's only remaining job is "never actually go below ground"
		// -- a strictly weaker, purely defensive condition. Two prior formulations
		// were tried and rejected: (1) clamping against
		// `rawTerrain + clearance*sin(pi*u)` DOUBLE-counted clearance on top of the
		// arc bump's own, compounding right where both were steepest (0.1275 m single-
		// dt=0.05-sample jump on a real single-riser "climb straight to touchdown"
		// event, maxToeStep 0.131 m against the 0.12 m bar); (2) storing the EXCESS
		// over a straight-line reference between the endpoints' own terrain heights
		// (rather than the raw terrain height) was meant to zero out the clamp for
		// that same common case, but a discrete STEP terrain function is "all excess"
		// relative to any LINEAR baseline near the step -- it didn't actually reduce
		// the clamp's contribution there at all, and made maxToeStep slightly WORSE
		// (0.135 m). The plain "raw terrain height, no added clearance" version here
		// is both simpler and empirically the best of the three: zArc alone already
		// keeps within ~2.8 cm of terrain on that same problem event (verified via a
		// standalone probe), so a bare non-penetration floor (no redundant margin)
		// closes that small remaining gap without reintroducing a large jump.
		const clearance = onStairs ? p.swingClearanceClimb : p.swingClearance;
		const pathLen = _hyp2( to.x - from.x, to.y - from.y );
		const profileSteps = Math.max( 1, Math.ceil( pathLen / 0.02 ) );
		const clampProfile = new Float64Array( profileSteps + 1 ); // raw terrain height, running max
		// Seed with ONLY from.x's terrain -- NOT to.x's, even though apexZ (below)
		// needs the true overall max INCLUDING the endpoints: the loop's own LAST
		// iteration (k=profileSteps, u=1) already reaches xx=to.x exactly, so
		// pre-seeding with terrain.heightAt(to.x) here would leak the touchdown's
		// (possibly much higher, e.g. one tread up) terrain height into EARLY profile
		// entries before the geometric path has actually reached that x -- confirmed
		// as a real bug when first written: on a real tread0->tread1 climb event it
		// put tread1's height at clampProfile[1] even though the true crossing doesn't
		// happen until roughly HALFWAY through the path.
		let maxTerrainAlongPath = terrain.heightAt( from.x );
		clampProfile[ 0 ] = maxTerrainAlongPath;
		for ( let k = 1; k <= profileSteps; k ++ ) {

			const u = k / profileSteps;
			const xx = from.x + ( to.x - from.x ) * u;
			const hh = terrain.heightAt( xx );
			maxTerrainAlongPath = Math.max( maxTerrainAlongPath, hh );
			clampProfile[ k ] = maxTerrainAlongPath; // running max -> monotone non-decreasing

		}
		const apexZ = maxTerrainAlongPath + clearance;

		const event = {
			foot: bestFoot,
			tLift: s.t,
			tLand: s.t + swingDur,
			from, to, fromYaw, toYaw,
			apexZ, clearance, clampProfile,
		};

		events[ bestFoot ].push( event );
		active[ bestFoot ] = event;
		state[ bestFoot ].swinging = true;

	}

	// Derive the phase timeline from the finished event lists (see the comment on
	// phaseAtSampleIdx's declaration above for why this is a separate pass).
	_fillPhaseTimeline( phaseAtSampleIdx, samples, events );

	return {
		samples, events, phaseAtSampleIdx,
		params: p,
	};

}

/** Find the index of the first sample at or after time `tTarget`, searching forward from `fromIdx` (never before it -- the schedule builder only ever needs to look FORWARD in time, matching "never velocity-extrapolate, walk the actual array"). Clamps to the last sample if tTarget exceeds the array's range (an event whose touchdown would fall past the clip's end still resolves to a sane target: the clip's final recorded pose). */
function _findSampleAtOrAfter( samples, fromIdx, tTarget ) {

	const n = samples.length;
	for ( let i = fromIdx; i < n; i ++ ) {

		if ( samples[ i ].t >= tTarget ) return i;

	}

	return n - 1;

}

/** Second pass: derive a clean, monotone-between-events phase timeline from the final event list, independent of the (message, order-sensitive) inline attempt during the main march. Each foot's OWN sequence of events defines a strictly increasing sequence of (liftTime -> phaseValue) breakpoints (0, 1, 2, ... for that foot, offset 0 for left / 0.5 for right); this pass merges both feet's breakpoints by time and holds the most-recently-reached phase value constant between breakpoints, exactly matching poseAt's "phase advances 0.5 per step ... interpolates only during swings, frozen between" contract. */
function _fillPhaseTimeline( phaseAtSampleIdx, samples, events ) {

	const breakpoints = [];
	for ( const e of events.left ) breakpoints.push( { t: e.tLift, phase: null, foot: 'left' } );
	for ( const e of events.right ) breakpoints.push( { t: e.tLift, phase: null, foot: 'right' } );
	breakpoints.sort( ( a, b ) => a.t - b.t );

	let leftCount = 0, rightCount = 0;
	for ( const bp of breakpoints ) {

		if ( bp.foot === 'left' ) { bp.phase = leftCount; leftCount += 1; }
		else { bp.phase = 0.5 + rightCount; rightCount += 1; }

	}

	let bpIdx = 0;
	let currentPhase = 0.0; // both feet planted at their t=0 nominal -> phase 0 until the first event's lift sample
	const n = samples.length;
	for ( let i = 0; i < n; i ++ ) {

		const t = samples[ i ].t;
		while ( bpIdx < breakpoints.length && breakpoints[ bpIdx ].t <= t ) {

			currentPhase = breakpoints[ bpIdx ].phase;
			bpIdx ++;

		}

		phaseAtSampleIdx[ i ] = currentPhase;

	}

}

// ===========================================================================
// Stateless pose query
// ===========================================================================

/**
 * Pure, stateless, deterministic function of t: the viewer is scrub-driven, so random
 * access at ANY t (in ANY order, repeated) must return exactly the same result every
 * time -- no memoized "last event" pointer, no incremental state. Binary/linear-
 * searches `schedule.events[foot]` fresh each call (event lists are tiny -- a few
 * dozen per clip -- so a linear scan is not a performance concern and keeps this
 * function trivially auditable for the determinism contract).
 *
 * Returns `{ leftFoot, rightFoot, gaitPhase, speed, groundSlope, rootX, rootY, rootYaw }`
 * where each foot is `{x,y,z,yaw,planted,swingU}` in P-frame meters/radians.
 * `speed`: instantaneous root speed (m/s), central-difference from the sample array
 * around t -- consumed by PatientHuman's torso-lean gain and the diagnostic
 * idleFootMotion check (root speed < 0.02 m/s defines "idle").
 * `groundSlope`: dz/dx of the terrain under the root's current X (0 on flat ground,
 * step_height_m/step_depth_m while on the staircase) -- consumed by the torso-lean
 * slope gain (AGENTS.md-documented "6-8 deg while climbing" target).
 */
export function poseAt( schedule, terrain, t ) {

	const { samples } = schedule;
	const rootSample = _sampleRootAt( samples, t );

	const leftFoot = _footPoseAt( schedule, terrain, 'left', t, + 1 );
	const rightFoot = _footPoseAt( schedule, terrain, 'right', t, - 1 );

	const gaitPhase = _phaseAt( schedule, t );
	const speed = _speedAt( samples, t );
	const groundSlope = _slopeAt( terrain, rootSample.x );

	return {
		leftFoot, rightFoot, gaitPhase, speed, groundSlope,
		rootX: rootSample.x, rootY: rootSample.y, rootYaw: rootSample.yaw, rootZ: rootSample.zRoot,
	};

}

/** Linear-interpolated {x,y,zRoot,yaw} at t from the sample array (clamped at either end). Yaw is interpolated linearly on the already-unwrapped (continuous) values extractPathSamples produced, so no branch-cut handling is needed here. */
function _sampleRootAt( samples, t ) {

	const n = samples.length;
	if ( t <= samples[ 0 ].t ) return samples[ 0 ];
	if ( t >= samples[ n - 1 ].t ) return samples[ n - 1 ];

	// Binary search for the bracketing pair (samples are time-sorted).
	let lo = 0, hi = n - 1;
	while ( hi - lo > 1 ) {

		const mid = ( lo + hi ) >> 1;
		if ( samples[ mid ].t <= t ) lo = mid; else hi = mid;

	}

	const s0 = samples[ lo ], s1 = samples[ hi ];
	const frac = s1.t > s0.t ? ( t - s0.t ) / ( s1.t - s0.t ) : 0;
	return {
		t,
		x: s0.x + ( s1.x - s0.x ) * frac,
		y: s0.y + ( s1.y - s0.y ) * frac,
		zRoot: s0.zRoot + ( s1.zRoot - s0.zRoot ) * frac,
		yaw: s0.yaw + ( s1.yaw - s0.yaw ) * frac,
	};

}

/** Central-difference root speed (m/s) at t, using a small fixed dt probe into the (already dense, ~30fps-sampled) root timeline -- NOT a stored per-sample value, so it stays exact under scrub (arbitrary t), not just at baked sample times. */
function _speedAt( samples, t ) {

	const dt = 0.02;
	const a = _sampleRootAt( samples, t - dt );
	const b = _sampleRootAt( samples, t + dt );
	const dx = b.x - a.x, dy = b.y - a.y;
	const denom = Math.max( 1e-6, ( t + dt <= samples[ samples.length - 1 ].t ? dt : ( samples[ samples.length - 1 ].t - t ) )
		+ ( t - dt >= samples[ 0 ].t ? dt : ( t - samples[ 0 ].t ) ) );
	return _hyp2( dx, dy ) / denom;

}

/**
 * Ground slope (dimensionless rise/run) at world-X x: 0 on flat ground (before the
 * stairs or on the top landing), stepH/stepD (the staircase's own AVERAGE slope,
 * e.g. ~0.475 for this app's commercial-spec stairs) anywhere on the staircase.
 *
 * Deliberately analytic, NOT a numeric finite-difference of terrain.heightAt -- that
 * was tried first (central difference at a small +-2cm epsilon) and produces wildly
 * wrong spikes near any tread boundary, since heightAt is a genuine STEP function: a
 * +-2cm probe straddling a riser measures that riser's FULL 0.145 m rise over just
 * 0.04 m of run, i.e. slope~=3.6 (confirmed live: exactly this value observed via
 * poseAt's groundSlope field while stepping through a real climb clip) -- nowhere
 * close to the staircase's true ~0.475 average slope, and it only occurs in the
 * narrow bands right at each tread edge, so it doesn't even average out over a
 * clip. This feeds PatientHuman's torso-lean slope gain (leanSlopeK*groundSlope,
 * target ~6-8 deg while climbing per AGENTS.md) -- a per-riser-edge spike there
 * would read as a jarring per-step lean lurch instead of the intended smooth,
 * sustained climbing lean.
 */
function _slopeAt( terrain, x ) {

	if ( x < terrain.startX || x >= terrain.topX ) return 0.0;
	return terrain.stepH / terrain.stepD;

}

/** Gait phase at t: holds the schedule's baked phaseAtSampleIdx timeline value for the sample bracketing t (piecewise-constant lookup, matching "frozen between events" -- see _fillPhaseTimeline). Interpolation happens implicitly via poseAt's swingU (per-foot), NOT by interpolating this scalar -- gaitPhase itself is a discrete "which beat are we on" counter used by PatientHuman for the anchor bob / canned-clip phase-lock, both of which want a value that jumps at each footfall and holds still between them (a smoothly-interpolated phase would make the bob/upper-body motion drift continuously even while genuinely idle, reintroducing exactly the "moving while stopped" bug this module exists to eliminate). */
function _phaseAt( schedule, t ) {

	const { samples, phaseAtSampleIdx } = schedule;
	const n = samples.length;
	if ( t <= samples[ 0 ].t ) return phaseAtSampleIdx[ 0 ];
	if ( t >= samples[ n - 1 ].t ) return phaseAtSampleIdx[ n - 1 ];

	let lo = 0, hi = n - 1;
	while ( hi - lo > 1 ) {

		const mid = ( lo + hi ) >> 1;
		if ( samples[ mid ].t <= t ) lo = mid; else hi = mid;

	}

	return phaseAtSampleIdx[ lo ];

}

/**
 * Pose one foot at time t: if t falls within one of this foot's scheduled swing
 * windows [tLift, tLand], blend; otherwise the foot is PLANTED at whichever event's
 * `to` position is the most recent one at or before t (or the schedule's initial
 * nominal, before this foot's first-ever event).
 *
 * Planted feet are EXACTLY the event's baked position/yaw -- zero drift by
 * construction (there is no "continue simulating a planted foot" code path at all;
 * it's a constant lookup), which is what makes `plantedDriftMax` and
 * `idleFootMotionMax` structurally zero rather than "tuned to be small".
 */
function _footPoseAt( schedule, terrain, foot, t, sign ) {

	const evs = schedule.events[ foot ];

	// Find a swing window containing t (evs is time-sorted and non-overlapping for a
	// single foot by construction -- buildSchedule never starts a new swing for a foot
	// still marked swinging).
	for ( let i = 0; i < evs.length; i ++ ) {

		const e = evs[ i ];
		if ( t >= e.tLift && t < e.tLand ) {

			const u = e.tLand > e.tLift ? ( t - e.tLift ) / ( e.tLand - e.tLift ) : 1.0;
			const ease = _smoothstep( u );

			const x = e.from.x + ( e.to.x - e.from.x ) * ease;
			const y = e.from.y + ( e.to.y - e.from.y ) * ease;
			const zEndpointBlend = e.from.z + ( e.to.z - e.from.z ) * ease;
			// Arc bump: zero at u=0 and u=1 (a plain sine half-arch), added on top of
			// the endpoint-exact blend so touchdown/liftoff are always pop-free
			// regardless of the arc's own amplitude.
			const arcBump = Math.max( 0.0, e.apexZ - Math.max( e.from.z, e.to.z ) ) * Math.sin( Math.PI * u );
			const zArc = zEndpointBlend + arcBump;

			// Terrain clamp: a strict, DEFENSIVE non-penetration floor only -- NOT a
			// second source of clearance margin. e.clampProfile stores the raw terrain
			// height (running max, precomputed at build time -- see its own comment
			// for why it must NOT be a live terrain.heightAt(x) query: the terrain
			// function's own discrete tread steps would leak straight into the clamp
			// floor, producing a single-sample pop right at a tread boundary), which
			// this clamps against DIRECTLY -- no added `+ clearance*sin(pi*u)` term
			// here, because zArc's OWN arc bump already provides that clearance (its
			// coefficient is `apexZ - max(from.z,to.z)`, and apexZ already bakes in
			// `+ clearance` over the path's highest terrain point -- see apexZ's build-
			// time comment for the two double-counting formulations that were tried
			// and rejected before landing on this one). This clamp only ever needs to
			// correct the small residual gap zArc's smooth blend can still leave below
			// terrain right at a boundary crossing (empirically a few cm, not the
			// clearance margin's own full amplitude).
			//
			// Indexed by EASE, not u: clampProfile was built as a function of LINEAR
			// horizontal path fraction (buildSchedule's own build loop samples the
			// straight liftoff->touchdown line at k/profileSteps), and x/y above are
			// placed at `ease` (the SMOOTHSTEP-eased fraction), not raw `u` -- ease and
			// u diverge substantially away from the swing's midpoint (smoothstep starts
			// and ends slower than linear, e.g. ease(0.111)~=0.034, roughly a third of
			// u). Indexing the profile by u instead of ease (tried first) looks up the
			// terrain far AHEAD of where the foot horizontally actually is early/late
			// in the swing, which can hit a tread-boundary rise before the foot's own x
			// has actually crossed it -- confirmed as a regression: it turned an
			// earlier u-indexed version's single-sample z pop into a WORSE one.
			//
			// CEILING lookup, not linear interpolation: clampProfile is a running max
			// of a genuine STEP function (the analytic terrain height), so two
			// adjacent samples can be DIFFERENT PLATEAUS (e.g. 0 then 0.145, with the
			// true step boundary sitting somewhere strictly between their x
			// positions) -- linearly interpolating between them produces intermediate
			// values the real terrain never actually takes on, which can read as safe
			// while the foot's true x has ALREADY crossed the step and the real
			// terrain there is the full higher plateau. Confirmed as a real bug: on
			// the same real "climb straight onto tread 0" event used above, linear
			// interpolation left a 0.0247 m penetration right after start_x (x
			// slightly past the step, terrain already 0.145, but the interpolated
			// clamp floor was still ~0.12). Rounding UP to the next profile sample
			// (the conservative/higher neighbour) instead guarantees the clamp floor
			// is always >= the true terrain height anywhere within the segment it
			// covers -- verified against the same event: 0 m penetration, and the
			// resulting per-dt=0.05-sample delta (0.062 m) stays comfortably under
			// the 0.12 m acceptance bar (the floor is now a piecewise-CONSTANT step
			// in ease-space rather than a smooth ramp, but zArc's own smooth blend is
			// what's actually visible for almost the whole swing -- the clamp only
			// ever WINS the max() right at the crossing, briefly).
			const profile = e.clampProfile;
			const profileIdxF = ease * ( profile.length - 1 );
			const profileIdxCeil = Math.min( profile.length - 1, Math.ceil( profileIdxF - 1e-9 ) );
			const clampFloor = profile[ profileIdxCeil ];
			const z = Math.max( zArc, clampFloor );

			const yaw = e.fromYaw + _angleDiff( e.toYaw, e.fromYaw ) * ease;

			return { x, y, z, yaw, planted: false, swingU: u };

		}

	}

	// Not swinging: planted at the most recent event's `to` (or the schedule's initial
	// nominal if this foot has no events yet at/before t).
	let lastLanded = null;
	for ( let i = 0; i < evs.length; i ++ ) {

		if ( evs[ i ].tLand <= t ) lastLanded = evs[ i ]; else break;

	}

	if ( lastLanded ) {

		return { x: lastLanded.to.x, y: lastLanded.to.y, z: lastLanded.to.z, yaw: lastLanded.toYaw, planted: true, swingU: null };

	}

	// Before this foot's first event (or the clip has none): planted at the initial
	// nominal computed from the FIRST sample (matches buildSchedule's own initial
	// state, so t=0 querying before any event is scheduled is consistent with what
	// buildSchedule assumed as its starting condition).
	const s0 = schedule.samples[ 0 ];
	const nominal = _nominalAt( s0, sign, schedule.params.footLateral, terrain );
	return { x: nominal.x, y: nominal.y, z: nominal.z, yaw: s0.yaw, planted: true, swingU: null };

}
