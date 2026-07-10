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
	swingClearanceClimb: 0.14, // m -- taller margin on stairs so the swinging foot clears a riser nosing, not just the tread top. F1 (integration_2.json diag, 2026-07-10) residual: widened 0.10->0.14 -- this profile clamps the ANKLE's own path (buildSchedule's from/to are ankle-level nominal points), but the diag measures the TOE bone, which leads the ankle horizontally by toeForwardLenM during a fast-advancing climb swing, so it can reach a tread's higher terrain slightly BEFORE the ankle-based clamp profile has climbed to match -- measured worst-case penetration during a mid-swing sample (climb t=20.1s, swingU=0.28) dropped 0.0351->0.0233 m with this widen (~34%). Does NOT fully eliminate the residual (0.0233 m still exceeds the -0.002 m bar) -- the remaining gap is a DIFFERENT, margin-independent mechanism (see PatientHuman.js's own _footRollPitch/heel-strike-pivot residual, same order of magnitude, present on FLAT ground too where this constant doesn't even apply) -- a proper fix needs a toe-aware swing clamp or an incident-#14-style full analytic re-derivation of the toe's real (not just ankle's) swing-time position, out of this pass's scope; left as a documented residual, this widen is a genuine partial improvement with no observed downside (Node audit stays green).
	minEventGap: 0.15, // s -- minimum time between one foot's swing ENDING and the SAME foot starting another (prevents rapid double-triggers)
	yawErrorWeight: 0.30, // m/rad -- converts a plant-vs-current yaw error into an equivalent "need" distance; tuned so ~25 deg (0.44 rad) of yaw error alone crosses stepTrigger (0.44*0.30=0.132, just under 0.16 -- combines with even a little translational need to trigger, matching "yaw alone eventually triggers, not instantly")
	idleSpeedThreshold: 0.02, // m/s -- SAME value the browser diagnostic (main.js patientDiag) uses to define "idle" for idleFootMotionMax; a swing may only START at a sample where root translational speed OR yaw rate clears its own idle floor (see buildSchedule) -- deliberately shared so "does a step trigger" and "does the diagnostic call this idle" can never disagree
	idleYawRateThreshold: 0.05, // rad/s -- companion to idleSpeedThreshold: an in-place turn (near-zero translational speed, real yaw rate) must still be able to trigger an adjustment step, so idleness requires BOTH speed and yaw-rate to be below their floors, not just speed alone
	idleSustainSamples: 3, // count -- the idle gate requires this many CONSECUTIVE trailing samples to all clear the idle floor (not just the trigger sample itself), so a trigger can't fire on the single leading-edge sample of a resume-from-stop, whose swing would otherwise still span mostly-idle samples just before it
	heelMargin: 0.05, // m -- keep the foot's heel/back edge this far from a tread's near (riser) edge
	nosingMargin: 0.05, // m -- keep the foot's toe this far from a tread's far (nosing) edge. F1 (integration_2.json diag, 2026-07-10): widened 0.03->0.05 -- the NOMINAL (flat-footed) toeForwardLen reach this margin is measured against assumes pitch=0, but a toe-off roll's REAL rendered toe (PatientHuman's roll model, pivoting the ANKLE about a toe contact point while the Foot->ToeBase offset itself also rotates through `pitch`) reaches further forward than that flat assumption by an amount that grows with roll angle -- measured empirically (climb clip, tread idx 2->3 boundary, t=3.25s) at ~0.0365 m beyond the flat-nominal toe position, i.e. the OLD 0.03 m margin was already fully consumed with 0.0065 m to spare in the wrong direction. This margin is shared by BOTH the pre-existing onStairs tread-to-tread clamp and F1's own startX base clamp, so widening it fixes both boundary classes with one tune. Does not touch heelForwardLen/heelMargin (0.05 m already had headroom; no matching heel-side violation was observed).
	bobAmplitude: 0.015, // m -- vertical anchor bob amplitude, phase-locked to gaitPhase (freezes when steps stop)
	footLateral: 0.09 * 0.75, // m -- half-stance-width (nominal foot lateral offset from the root). This default matches the OLD (now-retired) Python pipeline's anim_bake._PATIENT_LEG_HIP_OFFSET magnitude, kept only so this module stays usable standalone (Node tests, this file's own header) -- PatientHuman.buildGait() ALWAYS overrides this with the REAL measured hip-pivot lateral offset from Xbot's own bind pose (~0.082 m, close but not identical to this default) before building a schedule for the live app, exactly like toeForwardLen below
	toeForwardLen: 0.107, // m -- horizontal Foot->ToeBase reach, measured from Xbot's own bind pose (see PatientHuman.js's load-time measurement) -- default here is that measured value, duplicated so this module stays load-order-independent (PatientHuman passes the REAL measured value in at buildGait() time; this default only matters for standalone/Node testing)
	// Torso lean gains (consumed by PatientHuman.js, not this module -- kept here so
	// every gait-related tunable lives in one place): torsoPitch = clamp(leanBase +
	// leanSpeedK*speed + leanSlopeK*groundSlope, 0, 0.15) rad.
	leanBase: 0.02,
	leanSpeedK: 0.05,
	leanSlopeK: 0.55,
	// "Walk-on" tail (see _buildWalkOn): after the recorded gait stops stepping, carry
	// the patient a few steps further FORWARD along its own facing, then stand still.
	// Needed because the recorded root keeps GLIDING ~1.5 m across the top landing after
	// the last real footfall (the climbing robot ends up right on top of the patient's
	// frozen spot -- these steps move the patient clear of it). The advance is a smooth
	// straight glide the synthesized feet track, so the hip stays over the feet the
	// whole time (reach-safe, no recline) and the subsequent freeze is pop-free.
	walkOnSteps: 3, // number of forward steps to synthesize
	walkOnAdvance: 0.55, // m the hip travels forward over those steps
	walkOnStepDur: 0.65, // s per step (swing+stance); total walk-on time = walkOnSteps*this
	walkOnFootAhead: 0.13, // m each footfall lands ahead of the hip (a natural stride reach)
	walkOnMinRecordedForward: 0.30, // m -- only walk on if the recorded path itself still travels at least this far forward past the last footfall (skips the follow clip's negligible ~0.1 m tail)

	// -- v2 additions (IK_OVERHAUL_SPEC.md gait-realism overhaul) --------------------

	// G1 predictive trigger (buildSchedule's need computation, see its own comment):
	// ~0.6*swingDur (0.6*0.32=0.192, rounded) -- long enough to actually forecast past
	// the current instant, short enough to stay a genuine near-term forecast rather
	// than reaching towards a whole extra swing away.
	predictLeadSec: 0.19, // s -- how far ahead buildSchedule forecasts drift when evaluating a step trigger (sampled off the real array, never extrapolated -- see _findSampleAtOrAfter)

	// G2 speed-adaptive swing duration (_speedAdaptiveSwingDur): swingDur_eff =
	// clamp(swingDur * (refSpeedMps/max(speedAtTrigger,swingSpeedFloorMps))^0.25,
	// swingDur, swingDurSlowMax) -- a slower root takes slower, more deliberate steps.
	refSpeedMps: 0.4, // m/s -- root speed at which swingDur/swingDurClimb apply UNSCALED
	swingSpeedFloorMps: 0.05, // m/s -- floor under the measured trigger-time speed before it divides refSpeedMps (prevents the ratio blowing up as speed->0; the idle gates, not this floor, are what keep a genuinely-stopped root from triggering at all)
	swingDurSlowMax: 0.55, // s -- flat-ground cap on the speed-scaled swing duration
	swingDurSlowMaxClimb: 0.70, // s -- stair cap on the speed-scaled swing duration

	// G3 out-toeing -- applied ONLY at poseAt/_footPoseAt time (display yaw), never to
	// buildSchedule's own trigger/need math -- see _footPoseAt's comment for why, and
	// for the +Y=left/-Y=right sign derivation this default relies on.
	outToeRad: 0.10, // rad -- plant/swing yaw offset; LEFT foot gets +outToeRad, RIGHT gets -outToeRad (toes splay away from the midline)

	// G4 stance width -- buildSchedule adds this to p.footLateral EXACTLY ONCE, right
	// after params are merged (see buildSchedule's own comment there); every nominal/
	// plant lookup in this file already reads the (now-widened) p.footLateral
	// afterward, so no separate "effective width" constant is threaded elsewhere.
	stanceWidenM: 0.012, // m -- elderly slightly-wider-than-rig-measured stance, added to footLateral

	// G6 support (lateral weight-transfer signal, see _supportAt).
	supportEaseSec: 0.35, // s -- smoothstep duration easing `support` from +-1 (value at a swing's touchdown) back toward 0 (centered double support) while planted

	// G5 cane schedule (see _buildCaneEvents). caneEnabled defaults true; a caller sets
	// it false only if the rig layer genuinely cannot build/attach a cane (I10
	// safe-disable contract -- _buildCaneEvents logs once when that happens).
	caneEnabled: true,
	caneLeadSec: 0.08, // s -- cane liftoff leads its associated LEFT-foot event's own tLift by this much
	caneSwingDur: 0.30, // s -- cane's own (shorter) swing duration -- it must finish planting before its foot, see the tLand clamp in _buildCaneEvents
	caneForwardM: 0.18, // m -- tip target offset, forward of the root at the cane's own landing time
	caneLateralM: 0.32, // m -- tip target offset, to the RIGHT of the root (cane is always held in the right hand). F7 (integration_2.json diag, 2026-07-10): widened 0.28->0.32, M12_caneShaftClearanceMin measured 0.0065 m against a 0.03 m bar (shaft passing too close to the shin) -- +0.04 m lateral moves the whole shaft further from the leg; see this rewrite's own report for the re-measured landed value.
	caneClearanceM: 0.05, // m -- vertical swing-arc clearance margin (same role as swingClearance for feet)
	caneTreadMarginM: 0.02, // m -- point-footprint margin kept inside a tread's near/far edge when the tip snaps onto stairs (a cane tip is a point, so -- unlike heelMargin/nosingMargin's asymmetric foot-length margins -- both edges share this one small constant)
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

/**
 * I10 shared swing-finishing helper (IK_OVERHAUL_SPEC.md section 4 G-refactor):
 * build the apex height + terrain non-penetration clamp profile for a
 * liftoff->touchdown segment. Used by ALL swing producers in this file -- the
 * main march (feet), _buildWalkOn, and _buildCaneEvents -- so there is exactly
 * ONE implementation of this math, not three copies that could drift.
 *
 * clampProfile stores the RAW terrain height (running max, so it stays
 * monotone non-decreasing -> smooth to interpolate), sampled at <=2cm spacing
 * along the straight liftoff->touchdown line. The clamp that consumes this
 * (_footPoseAt/_canePoseAt) uses it as a strict non-penetration FLOOR ONLY --
 * it does NOT separately add the clearance margin (see _footPoseAt:
 * `z = max(zArc, terrainProfile(ease))`, no extra `+ clearance*sin(pi*u)` term
 * on the clamp side). Clearance is already fully provided by zArc's OWN arc
 * bump (its coefficient is `apexZ - max(from.z,to.z)`, and
 * `apexZ = maxTerrainAlongPath + clearance` -- i.e. the arc already peaks
 * `clearance` above the highest terrain along the path), so the clamp's only
 * remaining job is "never actually go below ground" -- a strictly weaker,
 * purely defensive condition. Two prior formulations were tried and rejected:
 * (1) clamping against `rawTerrain + clearance*sin(pi*u)` DOUBLE-counted
 * clearance on top of the arc bump's own, compounding right where both were
 * steepest (0.1275 m single-dt=0.05-sample jump on a real single-riser "climb
 * straight to touchdown" event, maxToeStep 0.131 m against the 0.12 m bar);
 * (2) storing the EXCESS over a straight-line reference between the
 * endpoints' own terrain heights (rather than the raw terrain height) was
 * meant to zero out the clamp for that same common case, but a discrete STEP
 * terrain function is "all excess" relative to any LINEAR baseline near the
 * step -- it didn't actually reduce the clamp's contribution there at all,
 * and made maxToeStep slightly WORSE (0.135 m). The plain "raw terrain
 * height, no added clearance" version here is both simpler and empirically
 * the best of the three: zArc alone already keeps within ~2.8 cm of terrain
 * on that same problem event (verified via a standalone probe), so a bare
 * non-penetration floor (no redundant margin) closes that small remaining gap
 * without reintroducing a large jump.
 */
function _buildSwingProfile( from, to, terrain, clearance ) {

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

	return { apexZ, clampProfile };

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
	// G4 stance width (IK_OVERHAUL_SPEC.md section 4): widen the nominal half-stance
	// by stanceWidenM ONCE, right here, so every downstream nominal/plant lookup in
	// this file (which all read p.footLateral, never a separate "effective width"
	// constant) picks up the widened value automatically -- an elderly gait stands
	// slightly wider than the rig's own measured hip-pivot offset.
	p.footLateral = p.footLateral + p.stanceWidenM;
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
			if ( _rootNearIdleAtIndex( samples, j, p.idleSpeedThreshold, p.idleYawRateThreshold ) ) { rootIsIdle = true; break; }

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
		// G1 predictive trigger (IK_OVERHAUL_SPEC.md section 4): forecast drift/yaw-
		// error at a near-future sample too, so a step can fire slightly BEFORE the
		// INSTANTANEOUS need alone would cross threshold -- this is what shrinks the
		// "glide" phase (both feet planted while the root has already drifted most of
		// the way to triggering, P3) without weakening the trigger threshold itself.
		// Sampled off the REAL future array entry via _findSampleAtOrAfter, searching
		// forward from the CURRENT index i -- never velocity-extrapolated (matches
		// AGENTS.md incident #6's "decide from an actual snapshot, don't
		// recompute-live-forward" discipline, and the touchdown-resolution walk just
		// below it); clamps at the array's end automatically, so a lead window that
		// overruns the clip's remaining samples just degrades to the last available
		// sample, never an out-of-range read or an extrapolated guess. Foot-
		// independent (only depends on the current sample index/time), so resolved
		// once here rather than inside the per-foot loop below.
		const predIdx = _findSampleAtOrAfter( samples, i, s.t + p.predictLeadSec );
		const sPred = samples[ predIdx ];
		// G2 speed-adaptive swing duration also reads the root speed AT this trigger
		// sample (see _speedAdaptiveSwingDur's call site below, after bestFoot is
		// chosen) -- computed once here via the SAME central-difference formula the
		// idle gate above uses (neighbor-sample difference, not poseAt's own fixed
		// +-0.02s probe), so "how fast is the root moving right now" can never
		// disagree between the idle gate and the swing-duration scaling.
		const speedAtTrigger = _speedAtIndex( samples, i );

		let bestFoot = null, bestNeed = - Infinity;

		for ( const foot of [ 'left', 'right' ] ) {

			if ( state[ foot ].swinging ) continue;
			if ( s.t - state[ foot ].lastEventEndT < p.minEventGap ) continue;
			// The OTHER foot must not currently be swinging (never both feet in the
			// air at once -- this is a walking gait, not a run).
			const other = foot === 'left' ? 'right' : 'left';
			if ( state[ other ].swinging ) continue;

			const planted = state[ foot ].plantedPos;

			const nominal = _nominalAt( s, sides[ foot ].sign, p.footLateral, terrain );
			const drift = _hyp2( nominal.x - planted.x, nominal.y - planted.y );
			const yawErr = Math.abs( _angleDiff( s.yaw, state[ foot ].plantYaw ) );
			const needNow = drift + p.yawErrorWeight * yawErr;

			// Same drift+yaw formula, against the SAME held `planted` position, but
			// measured at the forecast sample instead of the current one.
			const nominalPred = _nominalAt( sPred, sides[ foot ].sign, p.footLateral, terrain );
			const driftPred = _hyp2( nominalPred.x - planted.x, nominalPred.y - planted.y );
			const yawErrPred = Math.abs( _angleDiff( sPred.yaw, state[ foot ].plantYaw ) );
			const needPred = driftPred + p.yawErrorWeight * yawErrPred;

			const need = Math.max( needNow, needPred );

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
		// G2 speed-adaptive swing duration (IK_OVERHAUL_SPEC.md section 4,
		// _speedAdaptiveSwingDur): a slower root takes slower, more deliberate steps.
		// speedAtTrigger is fixed (computed once above, at the trigger sample) for
		// BOTH the flat and climb duration below -- only the base/cap PAIR flips when
		// the touchdown context resolves to stairs, mirroring the existing "at most
		// one extra forward-walk" re-resolution pattern.
		let swingDur = _speedAdaptiveSwingDur( p.swingDur, p.swingDurSlowMax, speedAtTrigger, p );
		let touchdownSampleIdx = _findSampleAtOrAfter( samples, i, s.t + swingDur );
		let touchdownSample = samples[ touchdownSampleIdx ];
		let touchdownNominal = _nominalAt( touchdownSample, sides[ bestFoot ].sign, p.footLateral, terrain );

		let touchdownOnStairs = terrain.treadIndexAt( touchdownNominal.x ) >= 0 && terrain.treadIndexAt( touchdownNominal.x ) < terrain.stepCount;
		if ( touchdownOnStairs ) {

			swingDur = _speedAdaptiveSwingDur( p.swingDurClimb, p.swingDurSlowMaxClimb, speedAtTrigger, p );
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
		// non-trivial amount) over the WHOLE prospective [liftoff, touchdown] window --
		// UNLESS that window is demonstrably a real, sustained walk throughout (see the
		// two-condition gate below). If it wouldn't (and isn't), defer -- don't fire
		// this sample; the still-growing need is simply re-evaluated at the next sample
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
		//
		// CORRECTED 2026-07-10 (orchestrator, IK_OVERHAUL_SPEC.md sections 4/8): the
		// original single-condition gate (`windowMotion < requiredTrigger * 0.5`)
		// conflates SLOW-BUT-SUSTAINED motion with DECELERATING-INTO-A-STOP -- both
		// produce a small liftoff-to-touchdown ENDPOINT distance, but only the latter is
		// what this gate exists to catch. Measured on the real "climb" clip's
		// top-landing stretch: a steady ~0.17 m/s walk sustained for 3+ seconds gives
		// windowMotion ~= 0.17*swingDur_eff ~= 0.068, permanently BELOW
		// requiredTrigger*0.5 (0.16*0.5=0.08 on flat ground) even though the root never
		// stops -- zero steps ever fired, freezing both feet while the root crept
		// 0.5+ m (an M6 max-root-travel-while-planted violation). Fix: only defer when
		// the window ALSO contains a near-idle sample -- i.e. the low windowMotion must
		// be explained by an actual stop/near-stop somewhere inside
		// [liftoff sample i, touchdownSampleIdx], not just by the root moving slowly
		// throughout. windowHasNearIdleSample reuses _rootNearIdleAtIndex, the SAME
		// central-difference construction the rootIsIdle sustain gate above calls
		// (looser 2x thresholds here -- this scan only needs to catch a genuine
		// stop/near-stop somewhere across a multi-sample window, not gate the trigger
		// instant itself the way rootIsIdle does). Net effect: a steady slow walk
		// (every sample in the window clears 2x the idle floors) now steps normally;
		// the t~=33.17s stop-and-go window above (which necessarily drops under the
		// idle floor at its stopped end) still gets deferred exactly as before.
		const windowDx = touchdownSample.x - s.x, windowDy = touchdownSample.y - s.y;
		const windowDist = _hyp2( windowDx, windowDy );
		const windowYawErr = Math.abs( _angleDiff( touchdownSample.yaw, s.yaw ) );
		const windowMotion = windowDist + p.yawErrorWeight * windowYawErr;
		let windowHasNearIdleSample = false;
		for ( let j = i; j <= touchdownSampleIdx; j ++ ) {

			if ( _rootNearIdleAtIndex( samples, j, 2 * p.idleSpeedThreshold, 2 * p.idleYawRateThreshold ) ) { windowHasNearIdleSample = true; break; }

		}
		if ( windowMotion < requiredTrigger * 0.5 && windowHasNearIdleSample ) continue;

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

		} else if ( terrain.treadIndexAt( toX ) < 0 && toX < terrain.startX && toX + p.toeForwardLen + p.nosingMargin > terrain.startX ) {

			// F1 (integration_2.json diag, 2026-07-10): a plant resolved to FLAT ground
			// just short of the staircase BASE still reaches, toe-first, past
			// terrain.startX -- heightAt() jumps 0 -> stepH the instant x crosses
			// startX (buildTerrain's own doc: no ramp, a genuine discontinuous riser),
			// so a toe that pokes even a few mm past startX gets compared against a
			// full tread-top height while the foot is actually resting on the flat
			// floor at z=0 -- exactly the "penetration reads a full riser" signature
			// (diag cluster: follow t~=19.7-22.1s, penetration.rightToe/leftToe up to
			// 0.167 m, plateauing at ~0.145 m == stepH while the stance holds). Clamp
			// the plant back so the toe stops AT the base, mirroring the onStairs
			// branch's own margin logic (toeForwardLen+nosingMargin) just measured
			// from the OPPOSITE edge (the staircase's near/start face instead of a
			// tread's far/nosing edge).
			toX = terrain.startX - p.toeForwardLen - p.nosingMargin;
			toZ = terrain.heightAt( toX );

		}

		const from = { x: state[ bestFoot ].plantedPos.x, y: state[ bestFoot ].plantedPos.y, z: state[ bestFoot ].plantedPos.z };
		const to = { x: toX, y: toY, z: toZ };
		const fromYaw = state[ bestFoot ].plantYaw;
		const toYaw = touchdownSample.yaw;

		// Apex/clamp data along liftoff->touchdown ("decide once and hold" --
		// AGENTS.md incident #6): see _buildSwingProfile's own docstring above for
		// the full clampProfile/apexZ design rationale (shared by every swing
		// producer in this file -- this march, _buildWalkOn, _buildCaneEvents -- so
		// there is exactly one copy of that reasoning, not several).
		const clearance = onStairs ? p.swingClearanceClimb : p.swingClearance;
		const { apexZ, clampProfile } = _buildSwingProfile( from, to, terrain, clearance );

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

	// Synthesize the forward "walk-on" tail (a few steps ahead of the last real footfall,
	// then stand still -- see _buildWalkOn / the walkOn* params). Appends its steps to
	// `events` BEFORE the phase timeline is derived so the anchor bob + upper-body walk
	// clip animate through the walk-on and freeze with it.
	const tail = _buildWalkOn( samples, events, terrain, p );

	// G5 cane schedule (IK_OVERHAUL_SPEC.md section 5, _buildCaneEvents): built from
	// the FINISHED events.left (main march + walk-on), so every left-foot step,
	// including the synthesized walk-on tail, gets a paired cane event. null when
	// params.caneEnabled is false (I10 safe-disable contract -- _buildCaneEvents
	// itself logs once when that happens).
	const caneEvents = _buildCaneEvents( samples, events, terrain, p );

	// Derive the phase timeline from the finished event lists (see the comment on
	// phaseAtSampleIdx's declaration above for why this is a separate pass).
	_fillPhaseTimeline( phaseAtSampleIdx, samples, events );

	// G6 phaseC/support (IK_OVERHAUL_SPEC.md section 3): both read a SINGLE merged,
	// tLift-sorted array of every foot event (main march + walk-on, both feet) --
	// built ONCE here (buildSchedule is the one allowed stateful/forward-marching
	// pass in this module, per this file's own header) so poseAt's phaseC/support
	// lookups (_phaseCAt/_supportAt) stay pure functions of (schedule, t), never
	// re-merging/re-sorting per query. Each event object already carries its own
	// `foot` field (set at construction in both the main march and _buildWalkOn), so
	// no re-tagging is needed here, just a merge + time sort.
	const mergedFootEvents = [ ...events.left, ...events.right ].sort( ( a, b ) => a.tLift - b.tLift );

	return {
		samples, events, phaseAtSampleIdx, mergedFootEvents, caneEvents,
		params: p,
		tail,
	};

}

/**
 * Build the forward "walk-on" tail and APPEND its steps to `events`. Returns a `tail`
 * descriptor `{ startT, freezeT, rootStart, rootEnd }` (rootStart/rootEnd are P-frame
 * {x,y,zRoot,yaw} root poses) that PatientHuman.sync() uses to drive the anchor: track
 * the recorded root up to `startT`, glide it straight from rootStart->rootEnd across
 * [startT, freezeT] (the patient stepping forward), then hold it frozen at rootEnd.
 *
 * WHY (product ask, 2026-07-08): the recorded root keeps GLIDING forward for ~13 s /
 * ~1.5 m after the patient's last real footfall (on the climb clip), and the climbing
 * robot ends up right on top of the patient's frozen spot. Rather than chase the
 * recorded track (its slow, pausing pace won't pass the gait's own step-commit gates,
 * and letting the hip glide while the feet stay planted is exactly the recline bug this
 * tail replaces), synthesize a short, deliberate forward walk: a smooth straight hip
 * glide of `walkOnAdvance` metres that `walkOnSteps` synthesized footfalls track, each
 * landing `walkOnFootAhead` ahead of the gliding hip. Because hip and feet advance
 * together the leg reach stays bounded (no recline/over-reach) and, since rootEnd is
 * where both the glide and the final footfall end up, the freeze into rootEnd is
 * pop-free. Skipped (startT === freezeT, a zero-length tail = "freeze in place at the
 * last footfall") when the recorded path doesn't itself continue forward far enough to
 * justify it -- e.g. the follow clip's ~0.1 m tail.
 *
 * The top landing is flat, so these are plain flat-ground steps (heightAt is constant
 * there); the code still queries terrain.heightAt per foot so it degrades sanely if a
 * future clip's walk-on ever started before the terrain leveled off.
 */
function _buildWalkOn( samples, events, terrain, p ) {

	const n = samples.length;
	const lastEv = ( f ) => ( events[ f ].length ? events[ f ][ events[ f ].length - 1 ] : null );

	const startT = Math.max(
		events.left.length ? events.left[ events.left.length - 1 ].tLand : samples[ 0 ].t,
		events.right.length ? events.right[ events.right.length - 1 ].tLand : samples[ 0 ].t,
	);

	const rootStart = _sampleRootAt( samples, startT ); // {t,x,y,zRoot,yaw}
	const yaw = rootStart.yaw;
	const fwdX = Math.cos( yaw ), fwdY = Math.sin( yaw );
	const latX = - Math.sin( yaw ), latY = Math.cos( yaw );

	// How far forward (along the current facing) the recorded root itself still travels
	// after startT. If it barely moves, there's nothing to walk onto -- return a
	// zero-length tail so sync() simply freezes in place at the last footfall.
	const rootEndRec = samples[ n - 1 ];
	const recordedForward = ( rootEndRec.x - rootStart.x ) * fwdX + ( rootEndRec.y - rootStart.y ) * fwdY;

	const noTail = {
		startT, freezeT: startT,
		rootStart,
		rootEnd: { x: rootStart.x, y: rootStart.y, zRoot: rootStart.zRoot, yaw },
	};
	if ( recordedForward < p.walkOnMinRecordedForward ) return noTail;

	// Never walk past where the recorded patient actually went (leave a small margin).
	const advance = Math.min( p.walkOnAdvance, recordedForward - 0.1 );
	if ( advance <= 0 ) return noTail;

	const nSteps = Math.max( 1, Math.round( p.walkOnSteps ) );
	const dur = nSteps * p.walkOnStepDur;
	const freezeT = startT + dur;

	const hipAt = ( frac ) => ( {
		x: rootStart.x + fwdX * advance * frac,
		y: rootStart.y + fwdY * advance * frac,
	} );
	const endHip = hipAt( 1 );
	const rootEnd = {
		x: endHip.x, y: endHip.y,
		zRoot: terrain.heightAt( endHip.x ) + PATIENT_HIP_HEIGHT_M,
		yaw,
	};

	// Current planted state per foot (start point of each foot's first synthesized swing).
	const plant = {};
	for ( const f of [ 'left', 'right' ] ) {

		const e = lastEv( f );
		if ( e ) {

			plant[ f ] = { x: e.to.x, y: e.to.y, z: e.to.z, yaw: e.toYaw, land: e.tLand };

		} else {

			const nm = _nominalAt( samples[ 0 ], f === 'left' ? + 1 : - 1, p.footLateral, terrain );
			plant[ f ] = { x: nm.x, y: nm.y, z: nm.z, yaw: samples[ 0 ].yaw, land: samples[ 0 ].t };

		}

	}

	// Step the foot that has been planted longer (landed earlier) first.
	let foot = plant.left.land <= plant.right.land ? 'left' : 'right';

	for ( let k = 1; k <= nSteps; k ++ ) {

		const sign = foot === 'left' ? + 1 : - 1;
		const tLand = startT + k * p.walkOnStepDur;
		const tLift = Math.max( plant[ foot ].land + 1e-3, tLand - p.swingDur );

		const hip = hipAt( k / nSteps );
		// Footfall lands walkOnFootAhead ahead of the hip, offset to this foot's side.
		// F1 (2026-07-10 diag) note: NOT mirroring the main march's startX base-clamp
		// here -- this tail only ever runs once the recorded root has already reached
		// its OWN last real footfall (startT = max of both feet's last tLand), and
		// walkOnMinRecordedForward (0.30 m) additionally skips it entirely on the
		// follow clip (whose ~0.1 m tail never reaches the stair base at all, see
		// this function's own header). On climb, that last real footfall is already
		// past the staircase (climb is complete by the time feet stop stepping), so
		// this synthesized tail walks forward on the flat TOP landing, never anywhere
		// near terrain.startX -- the base-clamp guard would be dead code here.
		const toX = hip.x + fwdX * p.walkOnFootAhead + latX * ( sign * p.footLateral );
		const toY = hip.y + fwdY * p.walkOnFootAhead + latY * ( sign * p.footLateral );
		const toZ = terrain.heightAt( toX );

		const from = { x: plant[ foot ].x, y: plant[ foot ].y, z: plant[ foot ].z };
		const to = { x: toX, y: toY, z: toZ };
		const clearance = p.swingClearance;
		// Same shared swing-profile helper as the main march + cane (_buildSwingProfile,
		// I10 refactor) rather than a bespoke 2-point shortcut. Equivalent here: from.z/
		// to.z are always freshly set FROM terrain.heightAt (toZ just above; the
		// previous iteration's plant[foot].z came from an earlier toZ the same way), so
		// the helper's from.x-seeded running-max scan reduces to the same effective
		// plateau(s) on this always-flat walk-on span, while staying correct (unlike a
		// bare endpoint check) if a future clip's walk-on ever started before the
		// terrain leveled off, crossing a tread boundary mid-path.
		const { apexZ, clampProfile } = _buildSwingProfile( from, to, terrain, clearance );

		events[ foot ].push( {
			foot, tLift, tLand, from, to,
			fromYaw: plant[ foot ].yaw, toYaw: yaw,
			apexZ, clearance, clampProfile,
		} );

		plant[ foot ] = { x: toX, y: toY, z: toZ, yaw, land: tLand };
		foot = foot === 'left' ? 'right' : 'left';

	}

	return { startT, freezeT, rootStart, rootEnd };

}

/**
 * G5 cane tip TARGET at a given root sample's pose (IK_OVERHAUL_SPEC.md section
 * 5): facing-frame offset, forward `caneForwardM` and to the RIGHT `caneLateralM`
 * (the cane is always held in the RIGHT hand), then stair-snapped the same way
 * feet are, except with a SYMMETRIC margin on both tread edges (caneTreadMarginM)
 * since a cane tip is a point -- unlike a foot's heel-to-toe footprint, there's no
 * asymmetric heelMargin/nosingMargin pair to reuse.
 *
 * Sign derivation (documented per this module's own "don't guess signs" discipline
 * -- see AGENTS.md incident #4's coordinate-reconciliation note): forward =
 * (cos yaw, sin yaw). This module's convention (DEFAULT_GAIT_PARAMS.outToeRad's
 * comment / _nominalAt: at yaw=0 facing +X, +Y is the LEFT side) means the LEFT
 * lateral unit vector is (-sin yaw, cos yaw) (see _nominalAt's own lateral-offset
 * formula) -- so RIGHT is that vector's negation: (sin yaw, -cos yaw). Shared by
 * the initial (t=0, before any cane event) nominal and every scheduled event's
 * touchdown target, so both use IDENTICAL placement math.
 */
function _caneTargetAt( rootSample, terrain, p ) {

	const yaw = rootSample.yaw;
	const cy = Math.cos( yaw ), sy = Math.sin( yaw );
	// forward=(cy,sy); right=(sy,-cy) -- see the function's own doc comment above.
	let x = rootSample.x + cy * p.caneForwardM + sy * p.caneLateralM;
	let y = rootSample.y + sy * p.caneForwardM - cy * p.caneLateralM;
	let z = terrain.heightAt( x );

	const treadIdx = terrain.treadIndexAt( x );
	if ( treadIdx >= 0 && treadIdx < terrain.stepCount ) {

		const span = terrain.treadSpan( treadIdx );
		const lo = span.xStart + p.caneTreadMarginM;
		const hi = span.xEnd - p.caneTreadMarginM;
		const clampedLo = Math.min( lo, hi );
		const clampedHi = Math.max( lo, hi );
		x = Math.min( clampedHi, Math.max( clampedLo, x ) );
		z = terrain.heightAt( x ); // EXACT tread-top height at the (possibly re-clamped) snapped X

	} else if ( x < terrain.startX && x + p.caneTreadMarginM > terrain.startX ) {

		// F1 point-margin mirror (2026-07-10 diag): same startX discontinuity as the
		// foot touchdown clamp above, but the cane tip is a POINT (no toeForwardLen
		// reach) -- caneTreadMarginM is the right (and only) margin to keep it off
		// the base riser's face.
		x = terrain.startX - p.caneTreadMarginM;
		z = terrain.heightAt( x );

	}

	return { x, y, z, yaw };

}

/**
 * G5 cane schedule (IK_OVERHAUL_SPEC.md section 5): synthesize one cane event per
 * LEFT-foot event (main march + walk-on, in the time order `events.left` is
 * already in -- see buildSchedule's own call site comment), so the cane advances
 * WITH (slightly leading) the contralateral left foot's swing, 3-point-pattern
 * style. Returns `null` (not an empty array) when caneEnabled is false, matching
 * poseAt's own `cane: null` contract -- and logs ONCE per the I10 safe-disable
 * contract (CLAUDE.md 8.8 / AGENTS.md: a "safe disable" guard must state what
 * feature is consequently off and why, not silently vanish).
 */
function _buildCaneEvents( samples, events, terrain, p ) {

	if ( ! p.caneEnabled ) {

		console.log( '[PatientGait] cane schedule disabled (params.caneEnabled=false) -- poseAt().cane will be null for this schedule.' );
		return null;

	}

	const leftEvents = events.left;
	const caneEvents = [];

	let plantedPos = _caneTargetAt( samples[ 0 ], terrain, p );
	let lastLandT = - Infinity;

	for ( let k = 0; k < leftEvents.length; k ++ ) {

		const leftEv = leftEvents[ k ];

		// Lead the paired left-foot liftoff by caneLeadSec, but never start before
		// the previous cane event has had a moment to finish (previous tLand + 0.05,
		// the same minEventGap-flavoured spacing the feet use) or before the clip's
		// own first sample.
		const tLift = Math.max( leftEv.tLift - p.caneLeadSec, lastLandT + 0.05, samples[ 0 ].t );
		// Finish planting no later than the paired foot (leftEv.tLand - 0.02), and
		// never longer than the cane's own (shorter) swing duration.
		const tLand = Math.min( leftEv.tLand - 0.02, tLift + p.caneSwingDur );

		if ( tLand - tLift < 0.05 ) continue; // degenerate window -- skip this step's cane event entirely

		const rootAtLand = _sampleRootAt( samples, tLand );
		const target = _caneTargetAt( rootAtLand, terrain, p );

		const from = { x: plantedPos.x, y: plantedPos.y, z: plantedPos.z };
		const to = { x: target.x, y: target.y, z: target.z };

		const { apexZ, clampProfile } = _buildSwingProfile( from, to, terrain, p.caneClearanceM );

		caneEvents.push( {
			foot: 'cane',
			tLift, tLand,
			from, to,
			fromYaw: plantedPos.yaw, toYaw: target.yaw,
			apexZ, clearance: p.caneClearanceM, clampProfile,
		} );

		plantedPos = target;
		lastLandT = tLand;

	}

	return caneEvents;

}

/** Find the index of the first sample at or after time `tTarget`, searching forward from `fromIdx` (never before it -- the schedule builder only ever needs to look FORWARD in time, matching "never velocity-extrapolate, walk the actual array"). Clamps to the last sample if tTarget exceeds the array's range (an event whose touchdown would fall past the clip's end still resolves to a sane target: the clip's final recorded pose). */
function _findSampleAtOrAfter( samples, fromIdx, tTarget ) {

	const n = samples.length;
	for ( let i = fromIdx; i < n; i ++ ) {

		if ( samples[ i ].t >= tTarget ) return i;

	}

	return n - 1;

}

/** Root "near-idle" test at sample index `j`: TRUE when BOTH the central-difference translational speed and yaw rate (neighbor-sample difference, clamped at the array ends -- a degenerate dt<=1e-6 neighbor pair returns false, "no evidence either way", matching the original inline rootIsIdle loop's own bare `continue`, never concluding idle from a zero-duration sample) are below the given thresholds. Parameterized on speedThreshold/yawRateThreshold so ONE formula backs both of buildSchedule's near-idle checks (both above -- earlier in this file's march loop): the rootIsIdle sustain gate (1x idleSpeedThreshold/idleYawRateThreshold) and the windowMotion "won't-actually-go-anywhere" gate's near-idle window scan (2x the same floors, see that gate's own comment) -- factored here so "is the root moving at sample j" can never quietly diverge between the two. */
function _rootNearIdleAtIndex( samples, j, speedThreshold, yawRateThreshold ) {

	const n = samples.length;
	const iPrev = Math.max( 0, j - 1 );
	const iNext = Math.min( n - 1, j + 1 );
	const sp = samples[ iPrev ], sn = samples[ iNext ];
	const dt = sn.t - sp.t;
	if ( dt <= 1e-6 ) return false;
	const spd = _hyp2( sn.x - sp.x, sn.y - sp.y ) / dt;
	const yr = Math.abs( sn.yaw - sp.yaw ) / dt;
	return spd < speedThreshold && yr < yawRateThreshold;

}

/** G2 root speed (m/s) at sample index `i`, via the SAME central-difference formula buildSchedule's own idle gate uses (neighbor-sample difference, clamped at the array ends) -- deliberately NOT poseAt's _speedAt (which probes a fixed +-0.02s window via interpolation): this one is queried by INDEX, at the trigger sample itself, so "how fast is the root moving right now" can never disagree between the idle gate and G2's swing-duration scaling. */
function _speedAtIndex( samples, i ) {

	const n = samples.length;
	const iPrev = Math.max( 0, i - 1 );
	const iNext = Math.min( n - 1, i + 1 );
	const sp = samples[ iPrev ], sn = samples[ iNext ];
	const dt = sn.t - sp.t;
	if ( dt <= 1e-6 ) return 0;
	return _hyp2( sn.x - sp.x, sn.y - sp.y ) / dt;

}

/**
 * G2 speed-adaptive swing duration (IK_OVERHAUL_SPEC.md section 4): a slower root
 * takes slower, more deliberate steps. `baseSwing`/`capForContext` are the
 * flat/climb swingDur and swingDurSlowMax pair (caller picks); the result is
 * never faster than `baseSwing` (only ever slowed down, floored at the base
 * value) and never slower than `capForContext`.
 */
function _speedAdaptiveSwingDur( baseSwing, capForContext, speedAtTrigger, p ) {

	const denom = Math.max( speedAtTrigger, p.swingSpeedFloorMps );
	const scaled = baseSwing * Math.pow( p.refSpeedMps / denom, 0.25 );
	return Math.min( capForContext, Math.max( baseSwing, scaled ) );

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
	// G6 phaseC/support (IK_OVERHAUL_SPEC.md section 3): both stateless lookups over
	// the SAME merged, tLift-sorted event array built once in buildSchedule (see
	// mergedFootEvents' own comment there) -- pure functions of (schedule, t), no
	// memoization, consistent with this module's scrub-safety contract.
	const phaseC = _phaseCAt( schedule.mergedFootEvents, t );
	const support = _supportAt( schedule.mergedFootEvents, t, schedule.params.supportEaseSec );
	const speed = _speedAt( samples, t );
	const groundSlope = _slopeAt( terrain, rootSample.x );
	// G5 cane (IK_OVERHAUL_SPEC.md section 3): null whenever caneEnabled was false at
	// buildSchedule time (schedule.caneEvents is null in that case too, see its own
	// comment) -- single source of truth, no separate params re-check needed here.
	const cane = schedule.caneEvents ? _canePoseAt( schedule, terrain, t ) : null;

	return {
		leftFoot, rightFoot, gaitPhase, phaseC, support, speed, groundSlope, cane,
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
 * G6 continuous phase (IK_OVERHAUL_SPEC.md section 3), CORRECTED CONTRACT
 * (2026-07-10): its OWN clean monotone counter, NOT required to equal legacy
 * gaitPhase at any point -- gaitPhase is NON-MONOTONE by construction (see this
 * file's header / _fillPhaseTimeline's own comment: left events get integer
 * phases and right half-integer, by PER-FOOT order, so when the right foot steps
 * first the merged sequence goes 0.5, 0.0, 1.5, 1.0, ... ), so no monotone signal
 * can match it at every boundary. Do NOT "fix" gaitPhase to make it match --
 * main.js depends on its current behavior.
 *
 * Definition: for the k-th event (0-based) in `merged` (ALL foot events -- main
 * march + walk-on, both feet -- tLift-sorted, built once in buildSchedule as
 * schedule.mergedFootEvents): phaseC(tLift)=0.5k, phaseC(tLand)=0.5(k+1), ramping
 * LINEARLY (matching poseAt's own swing progress u, NOT the smoothstep-eased
 * `ease` used for position/height) in between, frozen when t falls outside every
 * event's window. Well-defined because L/R swings never overlap (existing
 * invariant) -- at most one event's window contains any given t, and the merged
 * array's tLift order also orders tLand (no event starts before the previous one
 * lands, since that would require two feet swinging at once), so kCompleted (the
 * count of events already landed at/before t) is unambiguous from a forward scan.
 * Stateless: a pure function of (merged, t), no memoization.
 */
function _phaseCAt( merged, t ) {

	let kCompleted = 0;
	let progress = 0;

	for ( let i = 0; i < merged.length; i ++ ) {

		const e = merged[ i ];
		if ( e.tLand <= t ) { kCompleted ++; continue; }
		if ( t >= e.tLift && t < e.tLand ) {

			progress = e.tLand > e.tLift ? ( t - e.tLift ) / ( e.tLand - e.tLift ) : 1.0;

		}

	}

	return 0.5 * kCompleted + 0.5 * progress;

}

/**
 * G6 lateral weight-transfer signal (IK_OVERHAUL_SPEC.md section 3) in [-1, +1];
 * +1 = weight fully on the LEFT foot (i.e. during a RIGHT-foot swing, since the
 * right foot is off the ground), -1 = fully on the RIGHT (during a LEFT-foot
 * swing). Stateless over the SAME merged, tLift-sorted event array phaseC uses:
 * during an active swing the value is a flat step (the swinging foot's own sign);
 * after that swing's tLand it HOLDS, then eases (smoothstep) toward 0 (centered
 * double support) over `supportEaseSec`; 0 before the very first event. Because
 * L/R swings never overlap (see _phaseCAt's own comment), at most one event can
 * be "active" at any t, and the merged array's tLift order also orders tLand, so
 * a simple forward scan suffices for both the active-swing check and the
 * most-recently-landed lookup.
 */
function _supportAt( merged, t, supportEaseSec ) {

	for ( let i = 0; i < merged.length; i ++ ) {

		const e = merged[ i ];
		if ( t >= e.tLift && t < e.tLand ) return e.foot === 'left' ? - 1 : 1;

	}

	let lastLanded = null;
	for ( let i = 0; i < merged.length; i ++ ) {

		if ( merged[ i ].tLand <= t ) lastLanded = merged[ i ]; else break;

	}
	if ( ! lastLanded ) return 0;

	const held = lastLanded.foot === 'left' ? - 1 : 1;
	const uRaw = supportEaseSec > 1e-6 ? ( t - lastLanded.tLand ) / supportEaseSec : 1.0;
	const u = Math.max( 0, Math.min( 1, uRaw ) );
	const value = held * ( 1.0 - _smoothstep( u ) );

	return Math.max( - 1, Math.min( 1, value ) );

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

	// G3 out-toeing (IK_OVERHAUL_SPEC.md section 4): applied ONLY here, at
	// display-yaw time -- never to buildSchedule's own trigger/need math (which
	// reads state[foot].plantYaw / s.yaw directly and never calls this function)
	// or to _nominalAt's yaw (still raw), so scheduling behavior is completely
	// unaffected by this offset. `sign` (+1 left / -1 right -- the SAME convention
	// _nominalAt's own lateral offset uses; see its comment: "+Y is the LEFT
	// side") IS the correct out-toe sign directly: LEFT (sign=+1) gets
	// +outToeRad, RIGHT (sign=-1) gets -outToeRad, splaying both toes away from
	// the midline. Applied at each of this function's three RETURN sites below
	// (events/planted state itself keeps storing RAW yaws).
	const outToeRad = schedule.params.outToeRad;

	// G6 per-foot timing fields (IK_OVERHAUL_SPEC.md section 3): nextLiftAt is
	// independent of which branch below applies (swinging/planted/pre-first-event)
	// -- this foot's OWN event list is already time-sorted (events are appended in
	// increasing tLift order during buildSchedule's forward march, then the
	// walk-on tail appends further steps also in increasing order -- see
	// _buildWalkOn), so the first entry whose tLift >= t is the answer.
	let nextLiftAt = null;
	for ( let ni = 0; ni < evs.length; ni ++ ) {

		if ( evs[ ni ].tLift >= t ) { nextLiftAt = evs[ ni ].tLift; break; }

	}

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
			const strideLen = _hyp2( e.to.x - e.from.x, e.to.y - e.from.y );

			return {
				x, y, z, yaw: yaw + sign * outToeRad, planted: false, swingU: u,
				liftAt: e.tLift, landedAt: i > 0 ? evs[ i - 1 ].tLand : null, nextLiftAt, strideLen,
			};

		}

	}

	// Not swinging: planted at the most recent event's `to` (or the schedule's initial
	// nominal if this foot has no events yet at/before t).
	let lastLanded = null;
	for ( let i = 0; i < evs.length; i ++ ) {

		if ( evs[ i ].tLand <= t ) lastLanded = evs[ i ]; else break;

	}

	if ( lastLanded ) {

		const strideLen = _hyp2( lastLanded.to.x - lastLanded.from.x, lastLanded.to.y - lastLanded.from.y );
		return {
			x: lastLanded.to.x, y: lastLanded.to.y, z: lastLanded.to.z, yaw: lastLanded.toYaw + sign * outToeRad,
			planted: true, swingU: null, liftAt: null, landedAt: lastLanded.tLand, nextLiftAt, strideLen,
		};

	}

	// Before this foot's first event (or the clip has none): planted at the initial
	// nominal computed from the FIRST sample (matches buildSchedule's own initial
	// state, so t=0 querying before any event is scheduled is consistent with what
	// buildSchedule assumed as its starting condition).
	const s0 = schedule.samples[ 0 ];
	const nominal = _nominalAt( s0, sign, schedule.params.footLateral, terrain );
	return {
		x: nominal.x, y: nominal.y, z: nominal.z, yaw: s0.yaw + sign * outToeRad,
		planted: true, swingU: null, liftAt: null, landedAt: null, nextLiftAt, strideLen: 0,
	};

}

/**
 * G5 cane tip pose at time t (IK_OVERHAUL_SPEC.md section 3): same planted/swing
 * lookup shape as _footPoseAt (swing-window blend with arc + terrain clamp, else
 * planted at the most recent landed cane event, else the schedule's initial cane
 * nominal before the first cane event) but over schedule.caneEvents instead of a
 * foot's own event list, and with NO out-toe (a cane tip has no yaw-splay concept)
 * and no strideLen (not part of the cane's poseAt contract, see IK_OVERHAUL_SPEC.md
 * section 3's cane block). Only called when schedule.caneEvents is non-null
 * (poseAt itself gates on that).
 */
function _canePoseAt( schedule, terrain, t ) {

	const evs = schedule.caneEvents;

	let nextLiftAt = null;
	for ( let ni = 0; ni < evs.length; ni ++ ) {

		if ( evs[ ni ].tLift >= t ) { nextLiftAt = evs[ ni ].tLift; break; }

	}

	for ( let i = 0; i < evs.length; i ++ ) {

		const e = evs[ i ];
		if ( t >= e.tLift && t < e.tLand ) {

			const u = e.tLand > e.tLift ? ( t - e.tLift ) / ( e.tLand - e.tLift ) : 1.0;
			const ease = _smoothstep( u );

			const x = e.from.x + ( e.to.x - e.from.x ) * ease;
			const y = e.from.y + ( e.to.y - e.from.y ) * ease;
			const zEndpointBlend = e.from.z + ( e.to.z - e.from.z ) * ease;
			const arcBump = Math.max( 0.0, e.apexZ - Math.max( e.from.z, e.to.z ) ) * Math.sin( Math.PI * u );
			const zArc = zEndpointBlend + arcBump;

			// Same ceiling-indexed, ease-space clamp lookup as _footPoseAt -- see its
			// own comment for the full rationale (raw-terrain running-max profile,
			// indexed by ease not u, rounded UP to the conservative neighbour).
			const profile = e.clampProfile;
			const profileIdxF = ease * ( profile.length - 1 );
			const profileIdxCeil = Math.min( profile.length - 1, Math.ceil( profileIdxF - 1e-9 ) );
			const clampFloor = profile[ profileIdxCeil ];
			const z = Math.max( zArc, clampFloor );

			return {
				x, y, z, planted: false, swingU: u,
				liftAt: e.tLift, landedAt: i > 0 ? evs[ i - 1 ].tLand : null, nextLiftAt,
			};

		}

	}

	let lastLanded = null;
	for ( let i = 0; i < evs.length; i ++ ) {

		if ( evs[ i ].tLand <= t ) lastLanded = evs[ i ]; else break;

	}

	if ( lastLanded ) {

		return {
			x: lastLanded.to.x, y: lastLanded.to.y, z: lastLanded.to.z,
			planted: true, swingU: null, liftAt: null, landedAt: lastLanded.tLand, nextLiftAt,
		};

	}

	// Before the first cane event: planted at the initial nominal -- root at
	// samples[0], SAME forward/right offset formula as every scheduled cane
	// target (see _caneTargetAt), matching _footPoseAt's own "t=0 before any
	// event" convention.
	const s0 = schedule.samples[ 0 ];
	const nominal = _caneTargetAt( s0, terrain, schedule.params );
	return {
		x: nominal.x, y: nominal.y, z: nominal.z,
		planted: true, swingU: null, liftAt: null, landedAt: null, nextLiftAt,
	};

}
