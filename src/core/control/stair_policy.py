"""Stair-climb command policy and the stair-aware front-obstacle gate.

Extracted verbatim from main.py: sensor-derived stair-climb forward command,
depth-from-bbox helpers (with person exclusion), and the front-obstacle gate
that scales the forward command near obstacles. Pure functions driven by the
per-frame ``debug_info``/``args`` passed in by the main loop.
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.vision.depth_processor import DepthProcessor

# A stair riser reads "near" in the forward ROI just like a body does; it is told apart
# by a strong row-to-row depth gradient (near riser face at the top rows, open tread/
# ground receding toward the bottom rows). Same threshold the front-obstacle gate uses.
_RISER_GRADIENT_MM_PER_ROW = 20.0

# Shared "same physical surface" tolerance (m) between a live near-field/gap depth
# reading and the geometric depth-stair detector's OWN leading-edge reading -- used by
# both _stair_loss_forward_block's nearfield-riser context override and
# too_close_riser_gap_suppressed below. Sized well above the measured run-18 wedge
# agreement (|0.304-0.302|=0.002 m, run_sim_20260712_103237_267, sim_t=40.88 -- see both
# call sites' docstrings) while staying tight enough that an unrelated wall/body would
# not coincidentally match a real staircase's own leading edge.
NEARFIELD_LEADING_EDGE_AGREE_M = 0.15
# Nearest distance (m) at which the run-18 pass-as-riser context override still applies;
# closer than this the near-field block re-asserts so the dog holds at APPROACH distance
# instead of pressing the riser (see the override comment in _stair_loss_forward_block:
# stage-5 cannot mount from a dead-stand contact press, run 21). 0.45 sits inside the
# HandoffController approach_room window [0.40, 0.65] measured at the leading edge.
NEARFIELD_RISER_HOLD_STANDOFF_M = 0.45


@dataclass
class DepthStairGate:
    """Result of the depth-based near-field stair gate (see evaluate_depth_stair_gate)."""
    result: Dict[str, Any]     # raw DepthStairDetector.detect() output (leading_edge etc.)
    confirmed: bool            # stair_detected AND stair_count >= min_count
    person_masked: bool        # the followed person's bbox was zeroed before detect()


def evaluate_depth_stair_gate(
    depth_img_mm: np.ndarray,
    person_bbox: Optional[Sequence[float]],
    detector: Any,
    *,
    min_count: int,
) -> DepthStairGate:
    """Run the geometric depth stair detector, fixing two field bugs BY CONSTRUCTION.

    Extracted verbatim from the main control loop so the exact code of both incident-8.3
    defects is unit-testable off-robot (the loop had zero tests over it):

      * UNITS (P2-2): ``DepthStairDetector.detect`` treats its grid as METRES (it filters
        ``0.06 < d < ~2.2`` m and derives world heights), but the D435 depth is uint16
        MILLIMETRES everywhere else in the pipeline. Feeding mm made every pixel exceed
        the range gate, the valid-row filter emptied, and the detector fired on 0 frames
        (silently dead). We convert mm -> m here.
      * PERSON FALSE-STAIR (incident 8.3): a patient standing ~0.6 m ahead fills the
        detector's central column band; their body (feet->head at ~constant forward
        distance) back-projects into a stack of rising height LEVELS the clusterer reads
        as a multi-riser staircase (observed 8 fake risers at the follow standoff),
        latching stairs_detected from frame 1 and forcing stair mode on flat ground. We
        zero the followed person's bbox in the grid BEFORE detect() so only real terrain
        drives it. (RESIDUAL, per the incident ledger: masking can leave band-edge slivers
        that still occasionally confirm; the full fix additionally gates the depth-only
        latch on recent YOLO stair evidence -- not done here.)

    The input ``depth_img_mm`` is never mutated (the ``* 0.001`` produces a fresh array).
    """
    grid = np.asarray(depth_img_mm, dtype=np.float32) * 0.001   # mm -> m (units fix)
    person_masked = False
    if person_bbox is not None and len(person_bbox) >= 4:
        h, w = grid.shape[:2]
        x1 = max(0, int(round(float(person_bbox[0]))))
        y1 = max(0, int(round(float(person_bbox[1]))))
        x2 = min(w, int(round(float(person_bbox[2]))))
        y2 = min(h, int(round(float(person_bbox[3]))))
        if x2 > x1 and y2 > y1:
            grid[y1:y2, x1:x2] = 0.0
            person_masked = True
    result = detector.detect(grid)
    confirmed = (
        bool(result.get("stair_detected", False))
        and int(result.get("stair_count", 0)) >= int(min_count)
    )
    return DepthStairGate(result=result, confirmed=confirmed, person_masked=person_masked)


def depth_stair_latch_allowed(
    *,
    depth_confirmed: bool,
    now: float,
    last_yolo_stair_ts: float,
    persist_sec: float,
) -> bool:
    """Whether a DEPTH stair confirmation may (re)latch stair mode this frame.

    Only when YOLO-World has corroborated stairs within ``persist_sec`` (the design intent:
    YOLO detects the staircase from AFAR, depth carries it at close range where YOLO blanks).
    This blocks near-floor / person-edge depth false-positives -- which confirm >=2 fake risers
    on FLAT ground -- from latching stair mode and killing plain-follow (incident 8.3 residual).
    YOLO itself latches independently; this only governs the DEPTH-only path.
    """
    if not depth_confirmed:
        return False
    return (float(now) - float(last_yolo_stair_ts)) <= float(persist_sec)


def climb_gap_brake_scale(
    gap_m: Optional[float],
    *,
    brake_start_m: float,
    brake_stop_m: float,
    person_detected: bool,
) -> float:
    """Scale (0..1) for the mid-climb forward-speed CAP, from the live patient gap.

    Incident 8.15 / F2: 1.0 (no braking) at ``gap_m >= brake_start_m``, linearly tapering
    to 0.0 at ``gap_m <= brake_stop_m``.

    NONE-GAP CONTRACT (corrected 2026-07-11, second 8.15 scope correction) -- split by
    person VISIBILITY, which the caller MUST pass as an explicit argument (incident 8.5:
    never re-read from a downstream ``debug_info`` key; the producer is upstream --
    ``person_follower.update`` rebinds ``debug_info`` at ``core/main.py:727`` and writes
    ``person_detected`` at ``core/control/follow_controller.py:434``, ahead of every brake
    call site: ``core/main.py:1145`` -> ``_apply_stair_command_policy`` below, and
    ``core/main.py`` ~L1337 / ~L2110 / ~L2462):

    * ``person_detected`` False -> 1.0 (NO brake), regardless of ``gap_m``. The patient
      being out of view mid-climb is the NORMAL incident-8.3 blind-carry situation (they
      rise out of the close-range FOV near the crest), NOT a proximity risk -- the smoothed
      gap goes ``None`` there precisely BECAUSE nobody is visible. Failing toward the brake
      on that ``None`` parked the dog on the incline until it toppled rear-high
      (run_sim_20260711_153245_944: climbed to x=6.19, ~120 frames of STAIR_LOSS_FLOOR with
      ``gap_brake_scale=0.0`` and hold, flipped at roll 179 deg mid-crest). The 2026-07-11
      stop-probe (0/64 topples) only held commanded-zero for ~7 s; INDEFINITE zero-holds
      mid-staircase are not safe -- the blind-carry momentum is itself a stability strategy.
      The hard collision floors on ``last_person_gap_m`` at each call site still back-stop
      the person-was-close-then-dropped-out case.
    * ``person_detected`` True and ``gap_m`` ``None``/invalid/at-or-below the ``1e-3`` "no
      reading" sentinel -> 0.0 (fail toward the BRAKE; incident 8.8 unchanged): a person is
      VISIBLE but the gap cannot be measured -- a genuine sensor error while a real
      proximity risk exists must stop, not charge. (The pre-existing collision-floor checks
      in this module deliberately SKIP on that same sentinel; this stays the opposite, on
      purpose, for the detected case only.)
    * ``person_detected`` True with a valid gap -> the linear taper above.

    ``brake_stop_m`` should sit above ``--stair-climb-collision-floor`` (default 0.55 m)
    with margin for the measured residual creep at commanded vx=0 mid-stairs (2026-07-11
    policy probe: 0.146 m/s mean / 0.367 m/s p95, 0/64 topples over its ~7 s window) so the
    smooth taper -- not the pre-existing hard binary cutoff -- is what normally arrests the
    approach.
    """
    if not bool(person_detected):
        return 1.0
    if gap_m is None:
        return 0.0
    try:
        g = float(gap_m)
    except (TypeError, ValueError):
        return 0.0
    if g <= 1e-3:
        return 0.0
    start = float(brake_start_m)
    stop = float(brake_stop_m)
    if g >= start:
        return 1.0
    if g <= stop:
        return 0.0
    span = max(1e-6, start - stop)
    return float(np.clip((g - stop) / span, 0.0, 1.0))


def too_close_riser_gap_suppressed(
    *,
    person_detected: bool,
    gap_for_hold: Optional[float],
    depth_stair_confirmed: bool,
    depth_stair_leading_edge_m: Optional[float],
) -> bool:
    """True => the main loop's too-close stance-lock gap is actually the STAIRCASE, not
    the patient, and that specific ``too_close`` assertion should be suppressed.

    Mirror of incident 8.3 in the opposite direction: there, the person's own footprint
    was misread as stair risers; here, a stair reading is what a stale/contaminated
    follow-gap history could misread as the person. ``too_close`` (``core/main.py``
    ~L2005-2012) stance-locks on ``standoff_gap_ctrl_m`` -- a MEDIAN of the last few
    ``depth_distance_m`` samples (the person's own measured gap) -- falling below the
    standoff lower bound. That gap is only ever meaningful while a person IS detected; if
    ``person_detected`` is False THIS frame, whatever the median is currently reporting is
    stale by construction, and the specific failure mode this guards is a NEAR reading
    (from any producer feeding the same median) that happens to describe the confirmed
    staircase's own leading edge, not a person the tracker no longer sees at all.

    Requires ALL of: person NOT detected, a geometrically CONFIRMED multi-riser structure
    THIS frame (``depth_stair_confirmed`` -- not just any stray near reading, same
    reasoning as ``_stair_loss_forward_block``'s context override above), and that
    structure's own leading edge agreeing with the fused hold-gap within
    ``NEARFIELD_LEADING_EDGE_AGREE_M`` (the same "same physical surface" tolerance, sized
    from the run-18 wedge citation on ``_stair_loss_forward_block``).

    FAILS TOWARD HOLDING (CLAUDE.md 8.8) on every ambiguous case: person detected (the
    ordinary too-close case is untouched), no confirmed staircase this frame, or either
    reading missing -- returns False (do NOT suppress; the pre-existing stance-lock still
    applies) rather than guessing. Explicit keyword arguments only (CLAUDE.md 8.5): the
    caller (``core/main.py`` ~L2005) passes its own same-frame ``person_detected``/
    ``depth_stair_confirmed``/``depth_stair_leading_edge_m`` reads, never re-derived here.
    """
    if bool(person_detected):
        return False
    if not bool(depth_stair_confirmed):
        return False
    if gap_for_hold is None or depth_stair_leading_edge_m is None:
        return False
    return (
        abs(float(gap_for_hold) - float(depth_stair_leading_edge_m))
        <= float(NEARFIELD_LEADING_EDGE_AGREE_M)
    )


def effective_climb_gap_brake_scale(
    gap_brake_scale: float,
    *,
    person_detected: bool,
    hard_block: bool,
) -> float:
    """The EFFECTIVE (0..1) floor-authority fraction sent to ``controller.move``'s
    ``gap_brake_scale`` argument and forwarded over UDP to isaac_env's INDEPENDENT mid-climb
    floor (``handoff_climb_vx`` * this scale) -- a DIFFERENT authority from the caller's own
    ``trans_x``/``vx`` for this frame, which callers zero directly on ``hard_block`` and must
    keep doing so unchanged (this function does not touch that).

    Regression fix (run 15, run_sim_20260712_030822_222): the three mid-climb call sites in
    core/main.py (persistence-latch forced climb, committed climb, STAIR_LOSS_FLOOR) each
    folded an unrelated HARD-BLOCK flag (a stale-gap collision floor, a wall-clock blind-
    detection-timeout ceiling, a live near-field non-riser return, ...) into the stored/sent
    scale UNCONDITIONALLY -- including on every frame the person was simply not detected,
    which is the NORMAL incident-8.3 designed blind-carry (the patient climbing/standing out
    of the close-range FOV), not a proximity risk. Trace evidence
    (log/run_sim_20260712_030822_222/debug/debug_trace/vision_main_trace.jsonl, sim_t=43.8-
    45.0): ``person_detected=False`` and ``stair_climb_latch_blind_timeout=True`` the entire
    window -> the OLD fold pinned the stored scale at 0.0 on every one of those frames ->
    isaac_env's own independent floor obeyed (0.22 * 0.0 = 0.0) -> the blind-carry commanded
    zero indefinitely -> a sustained hold let ``hold_park`` park the dog at the stair base
    (x=1.78) -> ``_run_pgtt_handoff`` is skipped while parked -> ENGAGE never fires. Runs 13/14
    (pre-fold) climbed fine because the sim-side floor then ignored the caller's scale
    entirely and carried the dog through exactly this window.

    * ``person_detected`` False -> 1.0 ALWAYS, regardless of ``hard_block`` or the raw
      ``gap_brake_scale`` -- mirrors ``climb_gap_brake_scale``'s own 8.15-correction-2
      None-split for the identical reason: absence of a detected person is not a proximity
      risk, and the hard collision floors that fed ``hard_block`` already act on the CALLER's
      own vx term (unchanged), which is the correct place for a frozen/stale-gap backstop to
      bite.
    * ``person_detected`` True -> the pre-existing fold, unchanged: 0.0 if ``hard_block`` else
      the raw ``gap_brake_scale``.
    """
    if not bool(person_detected):
        return 1.0
    return 0.0 if bool(hard_block) else float(gap_brake_scale)


def mid_climb_floor_capped_command(
    trans_x_cmd: float,
    *,
    forward_floor: float,
    base_cap: float,
    gap_brake_scale: float,
) -> float:
    """Apply the stair forward floor, then clamp to the (gap-braked) speed cap.

    Incident S2 (2026-07-12 review of run_sim_20260712_023126_786): extracted from
    core/main.py's stair-climb persistence-latch call site (~L1343-1400), which computed
    the braked cap as ``base_cap * gap_brake_scale`` but only APPLIED it when the result
    was ``> 0.0`` -- a guard meant to mean "capping disabled" (``base_cap`` itself is 0,
    e.g. ``--stair-speed-scale 0``) but which, checked AFTER the brake multiplication, also
    silently skipped a LEGITIMATE full-strength brake (``gap_brake_scale == 0.0``, i.e.
    "stop -- the patient is right there"). With the clamp skipped, the raw UNBRAKED
    ``forward_floor`` (0.16 m/s, ``--stair-forward-floor``) leaked straight through for 8+
    consecutive frames while ``person_detected=True`` and the true GT dog<->patient gap sat
    at 0.45-0.52 m (fall_diag sim_t=30.975, x=2.126, gap_m=0.452, policy_cmd=[0.16, 0, 0];
    cross-checked against vision_main_trace.jsonl at its OWN sim_t=36.155 -- the two logs do
    NOT share a clock origin, offset ~5.62 s measured by matching robot x -- which showed
    ``stair_climb_gap_filtered_m=0.501`` (<< the 0.85 m ``climb_gap_brake_stop`` ->
    ``gap_brake_scale=0.0``) and ``stair_climb_latch_collision_block=False`` (the sim-GT-lead
    unblock, main.py ~L1380, had already cleared the separate hard floor check) yet
    ``command_trans_x_limited`` stayed pinned at 0.16). Neither
    ``_apply_stair_command_policy`` (this module, mid-climb dispatch) nor the STAIR_LOSS_FLOOR
    call site in core/main.py has this bug -- both clamp with a plain comparison against the
    (possibly-zero) braked value, never an ``if cap > 0`` gate on the ALREADY-braked number.
    Fixed here by taking the enable/disable test (``base_cap``, the UNBRAKED cap -- the true
    "capping disabled" sentinel) as an explicit separate argument from the value actually
    applied (``base_cap * gap_brake_scale``, which may legitimately be exactly 0.0).
    """
    out = max(float(trans_x_cmd), float(forward_floor))
    base = float(base_cap)
    if base > 0.0:
        out = min(out, base * float(gap_brake_scale))
    return float(out)


@dataclass
class ClimbGapFilterState:
    """Caller-owned rolling-window state for ``filtered_climb_gap_m`` (incident 8.15 / F2
    hardening, 2026-07-11 review).  One instance is created by the main loop and threaded
    through every frame -- mirrors ``standoff_state``/``carrot_state`` in ``core/main.py``
    (plain caller-owned containers, never a module global) rather than the ClimbFSM class
    pattern, since this is a single trailing-window buffer with no other behavior.
    """
    samples: List[Tuple[float, float]] = field(default_factory=list)  # (perf_counter ts, gap_m)


def filtered_climb_gap_m(
    gap_m: Optional[float],
    *,
    person_detected: bool,
    state: ClimbGapFilterState,
    now: float,
    window_sec: float,
) -> Optional[float]:
    """Conservative (rolling-MINIMUM) mid-climb patient gap over the trailing ``window_sec``.

    Incident 8.15 / F2 hardening (2026-07-11 review of run_sim_20260711_195618_941): the raw
    ``standoff_gap_ctrl_m`` (already median-filtered over the last 5 raw readings in
    ``_apply_follow_standoff_policy``) still swings wildly mid-climb -- the patient is half out
    of the D435's close-range FOV while climbing ahead of the dog -- e.g. that run logged
    ``0.282 -> 0.885 -> 0.629 -> None`` in three consecutive ~0.04-0.08 s control-loop frames,
    and separately ``1.837 -> 2.797 -> 1.862 -> 3.047 -> 0.788 -> 1.883`` oscillating frame to
    frame while the true (physically continuous) gap was closing under ~0.5 m.
    ``climb_gap_brake_scale`` is MEMORYLESS (one frame's gap in, one frame's scale out), so a
    single noisy "far" reading released the brake to 1.0 for that frame and the caller's forward
    floor/cap jumped straight to the unbraked ``max_forward`` (observed vx pulses to 0.383 m/s
    while the trailing minimum gap was under 0.85 m) -- tailgating the patient down to
    0.199-0.282 m against the 0.55-0.65 m hard collision floor.

    This wraps ``climb_gap_brake_scale``'s gap input with a trailing window that takes the
    MINIMUM (never the mean/median) of the valid samples seen in the last ``window_sec`` seconds
    -- the brake must fail toward BRAKING on noise, so a single genuinely-close reading anywhere
    in the window keeps the brake engaged even when surrounded by noisy "far" readings; it can
    only ever look MORE cautious than the raw live gap, never less.

    ``window_sec`` is a WALL-CLOCK duration and ``now`` MUST be ``time.perf_counter()`` (incident
    8.6) -- matching every other stair-timing signal in this module
    (``lost_person_speed_taper_scale``, ``HandoffController``).

    State ownership (incident 8.15 / F2 review): ``state`` is an explicit
    ``ClimbGapFilterState`` the CALLER owns for the whole run, never a module global.

    NONE-GAP / not-detected semantics (unchanged, 8.15 second correction preserved EXACTLY):
    samples are appended ONLY while ``person_detected`` is True and ``gap_m`` is a valid
    reading; a call with ``person_detected=False`` never adds a sample. Every call (detected or
    not) age-prunes the window against ``now`` first, so a detection gap LONGER than
    ``window_sec`` empties the window purely by aging out before the next valid sample can
    arrive -- a fresh detection after a long loss starts from an empty window and never
    inherits a stale pre-loss minimum. This function does NOT itself apply the
    ``person_detected`` "no brake" branch -- callers still pass ``person_detected`` to
    ``climb_gap_brake_scale`` separately (8.5: explicit argument, not re-derived here) exactly
    as before; this function only produces a more conservative ``gap_m`` for that call.

    Returns ``None`` when the window holds no valid samples (nothing detected yet this run, or
    the whole window aged out) -- callers pass this straight to ``climb_gap_brake_scale``, whose
    existing ``person_detected``-gated None contract already does the right thing (not-detected
    -> no brake regardless; detected + None -> brake, incident 8.8).
    """
    now_f = float(now)
    win = max(1e-3, float(window_sec))
    state.samples[:] = [(t, g) for (t, g) in state.samples if (now_f - t) <= win]
    if person_detected and gap_m is not None:
        try:
            g = float(gap_m)
        except (TypeError, ValueError):
            g = None
        if g is not None and g > 1e-3:
            state.samples.append((now_f, g))
    if not state.samples:
        return None
    return min(g for _, g in state.samples)


@dataclass
class DetectionAgeState:
    """Caller-owned anchor for ``detection_age_sec`` (incident 8.6 fix, 2026-07-12 review of
    run_sim_20260712_033341_649, "run 16"). Mirrors ``ClimbGapFilterState`` /
    ``LandingMarginState`` above -- one instance owned by the main loop for the whole run,
    never a module global.

    Records the last matched-detection instant in BOTH clock families at once (whichever were
    available at that instant), so ``detection_age_sec`` can pick the trustworthy clock for
    THIS frame's query without blending them.
    """
    last_wall_ts: Optional[float] = None   # time.perf_counter() at the last matched detection
    last_sim_t: Optional[float] = None     # frame_meta["sim_t"] at that same instant, or None
                                            # if sim_t was unavailable then (e.g. real hardware)


def note_detection_match(state: DetectionAgeState, *, now_wall: float, sim_t: Optional[float]) -> None:
    """Record a fresh matched-detection instant into ``state``.

    SINGLE PRODUCER (mirrors ``lost_person_speed_taper_scale``'s one-call-site contract,
    incident 8.15): call this from the exact site that already sets
    ``last_matched_visual_ts = time.perf_counter()`` in ``core/main.py`` (~L709-710, guarded
    on ``matched_visual_lock``) -- do not add a second call site. ``last_matched_visual_ts``
    itself is untouched by this fix (it also drives ``recent_visual_lock``/``lock_held``,
    outside this task's scope); this is a parallel, independent anchor used ONLY by
    ``detection_age_sec``.
    """
    state.last_wall_ts = float(now_wall)
    state.last_sim_t = None if sim_t is None else float(sim_t)


def detection_age_sec(state: DetectionAgeState, *, now_wall: float, sim_t: Optional[float]) -> float:
    """How long ago the last matched detection was -- in the clock family that actually paces
    the quantity ``--stair-blind-climb-timeout-sec`` bounds (incident 8.6: a duration must be
    measured in the same clock as the thing it bounds, never a different one, and never a
    blend of both).

    ROOT CAUSE (2026-07-12 review, runs 15+16, identical deadlock at the stair base): both
    call sites that read this age (``core/main.py``'s persistence-latch shaping ~L1398-1400
    and the STAIR_LOSS_FLOOR dispatch ~L2747-2749) compared a WALL-CLOCK
    (``time.perf_counter()``) age against the 8.0 s default ceiling, but the quantity being
    bounded -- "how long may the dog blind-carry while the patient walks ahead to build the
    2.4 m ``HandoffConfig.stair_entry_min_lead_m`` head-start lead" -- is paced by the SIM
    WORLD, which in the headless demo runs several times slower than the wall clock. Measured
    directly off run 16's own trace (``run_sim_20260712_033341_649``,
    ``log/.../debug_trace/vision_main_trace.jsonl``): the last matched detection before the
    terminal blind stretch lands at frame index 801, ``sim_t=31.675`` (``stair_climb_latch_det_
    age_sec`` resets to 0.12 there); by the end of the captured window at ``sim_t=47.075`` the
    WALL age had climbed to 88.17 s while ``sim_t`` had advanced only 15.40 s -- a measured
    ~5.7x wall:sim ratio in that stretch (``ts_mono`` delta 88.32 s / ``sim_t`` delta 15.40 s).
    Under the OLD wall-only rule the 8.0 s ceiling tripped after only ~1-2 SIM-seconds of loss
    (``stair_climb_latch_blind_timeout`` reads True by ``sim_t=32.795``, ~0.84-1.1 sim-s after
    the ``sim_t=31.955`` loss) -- nowhere near the several sim-seconds the patient needs to
    walk out to a 2.4 m lead -- so the dog held ``vx=0``/``hold=True`` for the rest of the
    trace (fsm_state stayed "STAIR_LOSS_FLOOR", trans_x_cmd pinned at 0.0 through
    ``sim_t=47.075``) and the head-start gate never got the runway it needed. With sim-time
    ageing the SAME 8.0 s default instead buys a full 8 SIM-seconds of blind-carry per loss
    (first tripping around ``sim_t=39.955`` in this trace, not ``32.795``) -- ample against the
    ~3-5 sim-second lead-building window the gate needs; it was NOT, however, "under 8 sim-s by
    sim_t~46" as a first-pass estimate assumed -- this run's loss segment happened to run past
    that too (measure, don't assume; CLAUDE.md 8.7).

    SEMANTICS (present / absent / vanishing -- an explicit either/or, never blended):

      * ``sim_t`` present THIS frame, ``state.last_sim_t`` present (recorded at the last
        match), and ``sim_t >= state.last_sim_t`` ("monotonically advancing" since that match)
        -> SIM-TIME age: ``sim_t - state.last_sim_t``. This is the normal in-sim path.
      * ``sim_t`` is ``None`` this frame (real hardware, where ``frame_meta`` never carries
        ``sim_t`` at all per ``shared/frame_source.py``'s ``FRAME_META_OPTIONAL_KEYS`` contract
        -- wall IS world there, so this is the CORRECT clock, not a degraded fallback) -> WALL
        age: ``now_wall - state.last_wall_ts``.
      * ``sim_t`` present now but ``state.last_sim_t`` is ``None`` (sim_t was unavailable AT
        the last match -- sim_t "appearing" mid-run) -> WALL age (no sim-time anchor exists to
        diff against).
      * ``sim_t`` present now and at the last match, but has gone BACKWARD
        (``sim_t < state.last_sim_t`` -- e.g. an episode/scene reset snapping ``sim_t`` back
        toward 0) -> WALL age. A negative or falsely-fresh "age=0" reading here would silently
        re-open the blind-carry window right after a reset; falling back to wall reproduces
        EXACTLY today's pre-fix behaviour, which is a safe (if conservative) known quantity,
        never worse than what already ships.
      * Never matched this run at all (``state.last_wall_ts is None``) -> ``1e9`` (unchanged
        sentinel, matches the pre-fix ``last_matched_visual_ts is None`` branch exactly).

    CAVEAT (CLAUDE.md 8.7 -- do not over-claim): this does not detect a FROZEN-but-present
    ``sim_t`` (paused sim reporting the same value every frame without a reset). That reads as
    "monotonically advancing" (``>=`` with equal values) and the computed age would stay flat
    at whatever it was when the freeze began, silently disabling the ceiling for the freeze's
    duration. Out of scope here -- no evidence of a paused-sim-clock failure mode exists in any
    reviewed run -- but a future agent extending this must not assume frozen clocks are
    handled.
    """
    if state.last_wall_ts is None:
        return 1e9
    if (sim_t is not None and state.last_sim_t is not None
            and float(sim_t) >= state.last_sim_t):
        return float(sim_t) - state.last_sim_t
    return float(now_wall) - state.last_wall_ts


def stair_loss_gap_block(
    last_person_gap_m: Optional[float],
    *,
    collision_floor_m: float,
    detection_age_sec: float,
    immediate_guard_sec: float,
) -> bool:
    """Whether the FROZEN last-known patient gap should still zero a stair loss-drive
    branch's forward command THIS frame (True => block / drive at 0.0).

    Incident 8.15-corr / run-17 fix (2026-07-12 review): runs 15, 16, AND 17 all settled
    IDENTICALLY at x~=1.77, dog completely STATIONARY, climb never engaged --
    isaac_env.jsonl logged ZERO ``handoff_engage`` / ``handoff_engage_vetoed_lead`` events
    across the whole run, because the wedge/approach engage path needs the CALLER to command
    a nonzero vx to even ATTEMPT one, and this guard's ORIGINAL, unaged form zeroed the
    caller's own vx for the rest of the run once it fired once.

    ``last_person_gap_m`` is a ONE-SHOT snapshot taken the instant the patient was last
    actually detected (``core/main.py`` ~L1271-1276) -- normally close (~0.4-0.8 m, the
    stair-approach follow standoff) right before they climb/rise out of the close-range FOV
    (incident 8.3, the NORMAL trigger at the stair base, not a fault). The two call sites this
    guards (``core/main.py``'s STAIR_LOSS_FLOOR ``_loss_block`` and STAIR_APPROACH_COMMIT
    ``_ap_block``) originally held that snapshot as "current" with NO aging at all -- "if it
    was unsafe, do not assume that elapsed time means the patient moved away; keep hold=False
    and wait for a real lock" -- but the patient keeps walking (climbing) away autonomously
    the ENTIRE time either branch holds vx=0 (there is no stance-lock here, per CLAUDE.md 8.9 /
    8.15 -- the gait stays alive), so elapsed time since that snapshot is exactly the evidence
    the true gap has reopened, not evidence to distrust. Held with no aging, the guard zeroed
    the dog's own forward command for the rest of the run and no downstream engage machinery
    (``go2_locomotion/handoff_controller.py``'s wedge-stall / approach-room engage) ever saw a
    commanded vx>0 to attempt on.

    Age-gated with the caller-supplied, sim-time-aware ``detection_age_sec`` (produced by this
    module's own ``detection_age_sec()`` function against the shared ``DetectionAgeState``,
    incident 8.6 -- passed in here as an explicit float, never recomputed, per incident 8.5)
    against ``immediate_guard_sec`` (callers pass ``--stair-loss-block-immediate-guard-sec``,
    default 2.0 sim-aware seconds): the frozen gap is trusted as "current" only for that first
    window since the patient was last actually seen -- long enough to still catch the "patient
    paused right at the step and detection happened to drop at that exact instant" case the
    LIVE near-field guard (``_stair_loss_forward_block``) and the longer detection-age
    staleness ceiling (``--stair-blind-climb-timeout-sec``) exist alongside this one to catch,
    short enough that by the time it expires the patient has unquestionably moved on (climbing
    at roughly 0.5 m/s, per CLAUDE.md 8.15's measured pacing).

    Returns ``False`` (no block) whenever ``last_person_gap_m`` is ``None`` (nobody has been
    detected yet this run -- there is no frozen gap to distrust) or is already at/above
    ``collision_floor_m`` (the last known reading was a safe distance, regardless of age).
    This does NOT replace either of the other two independent guards at each call site (the
    live near-field probe, or the detection-age staleness ceiling) -- both stay exactly as
    they were; they are correctly unaged-by-design backstops, not instances of this same
    frozen-gap-as-current defect.
    """
    if last_person_gap_m is None:
        return False
    if float(last_person_gap_m) >= float(collision_floor_m):
        return False
    return float(detection_age_sec) < float(immediate_guard_sec)


def lost_person_speed_taper_scale(
    lost_age_sec: Optional[float],
    *,
    taper_start_sec: float,
    taper_full_sec: float,
) -> float:
    """Scale (0..1) for POST-CREST / TOP-LANDING forward speed only, from continuous
    person-loss age.

    Incident 8.15 / F3: 1.0 (no taper) while the person has been detected within
    ``taper_start_sec`` seconds (``lost_age_sec`` is ``None`` -- currently detected -- or
    ``<= taper_start_sec``); linearly bleeds to 0.0 by ``taper_full_sec`` seconds of
    CONTINUOUS loss. ``lost_age_sec`` MUST be a wall-clock duration (incident 8.6):
    ``PersonFollower`` produces it off ``perf_counter`` (``current_time - self.last_lost_time``,
    ``core/control/follow_controller.py``), never a frame count.

    SCOPE (corrected 2026-07-11, post-shipping regression): callers MUST gate this on the
    SAME one-way latch that arms the top-landing edge guard (``_post_crest_landing_latched``
    in ``core/main.py``, keyed off ``frame_meta["stair_demo"]["phase"] == "top_landing"`` /
    ``debug_info["stair_finish_completed"]``) -- i.e. only once the crest is genuinely
    reached. Do NOT call this from any MID-CLIMB blind-carry path: the persistence-latch
    forced climb, the committed-climb branch, STAIR_LOSS_FLOOR, or the stair-approach-commit
    creep (all in ``core/main.py``), nor from the near-stairs brief_loss floor in
    ``_apply_stair_command_policy`` below. All of those are incident-8.3 DESIGNED
    blind-carry -- the person going undetected there (patient rises out of the close-range
    FOV at the stair base, or climbs ahead out of view) is the NORMAL trigger for the floor,
    not a fault to bleed toward zero. Applying this taper in those paths shipped as a live
    regression (run 2026-07-11_150906: patient rose out of FOV at the base as expected,
    STAIR_LOSS_FLOOR engaged as designed, the taper decayed to 0.0 and permanently parked
    the robot at x=1.86, ``robot_settled``, climb never engaged) -- see CLAUDE.md incident
    8.15. Mid-climb patient proximity is covered by ``climb_gap_brake_scale`` instead; the
    landing's forward drop-off is covered separately by ``detect_landing_edge_dropoff``. The
    ONE remaining call site is in ``core/main.py``'s general follow dispatch (the
    ``elif motion_allowed and controller is not None:`` branch), immediately gated on
    ``if _post_crest_landing_latched:`` right before the trans_x limiter update -- do not add
    a second call site, and do not resurrect a mid-climb one.

    Scoped to stair-mode / post-crest callers ONLY -- do NOT call this from flat plain-follow
    or from anything gating ByteTrack's ``max_time_lost`` coast window (incident 8.6
    counter-example: that window is correctly a frame count of missed DETECTION
    OPPORTUNITIES, not a duration, and converting it to seconds previously broke person-
    follow through zig-zag turns).

    Unlike ``climb_gap_brake_scale``, ``None`` here means "currently detected" (a genuinely
    different signal, not a missing measurement) and returns 1.0, not 0.0.
    """
    if lost_age_sec is None:
        return 1.0
    try:
        age = float(lost_age_sec)
    except (TypeError, ValueError):
        return 1.0
    start = float(taper_start_sec)
    full = float(taper_full_sec)
    if age <= start:
        return 1.0
    if age >= full:
        return 0.0
    span = max(1e-6, full - start)
    return float(np.clip(1.0 - (age - start) / span, 0.0, 1.0))


def detect_landing_edge_dropoff(
    depth_img_mm: Optional[np.ndarray],
    cfg: Any,
    *,
    reach_m: float,
    drop_m: float,
    band_frac: Optional[float] = None,
) -> Optional[bool]:
    """Forward descending-edge (drop-off) probe for the flat post-crest top landing.

    Incident 8.15 / F4: nothing else in this module detects a DESCENDING edge -- the riser
    gradient tests elsewhere (``_apply_front_obstacle_gate`` / ``_roi_depth_row_gradient``)
    are built to recognise an ASCENDING riser (near face above, open tread below); a
    descending edge is the opposite failure and was unguarded (run_sim_20260711_140745_054:
    the blind post-crest walk drove 3 m across the top landing and off a 2.1 m drop).

    Reuses the SAME central-column depth projection and camera calibration as
    ``go2_locomotion.handoff_detectors.DepthStairDetector.detect`` (``cfg.stair_cam_vfov_deg``
    / ``stair_cam_pitch_deg`` / ``stair_cam_height_m`` / ``stair_band_frac``) to back-project
    each valid depth row to ``(forward_distance_m, world_height_m)`` -- but reads it for the
    opposite question: that detector looks for a level ABOVE the nearest reading (a riser);
    this looks at whether the floor stays LEVEL out to ``reach_m`` ahead, using the CLOSEST
    confirmed floor point (smallest forward distance) as the height reference rather than
    trusting the camera's absolute zero -- consistent with how DepthStairDetector itself
    never trusts an absolute ground height either (it clusters by RELATIVE jumps).

    Returns:
        True  -- a drop-off is confirmed within ``reach_m`` (STOP), OR no valid floor return
                 was found anywhere inside ``reach_m`` at all (on a flat landing a working
                 depth camera always returns the near floor; a working camera reporting
                 nothing there means the floor is not there -- fails toward the edge finding,
                 incident 8.8).
        False -- floor confirmed present and level (no row drops >= drop_m below the nearest
                 confirmed floor point) within ``reach_m``.
        None  -- the probe could not run AT ALL this frame (no/garbage depth image, or an
                 exception during the projection). The caller (incident 8.8) must fail toward
                 STOPPING and boot-log that the guard is inactive; this function reports
                 "unknown", it does not itself pick the safe default.
    """
    if depth_img_mm is None:
        return None
    try:
        D = np.asarray(depth_img_mm, dtype=np.float32)
        if D.ndim != 2 or D.size == 0:
            return None
        grid_m = D * 0.001  # mm -> m, same convention as evaluate_depth_stair_gate
        H, W = grid_m.shape
        band = max(0.05, min(1.0, float(
            band_frac if band_frac is not None else getattr(cfg, "stair_band_frac", 0.40)
        )))
        c0 = max(0, int(round(W * (0.5 - band / 2.0))))
        c1 = min(W, max(c0 + 1, int(round(W * (0.5 + band / 2.0)))))
        sub = grid_m[:, c0:c1]

        vfov = math.radians(float(getattr(cfg, "stair_cam_vfov_deg", 56.5)))
        pitch = math.radians(float(getattr(cfg, "stair_cam_pitch_deg", 0.5)))
        cam_h = float(getattr(cfg, "stair_cam_height_m", 0.40))
        cy = (H - 1) / 2.0

        pts = []  # (x_fwd_m, world_height_m) within reach_m
        for r in range(H):
            row = sub[r]
            row = row[np.isfinite(row) & (row > 0.06) & (row < 6.0)]
            if row.shape[0] < 3:
                continue
            d = float(np.median(row))
            theta_v = ((r - cy) / float(H)) * vfov
            ang = pitch + theta_v
            rng = d / max(0.2, math.cos(theta_v))
            x_fwd = rng * math.cos(ang)
            z_h = cam_h - rng * math.sin(ang)
            if 0.10 <= x_fwd <= float(reach_m):
                pts.append((x_fwd, z_h))

        if not pts:
            return True  # no floor return inside reach_m -- fail toward the edge (8.8)

        pts.sort(key=lambda p: p[0])
        near_z = pts[0][1]  # height of the closest confirmed floor point
        for _, z in pts[1:]:
            if (near_z - z) >= float(drop_m):
                return True
        return False
    except Exception:
        return None


def landing_edge_guard_suppress_crest_artifact(
    *,
    crest_relative_m: Optional[float],
    since_crest_latch_sec: Optional[float],
    commanded_away_from_crest: bool,
    suppress_reach_m: float,
    suppress_time_sec: float,
) -> bool:
    """Whether a confirmed ``detect_landing_edge_dropoff() == True`` finding is a stale CREST
    ARTIFACT (from the robot's own still-unsettled body pitch right after cresting) rather than
    a genuine forward dropoff, and should therefore be suppressed (not block motion).

    ROOT CAUSE (2026-07-11 review of run_sim_20260711_223152_489, incident 8.15 / F4
    follow-up): ``detect_landing_edge_dropoff`` and ``DepthStairDetector.detect`` (the depth
    stair approach detector) both back-project the depth image using a STATIC camera pitch
    (``cfg.stair_cam_pitch_deg``, default 0.5 deg -- ``go2_locomotion/handoff_config.py:55``,
    "downward mount pitch ... at a level stand"), read from a ``HandoffConfig()`` instance
    built ONCE at startup (``core/main.py:193-194``) and never updated with the robot's actual
    per-frame body attitude. The parkour depth camera that produces the real image is RIGIDLY
    body-parented (``sim/isaac/env/cameras.py:99-134``, esp. 102-104: "inherits the body's true
    gait pitch/roll/bob"), so its TRUE downward angle each frame is ``stair_cam_pitch_deg +
    (the robot's actual body pitch)`` -- not the static constant alone. Immediately after
    cresting, that run's GT body pitch (``stair_demo.robot.pitch_deg``) sat at -8.6 to -9.9 deg
    for 40+ consecutive frames while the dog held still (a stance the PGTT policy does not
    settle out of on its own while commanded to hold) -- well past the +-5 deg level band
    ``_fully_on_top_landing`` (this module, ~L607-630) already treats as "not level yet"
    (``post_crest_fully_on_landing`` read False every one of those frames in the trace).
    Replaying that exact pitch deviation through ``detect_landing_edge_dropoff`` with an
    otherwise perfectly flat, level synthetic floor (mirroring
    ``tests/test_stair_speed_guards.py::synth_flat_depth``, feeding
    ``pitch_deg=0.5+robot_pitch_deg`` in place of the assumed-level 0.5) reproduces
    ``detect_landing_edge_dropoff(...) is True`` for robot pitch in roughly [-12, -2] deg
    (includes the observed -8.6..-9.9 range) on a floor with NO drop anywhere -- confirming the
    false block is a geometry misread from the stale static-pitch assumption, not a genuine
    edge, a masked-person artifact, or the stairs still being in frame (the camera's yaw already
    points away from the crest, toward the patient, at this position).

    FIX (this function): rather than correcting the projection's pitch input (a larger, riskier
    change touching the shared DepthStairDetector calibration), suppress a block finding for a
    short, bounded window immediately after the crest, while the commanded direction is AWAY
    from the crest (toward the patient) -- mirroring ``_fully_on_top_landing``'s OWN two-tier
    design (sim GT distance when available, wall-clock time as the hardware-portable fallback)
    without modifying that function (CLAUDE.md task constraint):

      * SIM (GT available): ``crest_relative_m`` -- the caller's distance travelled past the
        GT crest entry point (mirrors ``LandingMarginState.landing_entry_x``, read-only, NOT
        this module's protected ``_fully_on_top_landing`` state mutation). Suppresses while
        ``0 <= crest_relative_m < suppress_reach_m``.
      * HARDWARE (no GT phase ever confirmed this run): ``since_crest_latch_sec`` -- wall-clock
        seconds (incident 8.6: a duration, never a frame count) since
        ``_post_crest_landing_latched`` first armed. Suppresses while
        ``0 <= since_crest_latch_sec < suppress_time_sec``.

    Both paths additionally require ``commanded_away_from_crest`` (the policy's forward command
    sign, read BEFORE this guard can veto it) -- direction-of-travel discrimination per the
    task brief, so a reversed/backward command near the crest is never suppressed.

    Deliberately narrow: the suppression window is far shorter than the 3 m / 2.1 m genuine
    dropoff distance in run_sim_20260711_140745_054 (incident 8.15 / F4's original motivating
    failure), so a real edge further down the landing still blocks -- see
    ``test_edge_guard_far_from_crest_still_blocks`` in test_stair_speed_guards.py.
    """
    if not commanded_away_from_crest:
        return False
    if crest_relative_m is not None:
        return 0.0 <= float(crest_relative_m) < float(suppress_reach_m)
    if since_crest_latch_sec is not None:
        return 0.0 <= float(since_crest_latch_sec) < float(suppress_time_sec)
    return False


@dataclass
class LandingEdgeLatchState:
    """Caller-owned hysteresis state for the post-crest landing-edge block (incident
    8.15/8.16 F2 hardening, run 11 review). One instance owned by the main loop, mirroring
    ``ClimbGapFilterState`` / ``LandingMarginState`` -- never a module global.
    """
    latched_until: Optional[float] = None  # perf_counter deadline the block stays asserted through


def landing_edge_block_latched(
    raw_block: bool,
    *,
    state: LandingEdgeLatchState,
    now: float,
    dwell_sec: float,
) -> bool:
    """Seconds-based (incident 8.6) hysteresis over the raw per-frame landing-edge finding.

    Run 11 (run_sim_20260711_234004_424) evidence: ``landing_edge_block`` fired True at
    sim_t=71.78 then FLICKERED False at t=72.14 / 79.14 / 79.48 -- hold_request=False with
    rotation_cmd=+/-0.6283 at the platform edge in those exact frames -- because the raw probe
    (``detect_landing_edge_dropoff``) only sees the drop-off while it is inside the forward
    depth FOV, and the lost-person spin-search (F1) was rotating the cliff in and out of that
    FOV every leg of its scan. A raw True (re)arms the latch through ``now + dwell_sec``; a
    raw False does NOT clear it early -- the PRE-EXISTING deadline keeps counting down, so the
    block stays asserted until ``dwell_sec`` has elapsed since the LAST True reading, covering
    the FOV-rotation gaps between re-detections. ``now`` MUST be ``time.perf_counter()``
    (incident 8.6 -- a wall-clock duration, never a frame count, matching every other latch in
    this module).

    Returns the latched boolean the caller should use in place of the raw per-frame finding
    for BOTH ``debug_info["landing_edge_block"]`` and the trans_x/rotation_cmd clamp -- see
    ``core/main.py``'s landing-edge-guard block.
    """
    now_f = float(now)
    if bool(raw_block):
        state.latched_until = now_f + max(0.0, float(dwell_sec))
        return True
    if state.latched_until is not None and now_f < state.latched_until:
        return True
    state.latched_until = None
    return False


def landing_lost_person_hold_active(
    *,
    post_crest_landing_latched: bool,
    fully_on_top_landing: bool,
    person_detected: bool,
) -> bool:
    """Whether the post-crest top-landing lost-person hold (incident 8.15/8.16 F1) should
    override the ordinary follow dispatch with an immediate STAND STILL (vx=0, wz=0, hold),
    instead of letting the flat-ground ``PersonFollower`` bounded lost-search scan run.

    Run 11 (run_sim_20260711_234004_424): once genuinely clear of the stairs, the fused
    person-range died at close range (``fused_gap_m`` unavailable while the patient stood at
    her destination, sim_t~65.8 -- NORMAL close-range ranging loss, incident 8.3, not a
    fault). The ordinary flat-ground lost-search
    (``core/control/follow_controller.py:641-725``) then engaged, and its
    ``recovery_cmd_active=True`` alone satisfies ``recovery_motion_allowed`` in
    ``core/main.py`` regardless of ``person_detected`` -- so its +/-0.6283 rad ("36 deg")
    ping-pong ``rotation_cmd`` rode straight through the "elif motion_allowed" dispatch to
    ``controller.move`` with ``hold_request`` False, spinning the dog ~12 s at the platform
    edge (it also closed on the patient mid-spin -- fused gap 1.13 -> 0.59 -> 0.31 m at
    sim_t~69.3-70.0, the run's graded ``person_collision``) before it rolled off (roll 98 deg).
    The patient is known to be at the destination ahead; standing still preserves whatever
    standoff the dog last held instead of hunting for a re-acquire near an edge.

    Gated on BOTH ``post_crest_landing_latched`` (``_post_crest_landing_latched`` in
    ``core/main.py``) AND ``fully_on_top_landing`` (``_fully_on_top_landing``'s result) --
    explicit arguments, never a same-frame ``debug_info`` re-read (incident 8.5) -- so this
    can only ever fire once genuinely clear of the stairs, mirroring
    ``lost_person_speed_taper_scale``'s post-crest-only scope (incident 8.15 correction): it
    must NEVER suppress the flat pre-stairs approach lost-search (that one has never caused a
    graded failure) or any mid-climb incident-8.3 blind-carry path, where a person loss is the
    DESIGNED trigger, not a fault.

    No hysteresis here (unlike ``landing_edge_block_latched``): this is meant to toggle
    instantaneously with ``person_detected`` -- the moment the person is re-ranged, normal
    follow must take back over immediately, per the task brief ("release back to normal
    follow the moment the person is re-ranged").
    """
    return (
        bool(post_crest_landing_latched)
        and bool(fully_on_top_landing)
        and not bool(person_detected)
    )


def _depth_from_bbox(depth_img: np.ndarray, bbox: Optional[List[float]]) -> Optional[float]:
    if bbox is None:
        return None
    try:
        depth_m = DepthProcessor.foreground_depth_bimodal(
            depth_img,
            tuple(int(round(v)) for v in bbox[:4]),
            return_histogram=False,
        )
        return None if depth_m is None else float(depth_m)
    except Exception:
        return None


def _depth_from_bbox_excluding_person(
    depth_img: np.ndarray,
    stairs_bbox: Optional[List[float]],
    person_bbox: Optional[List[float]] = None,
) -> Optional[float]:
    """Measure stair depth from the depth image, masking out the person's bbox.

    Uses the 25th-percentile of valid (non-zero) pixels in the stair region
    after zeroing any overlap with the person bbox.  Falls back to the standard
    bimodal method when too few pixels remain after masking.
    """
    if stairs_bbox is None:
        return None
    try:
        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in stairs_bbox[:4]]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        region = np.array(depth_img[y1:y2, x1:x2], dtype=np.float32)

        if person_bbox is not None:
            px1, py1, px2, py2 = [int(round(v)) for v in person_bbox[:4]]
            rel_x1 = max(0, px1 - x1);  rel_y1 = max(0, py1 - y1)
            rel_x2 = min(x2 - x1, px2 - x1); rel_y2 = min(y2 - y1, py2 - y1)
            if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                region[rel_y1:rel_y2, rel_x1:rel_x2] = 0.0

        valid = region[region > 0.0]
        if len(valid) < 10:
            # Too few non-person pixels remain to read the stair edge. When a person
            # is in frame, do NOT fall back to the person-inclusive bbox depth -- that
            # returns the near person as the "stair" depth, which trips stairs_near and
            # engages the climb forward-floor on flat ground (the dog then drives
            # into/past the person). Report unknown; the main loop keeps the last
            # sensor-confirmed stair depth. With no person present, the bbox depth is
            # still a valid stair estimate.
            if person_bbox is not None:
                return None
            return _depth_from_bbox(depth_img, stairs_bbox)

        depth_mm = float(np.percentile(valid, 25))
        return (depth_mm * 0.001) if depth_mm > 0 else None
    except Exception:
        return None


def _crest_reached(
    frame_meta: Optional[Dict[str, Any]],
    debug_info: Dict[str, Any],
    *,
    stairs_ever_confirmed: bool = False,
) -> bool:
    """Whether the dog has reached the crest / a level landing (climb can finish).

    Detects the crest from whichever signal is available, preferring hardware-real ones:

      * ``sensor_imu_pitch`` (rad) / ``sensor_riser_dist_ahead`` (m) -- real sensors the
        Isaac side MAY thread through the frame sidecar; the body pitch flattening (or no
        riser ahead) means the top is reached. Works on the robot.
      * ``stair_demo`` (sim GROUND-TRUTH only) -- the demo phase levelling to the landing or
        the GT body pitch flattening. Fallback for sim where the sensors are absent.

    Read from ``frame_meta`` (populated EARLY in the loop), not ``debug_info["stair_demo"]``
    which the main loop writes LATER in the same frame -- so the old debug_info read here got
    the default and the brief-loss forward floor never cancelled at the crest (incident 8.5).

    ``stairs_ever_confirmed`` (incident 8.15 / F5, 2026-07-11 review of
    run_sim_20260711_195618_941): gates the sim-GT fallback's "flat_follow" / level-pitch arms
    -- see the inline comment at that block for why. Passed as an explicit argument (incident
    8.5) by the caller (``_apply_stair_command_policy`` below), which reads it from
    ``debug_info["stairs_depth_ever_confirmed"]`` -- the codebase's existing "a genuine
    DEPTH-confirmed stair reading has happened at least once this run" latch (set in
    ``core/main.py`` once ``_depth_from_bbox_excluding_person`` returns a real reading,
    written to ``debug_info`` at ``core/main.py:1077``, before this function's caller is
    invoked at ``core/main.py:1145`` -- no ordering hazard). Defaults to False so any other
    caller that does not pass it gets the SAFE (more restrictive) behaviour.

    E3 (2026-07-12, run_sim_20260712_013638_835): ``stairs_ever_confirmed`` alone still let the
    sim-GT fallback's "flat_follow" / level-pitch arms false-latch on the ordinary PRE-stairs
    approach (YOLO/depth confirm the staircase from ~x=1.6, well before the x=2.0 base) --
    those two arms now ALSO require ``climb_episode_evidence`` (the GT robot height,
    ``stair_demo.robot.z_m``, exceeding 0.5 m -- unreachable pre-stairs, reachable a few risers
    into a genuine climb), computed inline from the SAME ``frame_meta["stair_demo"]`` dict this
    function already reads (no new parameter/state). See the inline comment at that block.
    """
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    # 1. Hardware sensors (preferred; work on the robot). Backward-compatible: absent => skip.
    pitch = fm.get("sensor_imu_pitch")
    if pitch is None:
        pitch = debug_info.get("sensor_imu_pitch")
    if pitch is not None:
        try:
            # sensor_imu_pitch is radians; ~5 deg ~= 0.087 rad flat-enough for the landing.
            if abs(float(pitch)) <= 0.0873:
                return True
        except (TypeError, ValueError):
            pass
    riser_ahead = fm.get("sensor_riser_dist_ahead")
    if riser_ahead is None:
        riser_ahead = debug_info.get("sensor_riser_dist_ahead")
    if riser_ahead is not None:
        try:
            # No riser within a tread ahead => crest/landing reached.
            if float(riser_ahead) > 1.0:
                return True
        except (TypeError, ValueError):
            pass
    # 2. Sim ground-truth fallback (stair_demo). Read from frame_meta (populated early).
    #
    # "top_landing" is a reliable, purely x-position-derived phase (see _terrain_phase in
    # world/sim_go2_stairs.py) -- it can ONLY be true past the real crest, so it counts
    # unconditionally and stays ungated (matches the pre-existing behaviour).
    #
    # "flat_follow" and the standalone level-pitch check below are NOT reliable on their own:
    # _terrain_phase also returns "flat_follow", and the GT body pitch is near-level, for the
    # ENTIRE ordinary pre-stairs approach (x < start_x_m - 0.35, true from frame 1 -- long
    # before the dog has ever seen a riser). Both arms' intended meaning was "back on flat /
    # level ground AFTER climbing", not "currently on flat / level ground" -- and this
    # function's only caller already requires stairs_detected=True to even be reached (a
    # distant YOLO sighting is enough), so a brief person loss anywhere on the pre-stairs
    # approach used to read as "crest reached" (run_sim_20260711_195618_941:
    # _post_crest_landing_latched fired at t=48.76s, x=1.27m, phase=="flat_follow",
    # pitch=-0.29 deg -- 4+ metres before start_x_m=2.0).
    #
    # Fix (incident 8.15 / F5): gate BOTH arms on stairs_ever_confirmed (True once real stair
    # evidence -- a genuine depth-confirmed reading, not just a distant sighting -- has been
    # seen at least once this run). A real climb cannot happen without that confirmation
    # happening first -- but this alone does NOT narrow the window enough: YOLO/depth confirm
    # the stairs ON APPROACH from ~x=1.6 (run_sim_20260712_013638_835's t=38.9s frame: x=1.62,
    # phase=="flat_follow", pitch level, stairs_ever_confirmed already True -- 0.38 m before
    # start_x_m=2.0 -- _crest_reached still returned True there with only the F5 gate, arming
    # _post_crest_landing_latched via debug_info["stair_finish_completed"] on the flat
    # approach). E3 fix (2026-07-12): ALSO require ``climb_episode_evidence`` -- the robot's
    # GT world height (``stair_demo.robot.z_m``, already present in frame_meta, no new
    # sensor/state) exceeding 0.5 m. Flat pre-stairs ground reads ~0.25-0.30 m body height (see
    # the t=38.9 frame above: z_m=0.257); the 2.1 m staircase pushes it toward ~2.1-2.4 m a few
    # risers into a genuine climb. 0.5 m is unreachable pre-stairs (0.2+ m of margin over any
    # gait bob) and reachable early in a real climb -- proof a climb episode has ACTUALLY
    # happened, not just that stairs were sighted/confirmed. Being a SAME-FRAME height check
    # (no falling-edge/history state needed -- avoids threading a new caller-owned state
    # dataclass through _apply_stair_command_policy for a single boolean), it also makes the
    # "phase == 'flat_follow'" half of the OR below STRUCTURALLY unreachable: _terrain_phase
    # (world/sim_go2_stairs.py:121-131) returns "flat_follow" ONLY for x < start_x_m - 0.35,
    # and z_m cannot exceed 0.5 m at that same x (the dog has not reached the risers yet) --
    # 8.7 note: the ORIGINAL docstring/comment above claiming this arm's intent was "back on
    # flat ground AFTER climbing" was never actually reachable post-climb either, since
    # _terrain_phase never reverts to "flat_follow" once x >= end_x_m (it returns "top_landing"
    # forever, already handled unconditionally above) -- left in place (not deleted, minimal
    # surgical change) since the height gate now makes it provably inert rather than a latent
    # false-positive, and a different terrain preset could in principle reintroduce a
    # legitimate mid-route flat segment.
    stair_demo = fm.get("stair_demo")
    if isinstance(stair_demo, dict):
        phase = stair_demo.get("phase")
        if phase == "top_landing":
            return True
        robot_gt = stair_demo.get("robot", {}) or {}
        pitch_deg = robot_gt.get("pitch_deg", 0.0)
        climbed_z_m = robot_gt.get("z_m")
        try:
            climb_episode_evidence = (
                climbed_z_m is not None and float(climbed_z_m) > 0.5
            )
            if (bool(stairs_ever_confirmed) and climb_episode_evidence and (
                    phase == "flat_follow" or abs(float(pitch_deg)) <= 5.0)):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _landing_pitch_deg(
    frame_meta: Optional[Dict[str, Any]], debug_info: Dict[str, Any]
) -> Optional[float]:
    """Body pitch (deg), preferring the hardware sensor path over the sim-GT fallback --
    same preference order ``_crest_reached`` uses: ``sensor_imu_pitch`` (rad, from
    ``frame_meta`` then ``debug_info``) first, else ``stair_demo.robot.pitch_deg`` (sim GT).
    ``None`` when neither reading is available this frame.

    Extracted out of ``_fully_on_top_landing`` (2026-07-12, D2 / run-12 review) so
    ``stair_climbing_latch_release_eligible`` below gates on IDENTICAL pitch semantics
    instead of a second, potentially-drifting copy of this extraction. Pure refactor of
    ``_fully_on_top_landing``'s own logic -- no behavior change there.
    """
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    pitch = fm.get("sensor_imu_pitch")
    if pitch is None:
        pitch = debug_info.get("sensor_imu_pitch")
    if pitch is not None:
        try:
            return math.degrees(float(pitch))
        except (TypeError, ValueError):
            pass
    stair_demo = fm.get("stair_demo")
    if isinstance(stair_demo, dict):
        robot = stair_demo.get("robot")
        if isinstance(robot, dict) and robot.get("pitch_deg") is not None:
            try:
                return float(robot["pitch_deg"])
            except (TypeError, ValueError):
                pass
    return None


@dataclass
class LandingMarginState:
    """Caller-owned state for ``_fully_on_top_landing`` (incident 8.15 / F3 third rescope).
    One instance owned by the main loop, mirroring ``standoff_state``/``ClimbGapFilterState``
    -- never a module global.
    """
    landing_entry_x: Optional[float] = None   # GT robot x_m the first CONFIRMED top_landing sample
    level_since: Optional[float] = None       # perf_counter when a continuous level-pitch run began


def _fully_on_top_landing(
    frame_meta: Optional[Dict[str, Any]],
    debug_info: Dict[str, Any],
    state: LandingMarginState,
    *,
    now: float,
    level_deg: float,
    margin_m: float,
    margin_time_sec: float,
) -> bool:
    """Whether the dog is FULLY clear of the stairs (not merely "crest reached").

    Incident 8.15 / F3 third rescope (2026-07-11 review of run_sim_20260711_195618_941): the
    post-crest hold/taper semantics (``core/main.py``'s ``_post_crest_landing_latched``) treated
    "the crest was reached" as "safe to hold/taper". That one-way latch can fire well before the
    dog is actually clear of the stairs:

      * STRADDLE: the dog can stop with front feet on the landing and rear feet still on the last
        riser/tread -- GT phase still reads "staircase" (x below the terrain's ``end_x_m``) with
        a nonzero pitch (that run: x=6.15/6.27, pitch=-8.5 deg). Holding dead-still there is
        mid-staircase, not "on the landing" (CLAUDE.md 8.9 / 8.15: never stance-lock on the
        incline / straddle).
      * PRE-STAIRS FALSE LATCH: ``_crest_reached``'s sim-GT fallback above (this function's
        sibling) also accepts ``phase == "flat_follow"``, which is TRUE for the entire ordinary
        approach BEFORE the stairs are ever reached (``_terrain_phase`` in
        ``world/sim_go2_stairs.py`` returns "flat_follow" for ``x < start_x_m - 0.35``, the same
        string used after a real climb). That run's ``_post_crest_landing_latched`` fired at
        t=48.76 s, x=1.27 m -- 4+ metres and ~40 seconds before the real staircase -- because the
        person was briefly lost during the ordinary flat approach while stairs_detected happened
        to already be true from a distant YOLO sighting.

    Requires BOTH:
      (a) a LEVEL body pitch (``abs(pitch_deg) <= level_deg``), preferring the same hardware
          sensor path ``_crest_reached`` prefers (``sensor_imu_pitch``), and
      (b) a confirmed travel/time margin PAST the crest -- NOT past wherever the (possibly false)
          one-way latch happened to fire, so a false-positive latch does not shortcut this check:
            * Sim GT (``frame_meta["stair_demo"]["phase"]``): tracked fresh every call from the
              GT terrain phase, NOT the latch. The first time phase reads "top_landing" this run
              (a reliable, non-latched, purely x-position-derived signal -- unlike "flat_follow"
              it can ONLY be true past the real crest, see ``_terrain_phase``), the entry x is
              captured once in ``state.landing_entry_m``; the margin is met once the dog has
              travelled ``margin_m`` past THAT point. Reuses the exact GT fields
              (``stair_demo.phase`` / ``stair_demo.robot.x_m``) the crest handback / egress logic
              already reads -- no new sensor.
            * GT present but NOT YET "top_landing" (sim: phase is "flat_follow" /
              "stair_approach" / "staircase"): GT is an AUTHORITATIVE, purely x-position-
              derived signal here (``_terrain_phase``, ``world/sim_go2_stairs.py``) -- it
              already answers "are we on the landing" directly, so ``margin_ok`` is simply
              False, unconditionally. Does NOT fall into the level-pitch proxy below (fixed
              2026-07-12, run-17 audit -- see the incident note below).
            * No GT phase available AT ALL this frame (``frame_meta["stair_demo"]`` absent --
              real hardware, or sim before the sidecar has ever populated): falls back to
              requiring the level-pitch condition to hold CONTINUOUSLY for
              ``margin_time_sec`` seconds (wall-clock, incident 8.6) -- a hardware-portable
              proxy built from the SAME pitch reading, sustained instead of instantaneous,
              per "do not invent a new sensor".

    Incident 8.15-corr / run-17 audit fix (2026-07-12 review): the ORIGINAL code took the
    level-pitch-proxy branch whenever ``phase != "top_landing"``, which conflated "GT is
    unavailable" (the proxy's actual intended trigger) with "GT IS available and says we are
    still on the ordinary flat pre-stairs approach" (``phase == "flat_follow"``). A dog
    stalled anywhere on FLAT GROUND before ever reaching the stairs -- pitch trivially level,
    by definition -- satisfied the proxy's ``margin_time_sec`` (default 1.5 s) sustained-level
    check after standing still for over a second and a half, and this function then
    (incorrectly) reported "fully on the landing" (``post_crest_fully_on_landing=True``)
    despite GT robot x/z showing the dog had never left the stair BASE (run
    run_sim_20260712_0955xx / "run 17": x=1.63, z_m=0.284 -- flat-ground height, not a climb).
    This defeated ``stair_loss_floor_eligible``'s latch-only arm (its
    ``not fully_on_top_landing`` requirement), so dispatch fell through STAIR_LOSS_FLOOR and
    STAIR_APPROACH_COMMIT all the way to a plain ``controller.stop()`` for the rest of the
    run -- runs 15, 16, and 17 all settled identically at x~=1.77, dog stationary, climb never
    engaged. This mirrors ``_crest_reached``'s OWN already-fixed version of the identical
    defect (its ``stairs_ever_confirmed`` / ``climb_episode_evidence`` gates, see its inline
    comment above) -- that fix was never carried over to this sibling function.
    ``_terrain_phase`` also never reverts to "flat_follow"/etc. once genuinely past the crest
    (it returns "top_landing" forever once reached, per ``_crest_reached``'s own 8.7 note
    above), so there is no legitimate post-climb scenario this narrowing could break.

    State is reset (not one-way) whenever the qualifying condition is not currently met, so it
    correctly re-arms if the dog leaves and re-enters a landing-like state.
    """
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    now_f = float(now)

    # (a) Level pitch -- same preference order as _crest_reached (hardware sensor first).
    pitch_deg = _landing_pitch_deg(frame_meta, debug_info)
    if pitch_deg is None:
        # No pitch reading at all this frame -- cannot confirm level. Fail toward "not fully on
        # the landing" (incident 8.8): the taper/hold-easing this gates must not turn on blind.
        return False
    pitch_ok = abs(pitch_deg) <= float(level_deg)

    # (b) Travel/time margin PAST the crest.
    stair_demo = fm.get("stair_demo")
    phase = stair_demo.get("phase") if isinstance(stair_demo, dict) else None
    robot = stair_demo.get("robot") if isinstance(stair_demo, dict) else None
    x_m = robot.get("x_m") if isinstance(robot, dict) else None

    if phase == "top_landing" and x_m is not None:
        state.level_since = None
        try:
            x_f = float(x_m)
        except (TypeError, ValueError):
            return False
        if state.landing_entry_x is None:
            state.landing_entry_x = x_f
        margin_ok = abs(x_f - state.landing_entry_x) >= float(margin_m)
    elif isinstance(stair_demo, dict):
        # Incident 8.15-corr / run-17 audit fix: GT IS available this frame and explicitly
        # says we are not (yet) on the top landing -- do NOT fall into the level-pitch proxy
        # below, which is a hardware-portable stand-in for exactly the case GT cannot answer
        # (see the docstring's incident note). Trusting the weaker proxy here false-positived
        # on a dog stalled anywhere on the flat pre-stairs ground for over margin_time_sec.
        state.landing_entry_x = None
        state.level_since = None
        margin_ok = False
    else:
        state.landing_entry_x = None
        if pitch_ok:
            if state.level_since is None:
                state.level_since = now_f
            margin_ok = (now_f - state.level_since) >= float(margin_time_sec)
        else:
            state.level_since = None
            margin_ok = False

    return bool(pitch_ok and margin_ok)


def stair_loss_floor_eligible(
    *,
    stairs_now: bool,
    stair_climbing_latch: bool,
    person_detected: bool,
    fully_on_top_landing: bool,
) -> bool:
    """Whether ``core/main.py``'s STAIR_LOSS_FLOOR dispatch branch should keep the climb/
    egress moving (forward floor, gap-braked) instead of falling through to a hard
    ``controller.stop()`` stance-lock.

    Incident 8.15 / F5 (2026-07-11 review of run_sim_20260711_195618_941): the branch's
    original entry condition was ``stairs_now`` alone -- a fresh GENUINE on-stairs detection
    within ``--stair-hold-suppress-sec`` (default 4.0 s) of the last confirmed frame. At the
    crest that staleness window is much SHORTER than the climb PERSISTENCE latch
    (``stair_climbing_latch`` / ``ClimbFSM._climbing_persist_until``, up to 6 s and
    self-extending on recent YOLO stair evidence or -- in sim -- the GT ``stair_demo.phase``
    reading "stair_approach"/"staircase" every frame). Once genuine YOLO/depth detection goes
    stale (patient straddling the crest lip, stairs no longer confirmed near) but the
    persistence latch is still on, ``stairs_now`` alone went False well before
    ``stair_climbing_latch`` did, and dispatch fell through STAIR_APPROACH_COMMIT and
    FLAT_LOSS_GLIDE (neither matches a long-stale person loss at the crest) all the way to a
    plain ``controller.stop()`` -- a commanded-zero stance-lock mid-straddle (front feet on
    the landing, rear feet still on the last riser) that CLAUDE.md 8.9 / 8.15 already
    establish is unsafe on an incline/straddle. Trace evidence (sim_t / GT x_m / roll_deg from
    that run's ``vision_main_trace.jsonl``): ``command_trans_x_limited`` pinned at 0.0 with
    ``hold_request=True`` and the three markers that are ONLY reset by the plain-stop branch
    (``stairs_committed_climb_on_loss``, ``stair_approach_commit_active``,
    ``flat_loss_glide_active``, all False) from sim_t=94.5 while roll climbed 4.3 -> 148 deg
    and the robot flipped off the 2.1 m top-landing edge by sim_t=99.5.

    This mirrors ``ClimbFSM.update``'s OWN "STAIR_LOSS_FLOOR" state condition
    (``not person_detected and climbing_latched and not stair_climb_committed and not
    motion_allowed``, ``core/control/climb_fsm.py`` ~L241-243) -- ``debug_info["fsm_state"]``
    already read "STAIR_LOSS_FLOOR" for every one of these frames because the FSM's OWN
    (separately-computed, telemetry-only) condition already reflected this; ``core/main.py``'s
    ACTUAL dispatch elif chain, which the FSM update explicitly does NOT drive (see
    ``ClimbFSM``'s module docstring), never matched it. This function closes that gap by
    giving the real dispatch chain the same eligibility the FSM's label already implied.

    ``fully_on_top_landing`` (the caller's ``_fully_on_top_landing(...)`` result, incident
    8.15 / F3 third rescope) MUST be False for the latch-only path: once genuinely clear of
    the stairs, a continued loss is a "patient walked away on the flat landing" case, not
    incident-8.3 blind-carry, and the post-crest hold/taper path must take back over. This is
    what keeps this eligibility from reopening the 8.15 correction-1 regression (taper-caused
    permanent park at the stair BASE, run 2026-07-11_150906) -- ``stair_climbing_latch`` only
    ever turns True once GENUINE stair evidence has been seen at least once (see
    ``_apply_stair_command_policy`` / the persistence-latch block in ``core/main.py``), so at
    the stair BASE (before any such evidence) this predicate is unreachable via the latch arm,
    and the taper itself is not called from this branch (unaffected either way -- see
    ``lost_person_speed_taper_scale``'s docstring).

    The caller (``core/main.py``) keeps its own ``and not _edge_block`` on this branch --
    a confirmed landing-edge drop-off must always win and is intentionally NOT folded into
    this predicate, mirroring how the pre-existing hard collision/near-field/blind-timeout
    guards inside the branch body are untouched by this change.
    """
    if bool(stairs_now):
        return True
    return (
        bool(stair_climbing_latch)
        and not bool(person_detected)
        and not bool(fully_on_top_landing)
    )


@dataclass
class StairLatchGhostReleaseState:
    """Caller-owned hysteresis state for ``stair_climbing_latch_release_eligible`` (D2 /
    run-12 review, 2026-07-12). One instance owned by the main loop, mirroring
    ``LandingEdgeLatchState`` / ``LandingMarginState`` -- never a module global.
    """
    sustained_since: Optional[float] = None  # perf_counter when all 3 release conditions
                                               # most recently became continuously true


def stair_climbing_latch_release_eligible(
    frame_meta: Optional[Dict[str, Any]],
    debug_info: Dict[str, Any],
    *,
    post_crest_landing_latched: bool,
    stairs_action_active_genuine: bool,
    state: StairLatchGhostReleaseState,
    now: float,
    level_deg: float,
    hysteresis_sec: float,
) -> bool:
    """Whether ``core/main.py`` should force-release ``stair_climbing_latch`` (and, via the
    ``_climbing_latched and not _genuine_stairs`` re-force block, transitively
    ``stairs_action_active``) THIS frame even though the sim-GT ``_on_top_landing`` phase
    check (``core/main.py``, the ``if _on_top_landing:`` block right above this function's
    call site) has not fired.

    RUN-12 EVIDENCE (2026-07-12 review, run_sim trace t=54.x-74.1): the dog sat LEVEL on the
    landing lip (x~6.2-6.3, GT phase still "staircase" -- short of ``end_x_m``, so the
    unconditional ``_on_top_landing`` release never ran) with
    ``debug_info["stairs_action_active"]==True`` but
    ``debug_info["stairs_action_active_genuine"]==False`` and ``stairs_raw_detected`` (YOLO)
    False the whole window -- a depth-only person-as-risers ghost (incident 8.3 class: the
    STANDING PATIENT ~2 m ahead back-projects as a stack of risers) kept re-arming
    ``stair_climbing_latch`` (``core/main.py``'s ``_climbing_persist_until`` block, ~L1222-
    1325) via its ``_depth_climb_engage`` / ``_gt_on_stairs`` entry arms, neither of which
    requires YOLO. With the latch alive, ``stair_loss_floor_eligible``'s latch-only arm kept
    firing (person not detected, ``fully_on_top_landing`` False the whole window -- a HELD dog
    never travels the 0.45 m ``_fully_on_top_landing`` margin) and pulsed the persistence-
    latch forced-climb's forward floor toward the patient (min GT gap 0.549 m, the run's
    graded ``person_collision``) while the landing-edge probe stayed gated off (it requires
    ``not stairs_action_active``, ``core/main.py`` ~L1741) -- 15+ seconds blind to the far
    drop-off with the dog stopped dead on the lip.

    Release requires ALL THREE, each an explicit argument (incident 8.5 -- never a same-frame
    ``debug_info`` re-read of a value this function's own caller has not yet produced):

      1. ``post_crest_landing_latched`` (``core/main.py``'s ``_post_crest_landing_latched``,
         computed earlier this same frame at ~L1715 -- a ONE-WAY latch, so once the crest is
         genuinely reached this stays True for the rest of the run and cannot itself be a
         ghost artifact).
      2. Level pitch, via ``_landing_pitch_deg`` (the SAME hardware-sensor-preferred reading
         and the SAME ``level_deg`` threshold ``_fully_on_top_landing`` uses -- callers should
         pass ``args.landing_margin_level_deg``, default 5.0 deg). INTERPLAY VERIFICATION
         (task brief / CLAUDE.md 8.16-3): ``stair_loss_floor_eligible``'s latch-only arm exists
         specifically to cover a CREST STRADDLE (front feet on the landing, rear feet still on
         a riser) without stance-locking mid-incline (CLAUDE.md 8.9/8.15). A straddle's pitch
         is well outside the 5.0 deg level band -- incident 8.16 run 6 measured -8.5 deg at the
         straddle that motivated ``stair_loss_floor_eligible`` itself, and the F4 crest-
         artifact follow-up separately measured -8.6 to -9.9 deg for 40+ frames right after
         cresting -- so this level-pitch gate can never be satisfied during a straddle and
         cannot defeat ``stair_loss_floor_eligible``'s straddle protection: releasing the latch
         on a genuinely LEVEL lip (this function) and keeping it alive through a STEEP straddle
         (``stair_loss_floor_eligible``) are the same 5.0 deg boundary read two ways, not two
         independent thresholds that could disagree.
      3. NOT ``stairs_action_active_genuine`` (the PRE-OVERRIDE per-frame value stashed at
         ``core/main.py`` ~L1247, before the persistence latch / close-range dropout force
         ``stairs_action_active`` True -- a real, YOLO-corroborated stair reading must keep the
         latch alive; only a ghost-only frame counts toward release).

    ...sustained CONTINUOUSLY for ``hysteresis_sec`` (~1.5 s recommended) via ``state`` -- a
    single noisy frame (a stray genuine detection, a momentary pitch spike) must not release a
    latch that is still covering a real climb tail; ANY frame where the three conditions are
    not ALL true resets the accumulator (mirrors ``ClimbGapFilterState``'s caller-owned-state
    style, not ``landing_edge_block_latched``'s re-arm-and-hold style -- this predicate needs
    the OPPOSITE bias, sustained truth to fire, since a ghost that flickers genuine every
    couple of frames is exactly the ambiguous case that must NOT release the latch on a real
    climb tail).

    Deliberately does NOT require ``_fully_on_top_landing``'s travel/time margin (unlike
    ``stair_loss_floor_eligible``'s own ``fully_on_top_landing`` check) -- run 12's
    ``post_crest_fully_on_landing`` stayed False for the entire t=54-74 window (a dog held
    stationary by the very latch this function releases never accrues the 0.45 m / 1.5 s
    margin), so requiring it here would make this function unreachable in exactly the
    scenario it exists to fix.

    Does not touch the depth detector, YOLO gating (``depth_stair_latch_allowed``), or any
    mid-climb blind-carry path -- CLAUDE.md 8.3's designed person-loss-mid-climb blind-carry is
    untouched: this only fires post-crest (gate 1), a state blind-carry never reaches (the
    committed-climb / persistence-latch / STAIR_LOSS_FLOOR paths never set
    ``_post_crest_landing_latched``).
    """
    ok = bool(post_crest_landing_latched) and not bool(stairs_action_active_genuine)
    if ok:
        pitch_deg = _landing_pitch_deg(frame_meta, debug_info)
        ok = pitch_deg is not None and abs(float(pitch_deg)) <= float(level_deg)
    now_f = float(now)
    if not ok:
        state.sustained_since = None
        return False
    if state.sustained_since is None:
        state.sustained_since = now_f
    return (now_f - state.sustained_since) >= max(0.0, float(hysteresis_sec))


def _apply_stair_command_policy(
    args,
    trans_x_cmd: float,
    rotation_cmd: float,
    debug_info: Dict[str, Any],
    frame_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    if not bool(debug_info.get("stairs_detected", False)):
        debug_info["stairs_action_active"] = False
        return float(trans_x_cmd), float(rotation_cmd)

    # Climb finished -- robot is on the flat TOP LANDING. Release stair mode even with the
    # person still in view, so the normal follow standoff re-engages on flat ground. While
    # stairs_action_active stays True the standoff is BYPASSED (incident 8.9) and the RL
    # climber's lean-on-creep keeps pushing forward with no distance regulation, so the dog
    # crept right up to the standing patient on the landing (observed GT gap 0.33 m vs the
    # 1.0 m follow target -- "collided with patient"). The robot's own top-landing phase is
    # the discriminator: it is DISTINCT from the flat APPROACH (phase "flat_follow", before
    # the stairs), so this only fires AFTER the climb, not before it. On the robot (no
    # stair_demo sidecar) this is a no-op and the sensor-crest path below still applies.
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    _sd = fm.get("stair_demo")
    if isinstance(_sd, dict) and _sd.get("phase") == "top_landing":
        debug_info["stairs_action_active"] = False
        debug_info["stair_finish_completed"] = True
        debug_info["stairs_top_landing_released"] = True
        return float(trans_x_cmd), float(rotation_cmd)

    # Gate: stair behavior requires the person to be actively detected -- EXCEPT for a
    # BRIEF loss while the staircase is already latched. On a brief loss we still hold the
    # forward floor (below) so the climb keeps advancing instead of stranding the policy
    # at vx=0 mid-step, but we suppress centering/recovery yaw: applying yaw amplification
    # without a fresh detection over-rotates the body and falls (the original gate intent).
    # In parkour mode steering is via delta_yaw (the predicted bearing), not this wz, so the
    # robot still aims at the last-known person while the floor keeps it climbing.
    if not getattr(args, "stair_waypoint_test", False) and not bool(debug_info.get("person_detected", False)):
        lost_age = debug_info.get("lost_age_sec")
        lost_grace = debug_info.get("lost_search_timeout_sec")
        brief_loss = (
            lost_age is not None
            and lost_grace is not None
            and float(lost_age) <= float(lost_grace)
        )
        # Bounded stair finish-to-footing: stop early if we have reached flat ground/top or pitch
        # levels off. Detect the crest from frame_meta (populated early) / hardware sensors, NOT
        # debug_info["stair_demo"] which main.py writes LATER this frame -- the old read got the
        # default so this exit never fired and the brief-loss floor kept pushing at the crest
        # (incident 8.5). _crest_reached also works on the robot via sensor_imu_pitch.
        # stairs_ever_confirmed (incident 8.15 / F5) gates _crest_reached's pre-stairs
        # false-positive window -- see that function's docstring. Read from debug_info here
        # (upstream producer: core/main.py:1077, well before this function is called at
        # core/main.py:1145 -- no 8.5 ordering hazard) and passed down explicitly.
        _stairs_ever_confirmed = bool(debug_info.get("stairs_depth_ever_confirmed", False))
        if brief_loss and _crest_reached(
                frame_meta, debug_info, stairs_ever_confirmed=_stairs_ever_confirmed):
            brief_loss = False
            debug_info["stair_finish_completed"] = True
        if not brief_loss:
            debug_info["stairs_action_active"] = False
            debug_info["stairs_gated_no_person"] = True
            return float(trans_x_cmd), float(rotation_cmd)
        debug_info["stairs_gated_no_person"] = False
        debug_info["stairs_brief_loss_floor"] = True
        rotation_cmd = 0.0

    stair_depth_m = debug_info.get("stairs_depth_m")

    # Approach slowdown: as soon as the YOLO model identifies stairs ahead, ease off the
    # throttle so the dog decelerates INTO the staircase instead of charging the base at
    # full follow speed. This runs during the approach -- before a confirmed depth or the
    # near threshold below engage the full climb policy. trans_x_cmd is the fresh follower
    # output each frame, so scaling it here does not compound across frames.
    approach_scale = float(args.stair_approach_speed_scale)
    approach_x = float(trans_x_cmd)
    if approach_x > 0.0 and approach_scale < 1.0:
        approach_x = approach_x * approach_scale
    approach_slowed = approach_x < float(trans_x_cmd)

    # Require at least one sensor-confirmed (non-latched-only) depth reading before
    # engaging the full near climb policy.  This prevents reaction to distant YOLO
    # detections where depth could not be measured -- but still slow the approach.
    if not getattr(args, "stair_waypoint_test", False) and stair_depth_m is None and not bool(debug_info.get("stairs_depth_ever_confirmed", False)):
        debug_info["stairs_action_active"] = False
        debug_info["stairs_gated_no_depth"] = True
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    stairs_near = stair_depth_m is None or float(stair_depth_m) <= float(args.stair_near_distance)
    debug_info["stairs_near"] = bool(stairs_near)
    if not stairs_near:
        debug_info["stairs_action_active"] = False
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    original_x = float(trans_x_cmd)
    original_wz = float(rotation_cmd)

    # Calculate the bounded stair floor before either safety branch so telemetry remains valid
    # when the collision block forces the command to zero.
    max_forward = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
    forward_floor = max(0.0, float(args.stair_forward_floor))
    if max_forward > 0.0:
        forward_floor = min(forward_floor, max_forward)

    # Hard collision floor on stairs: if the smoothed gap drops below the collision floor,
    # zero the drive (no stance-lock -- a blend at speed on the slope nose-dives) so the
    # dog never climbs into the patient.
    _gap_ctrl = debug_info.get("standoff_gap_ctrl_m")
    _climb_block = (
        _gap_ctrl is not None and float(_gap_ctrl) > 1e-3
        and float(_gap_ctrl) < float(args.stair_climb_collision_floor)
    )
    if _climb_block:
        trans_x_cmd = 0.0
        debug_info["stair_follow_collision_block"] = True
    else:
        debug_info["stair_follow_collision_block"] = False
        # Forward floor while climbing: the person-follow PID collapses vx to ~0 once
        # the dog reaches its standoff at the stair base, which strands the (blind) RL
        # policy with no drive to step up. Hold a minimum forward command and cap it at
        # the stair speed limit so the climb keeps advancing instead of parking.
        trans_x_cmd = max(float(trans_x_cmd), forward_floor)
        if max_forward > 0.0 and trans_x_cmd > max_forward:
            trans_x_cmd = max_forward

    # Mid-climb patient-gap speed brake (incident 8.15 / F2). Smoothly caps trans_x_cmd
    # AHEAD of the hard binary _climb_block above (brake_stop_m sits above the collision
    # floor). min(), not a reassignment, so it can only ever pull the command DOWN -- the
    # forward floor above can never push it back past the brake. person_detected is passed
    # explicitly (8.5); its producer is upstream of this whole function (person_follower
    # rebinds debug_info at core/main.py:727, key written follow_controller.py:434, and
    # this policy is called at core/main.py:1145) and this function's own person gate
    # already read the same key above (~L655). Fails toward the brake on an unmeasured gap
    # ONLY while the person is visible (8.8); with the person out of view it stays at full
    # scale -- the brief_loss blind-carry (8.3) must keep advancing, and braking on the
    # not-visible None gap flipped the dog mid-crest (run_sim_20260711_153245_944).
    #
    # Incident 8.15 / F2 hardening: read the FILTERED (rolling-minimum) gap here, NOT
    # _gap_ctrl (the raw standoff_gap_ctrl_m used by the hard _climb_block above) -- the raw
    # live gap is memoryless and a single noisy "far" frame released this brake to full scale
    # (run_sim_20260711_195618_941). The filtered value is produced ONCE per frame by
    # core/main.py (single producer, incident 8.5 -- see filtered_climb_gap_m's docstring)
    # right after _apply_follow_standoff_policy populates standoff_gap_ctrl_m, ahead of this
    # function's call at core/main.py:1145, so it is always fresh by the time this reads it.
    _gap_brake_scale = climb_gap_brake_scale(
        debug_info.get("stair_climb_gap_filtered_m"),
        brake_start_m=float(getattr(args, "climb_gap_brake_start", 1.2)),
        brake_stop_m=float(getattr(args, "climb_gap_brake_stop", 0.85)),
        person_detected=bool(debug_info.get("person_detected", False)),
    )
    _gap_capped = float(max_forward) * _gap_brake_scale
    if trans_x_cmd > _gap_capped:
        trans_x_cmd = _gap_capped
    # Incident E1 (2026-07-12 review of run_sim_20260712_013638_835): the STORED value is the
    # EFFECTIVE floor fraction applied this frame -- 0.0 whenever the hard _climb_block above
    # also fired, not just the raw gap taper -- so a downstream consumer that trusts this single
    # number (core/main.py forwards it to isaac_env as the UDP `gap_brake_scale` payload field,
    # which scales isaac_env's OWN mid-climb `handoff_climb_vx` floor -- see
    # go2_locomotion/locomotion_arbiter.arbitrate_climb_vx / isaac_env._step_go2_locomotion)
    # cannot re-inflate a vx this function already zeroed for a DIFFERENT reason than the taper.
    # In practice these two conditions rarely disagree by construction (brake_stop_m, default
    # 0.85 m, sits above stair_climb_collision_floor, default 0.55 m, per climb_gap_brake_scale's
    # own docstring -- so _gap_brake_scale is already ~0 whenever _climb_block's raw gap is this
    # close), but folding _climb_block in directly removes the reliance on that coincidence.
    debug_info["stair_climb_gap_brake_scale"] = round(
        float(0.0 if _climb_block else _gap_brake_scale), 3)

    # NOT the lost-person forward-speed taper here. The brief_loss window above (person not
    # detected but within lost_search_timeout_sec) is incident-8.3-class designed blind-carry
    # -- the forward floor exists specifically to keep the climb advancing through it. Bleeding
    # that floor toward zero the longer the loss persists is the same shape as incident 8.3's
    # regression at the stair base (run 2026-07-11_150906: parked at x=1.86, robot_settled,
    # climb never engaged) -- this function's own hard gate above (brief_loss/_crest_reached)
    # already bounds how long the floor holds. The taper is scoped to the post-crest /
    # top-landing phase only (see lost_person_speed_taper_scale docstring).

    # Tame yaw on the stairs. The follower's bbox edge/size penalty amplifies the
    # centering error; on a step that becomes a +/-max yaw saw that twists the body
    # and breaks the climb. Apply a small centering deadband, the (sub-unity) stair
    # centering scale, and a lower stair-specific yaw cap.
    rotation_error_deg = debug_info.get("rotation_error_deg")
    yaw_deadband = max(0.0, float(args.stair_yaw_deadband_deg))
    if rotation_error_deg is not None and abs(float(rotation_error_deg)) <= yaw_deadband:
        rotation_cmd = 0.0
        debug_info["stairs_yaw_deadband_active"] = True
    else:
        rotation_cmd = float(rotation_cmd) * float(args.stair_centering_scale)
        debug_info["stairs_yaw_deadband_active"] = False
    stair_rot_max = max(0.0, float(args.stair_rot_max))
    if stair_rot_max > 0.0:
        rotation_cmd = float(np.clip(rotation_cmd, -stair_rot_max, stair_rot_max))

    debug_info["stairs_action_active"] = True
    debug_info["stairs_approach_active"] = False
    debug_info["stairs_forward_floor_mps"] = float(forward_floor)
    debug_info["stairs_speed_limit_mps"] = float(trans_x_cmd)
    debug_info["stairs_trans_x_before"] = original_x
    debug_info["stairs_rotation_before"] = original_wz
    return float(trans_x_cmd), float(rotation_cmd)


def _apply_front_obstacle_gate(
    args,
    trans_x_cmd: float,
    depth_img: np.ndarray,
    debug_info: Dict[str, Any],
) -> float:
    # On the stairs the stair policy owns the forward command, and the staircase
    # itself reads as a near "obstacle" in the central ROI -- gating here would
    # zero the climb's forward floor. Let the stair policy govern instead. The
    # stair_climbing_latch extends this bypass through a stairs-DETECTION dropout while
    # the dog is still physically climbing (otherwise the next riser is read as a
    # blocking wall and the climb command is zeroed -> the dog wedges on the step;
    # run_sim_20260619_134034). The latch is collision-gated upstream.
    if bool(debug_info.get("stairs_action_active", False)) or bool(debug_info.get("stair_climbing_latch", False)):
        debug_info["front_obstacle_gate_active"] = False
        debug_info["front_obstacle_skipped_on_stairs"] = True
        return float(trans_x_cmd)

    if not bool(args.obstacle_stop_enabled) or trans_x_cmd <= 0.0:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    nearest_m, roi_info = DepthProcessor.central_roi_nearest_depth(
        depth_img,
        width_ratio=args.obstacle_roi_width_ratio,
        height_ratio=args.obstacle_roi_height_ratio,
    )
    debug_info["front_obstacle_depth_m"] = nearest_m
    debug_info["front_obstacle_roi"] = roi_info.get("roi")
    debug_info["front_obstacle_valid_pixels"] = roi_info.get("valid_pixels", 0)
    if nearest_m is None:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    # Riser-shape test: a stair riser produces a depth profile where values INCREASE
    # from top row to bottom row (near riser face at top → ground level at bottom).
    # This is the opposite of a flat wall (uniform depth). Suppress the gate when the
    # ROI pattern looks like a riser so 1-frame latch gaps don't zero the climb command.
    try:
        roi = roi_info.get("roi")
        if roi is not None:
            # central_roi_nearest_depth returns roi = (x1, y1, x2, y2) (columns FIRST,
            # then rows), so the crop is depth_img[y1:y2, x1:x2]. The old unpack read it
            # rows-first (ry1,ry2,rx1,rx2 = roi[0..3]) and cropped [x1:y1, x2:y2] with the
            # axes swapped -> an empty/degenerate crop (e.g. depth_img[486:360, 793:662]),
            # so this whole riser-suppression path was dead. Mirror _roi_depth_row_gradient.
            rx1, ry1, rx2, ry2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            _roi_crop = depth_img[ry1:ry2, rx1:rx2]
            if _roi_crop.size > 0:
                # Use mm values directly (depth image is in mm); row-wise minimum depth.
                _row_min = np.array(
                    [_roi_crop[r, _roi_crop[r] > 0].min() if np.any(_roi_crop[r] > 0) else 0
                     for r in range(_roi_crop.shape[0])],
                    dtype=np.float32,
                )
                _valid = _row_min[_row_min > 0]
                if len(_valid) >= 4:
                    # Gradient in mm/row: positive = depth increases toward the bottom
                    # (riser face above, open space below). Threshold ~20 mm/row ≈ a
                    # visible depth gradient across a 0.15 m riser.
                    _grad = float(np.mean(np.diff(_valid)))
                    debug_info["front_obstacle_depth_gradient"] = round(_grad, 1)
                    if _grad > 20.0:
                        debug_info["front_obstacle_gate_active"] = False
                        debug_info["front_obstacle_riser_pattern"] = True
                        return float(trans_x_cmd)
    except Exception:
        pass

    target_depth = debug_info.get("depth_distance_m")
    if target_depth is not None:
        try:
            if float(nearest_m) >= (float(target_depth) - float(args.obstacle_target_clearance)):
                debug_info["front_obstacle_gate_active"] = False
                debug_info["front_obstacle_reason"] = "not_closer_than_target"
                return float(trans_x_cmd)
        except Exception:
            pass

    if nearest_m > args.obstacle_slow_distance:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    original_cmd = float(trans_x_cmd)
    if nearest_m <= args.obstacle_stop_distance:
        trans_x_cmd = 0.0
        scale = 0.0
    else:
        span = max(1e-3, float(args.obstacle_slow_distance) - float(args.obstacle_stop_distance))
        scale = max(0.0, min(1.0, (float(nearest_m) - float(args.obstacle_stop_distance)) / span))
        trans_x_cmd = float(trans_x_cmd) * scale

    debug_info["front_obstacle_gate_active"] = True
    debug_info["front_obstacle_scale"] = float(scale)
    debug_info["front_obstacle_trans_x_before"] = original_cmd
    return float(trans_x_cmd)


def _roi_depth_row_gradient(depth_img: np.ndarray, roi) -> Optional[float]:
    """Mean row-to-row change (mm/row) of the nearest depth down an ROI, or None.

    ``roi`` is the ``(x1, y1, x2, y2)`` tuple ``central_roi_nearest_depth`` returns
    (x = columns, y = rows), so the crop is ``depth_img[y1:y2, x1:x2]``. A POSITIVE
    result means depth increases toward the bottom rows -- the profile of a stair riser
    (near face above, open tread/ground below), not a flat body/wall.
    """
    try:
        if roi is None:
            return None
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        crop = depth_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        row_min = np.array(
            [crop[r][crop[r] > 0].min() if np.any(crop[r] > 0) else 0
             for r in range(crop.shape[0])],
            dtype=np.float32,
        )
        valid = row_min[row_min > 0]
        if len(valid) < 4:
            return None
        return float(np.mean(np.diff(valid)))
    except Exception:
        return None


def _stair_loss_forward_block(
    args,
    depth_img,
    debug_info: Dict[str, Any],
    *,
    committed_climb: bool = False,
    depth_stair_confirmed: bool = False,
    depth_stair_leading_edge_m: Optional[float] = None,
) -> bool:
    """LIVE near-field guard for the person-loss stair forward drive (returns True => block).

    The STAIR_LOSS_FLOOR path drives a modest forward floor UP the stairs when the patient
    lock is lost mid-climb, gated ONLY on the last-known patient gap -- which goes stale
    exactly when it matters: the patient stops on the step just ahead and detection drops,
    so the remembered gap still reads "far" while a body now fills the near field. This adds
    the missing check: read the nearest lower-center depth THIS frame and block the drive
    when something is close ahead that is NOT a stair riser (a body/wall). A real riser reads
    "near" too, so it is distinguished by the row-wise depth gradient and is NOT blocked
    (blocking on every riser would freeze the climb at the base).

    STRUCTURAL LIMIT of the gradient test (2026-07-12 review, run 18 wedge,
    run_sim_20260712_103237_267): at close range (observed <~0.45 m, this guard's own
    ROI) the riser FACE fills the ENTIRE ROI -- there is no tread visible below it to
    produce the downward row-gradient the test above relies on -- so a genuine riser
    reads exactly like a flat wall/body slab (run 18's decisive frame, vision_main_trace
    sim_t=40.88: stairs_loss_nearfield_depth_m=0.304, stairs_loss_nearfield_gradient=
    -1.0, riser=False -- 342 consecutive such frames, dog pressed against riser 1 at
    base_x=1.74, staircase confirmed 2.0 m ahead). The gradient test therefore CANNOT be
    fixed by loosening its own threshold (CLAUDE.md 8.1 "do not refactor architecture" /
    minimal-fix spirit; a looser threshold would also swallow real body/wall reads at the
    same range). Instead, use CONTEXT the gradient test does not have: main.py's OWN
    geometric depth-stair gate (``evaluate_depth_stair_gate``, ~L890-917, always computed
    well before this call site so these are safe same-frame reads per CLAUDE.md 8.5)
    independently confirmed a multi-riser structure the SAME frame
    (``depth_stair_confirmed=True``, ``depth_stair_leading_edge_m=0.302`` in run 18) --
    and that confirmed structure's own leading edge is ~the same surface as this guard's
    near-field return (``|0.304-0.302|=0.002 m``, well inside
    ``NEARFIELD_LEADING_EDGE_AGREE_M``). When the loss window is COMMITTED to a climb
    (``committed_climb`` -- main.py's STAIR_LOSS_FLOOR dispatch branch, this function's
    only caller, passes the literal ``True`` it structurally always is there; see that
    call site) AND a confirmed staircase's own leading edge AGREES with this near-field
    return, classify the surface as the riser we are already climbing, not a wall.
    Requiring BOTH the commit context and an independently-confirmed multi-riser
    structure (not just any near reading) means a genuine body/wall ahead -- which has no
    confirmed staircase to agree with -- is UNAFFECTED and still blocks (fails toward
    blocking on disagreement/missing signals, CLAUDE.md 8.8). Explicit keyword arguments
    only (CLAUDE.md 8.5) -- never re-derived from a downstream debug_info read here.
    """
    if depth_img is None:
        # No depth frame at all this iteration -> the depth pipeline is stale, not garbage.
        # The OTHER loss guards (last-known-gap collision block + detection-age ceiling)
        # still apply, so this guard is a no-op here (matches test_no_depth_frame).
        return False
    try:
        nearest_m, roi_info = DepthProcessor.central_roi_nearest_depth(
            depth_img,
            width_ratio=float(args.obstacle_roi_width_ratio),
            height_ratio=float(args.obstacle_roi_height_ratio),
        )
    except Exception:
        # FAIL CLOSED: a depth frame EXISTS but the near-field probe threw (garbage/malformed
        # depth). We cannot rule out a body/wall close ahead, so block the blind forward drive
        # rather than driving into a possibly-close patient (safety-critical, patient-adjacent).
        debug_info["stairs_loss_nearfield_probe_error"] = True
        debug_info["stairs_loss_nearfield_block"] = True
        return True
    debug_info["stairs_loss_nearfield_depth_m"] = (
        None if nearest_m is None else round(float(nearest_m), 3))
    if nearest_m is None or float(nearest_m) > float(args.obstacle_stop_distance):
        debug_info["stairs_loss_nearfield_block"] = False
        return False
    grad = _roi_depth_row_gradient(depth_img, roi_info.get("roi"))
    if grad is not None:
        debug_info["stairs_loss_nearfield_gradient"] = round(float(grad), 1)
    is_riser = grad is not None and grad > _RISER_GRADIENT_MM_PER_ROW
    # Context override -- see the docstring above for the full run-18 numeric citation.
    # Only reclassifies a would-BLOCK (not-riser) verdict; never overrides a gradient
    # read that already passed as a riser.
    leading_edge_agree = (
        bool(committed_climb)
        and bool(depth_stair_confirmed)
        and depth_stair_leading_edge_m is not None
        and abs(float(nearest_m) - float(depth_stair_leading_edge_m))
        <= float(NEARFIELD_LEADING_EDGE_AGREE_M)
    )
    debug_info["stairs_loss_nearfield_leading_edge_agree"] = bool(leading_edge_agree)
    # APPROACH HOLD-STANDOFF (run 21, run_sim_20260712_112519_790): the override below
    # originally passed the riser at ANY range, letting the loss floor press the dog to
    # 0.30 m of riser 1 -- the wedge_stall engage then hot-swapped the climber into a
    # DEAD-STAND contact mount, which the stage-5 policy cannot execute (climb_stalled
    # after all retries, worst pitch 7.3 deg = never reared). Run 13's successful engage
    # had riser_dist 0.4 / leading_edge 0.451 of run-up. So the pass-as-riser override
    # only applies at or beyond NEARFIELD_RISER_HOLD_STANDOFF_M: nearer than that the
    # block re-asserts and the dog HOLDS at approach distance (inside HandoffController's
    # approach_room window [climb_min_room_m 0.40, climb_engage_standoff_m 0.65]) until
    # the stair-entry lead gate opens, then engages WITH momentum instead of from a press.
    if not is_riser and leading_edge_agree and float(nearest_m) >= float(
            NEARFIELD_RISER_HOLD_STANDOFF_M):
        is_riser = True
    debug_info["stairs_loss_nearfield_riser"] = bool(is_riser)
    blocked = not is_riser
    debug_info["stairs_loss_nearfield_block"] = bool(blocked)
    return blocked
