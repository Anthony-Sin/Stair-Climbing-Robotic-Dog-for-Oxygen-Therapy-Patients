"""Quick fall_diag analyzer for stair-climb verification.

Reads the latest run's debug/isaac_env.jsonl and reports the physics ground-truth
that matters for the stair goal: how far the robot climbed, whether it beached/fell,
whether the climb policy engaged and where, follow quality, and patient collision.

Usage:  python analyze_climb.py [path/to/run_dir]
"""
import json
import sys
import os
import glob

STAIR_BASE_X = 2.0
STEP_RUN = 0.305   # commercial run (m); demo_gentle 0.30, residential 0.279

# Verdict thresholds. These MIRROR the isaac_env live fall watchdog so the offline
# verdict and the running sim agree on what counts as a fall vs. an upright stair
# COLLISION (the case the old verdict scored as a clean climb): a robot can plow
# nose-first into the risers and wedge -- upright, never flipping, never reaching the
# top -- and the old max_x-only check called that "CLIMBED: True".
FALL_TILT_DEG = 60.0      # roll/pitch beyond this == flipped/toppled (ROBOT_FALL_TILT_RAD 1.05 rad)
COLLAPSE_H_M = 0.18       # height-above-terrain under this == collapsed/dragging (ROBOT_COLLAPSE_HEIGHT_M)
CLEAN_STAND_H_M = 0.22    # a cleanly-standing climb holds at least this much above the step
UPRIGHT_TILT_DEG = 18.0   # a clean climb keeps |roll| and |pitch| within this band
NOSE_DOWN_MEAN_DEG = -8.0 # mean on-stairs pitch at/under this == persistently plowing nose-first
NOSE_DOWN_MIN_DEG = -22.0 # a single nose-dive this deep into a riser == a collision


def latest_run_dir():
    # this file lives at sim/analysis/, so repo root is three levels up
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ptr = os.path.join(root, "log", "latest_run.txt")
    if os.path.exists(ptr):
        with open(ptr, encoding="utf-8-sig") as f:
            d = f.read().strip()
            if d and os.path.isdir(d):
                return d
    cands = glob.glob(os.path.join(root, "log", "run_sim_*"))
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else latest_run_dir()
    if not run_dir:
        print("no run dir found")
        return
    jl = os.path.join(run_dir, "debug", "isaac_env.jsonl")
    print(f"run: {run_dir}")
    rows = []
    for line in open(jl, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        if isinstance(ev, dict) and ev.get("action") == "fall_diag":
            rows.append(d.get("sim", {}))
    if not rows:
        print("no fall_diag rows")
        return

    xs = [s.get("x", -99) for s in rows]
    max_x = max(xs)
    final = rows[-1]
    # spiral / off-axis detection
    ys = [abs(s.get("y", 0)) for s in rows]
    max_abs_y = max(ys)
    yaws = [s.get("yaw", 0) for s in rows]
    yaw_span = max(yaws) - min(yaws)
    # yaw near the stair base (x in [1.9, 2.2]) = entry alignment (0 = head-on)
    entry = [abs(s.get("yaw", 0)) for s in rows if 1.9 <= s.get("x", -99) <= 2.25]
    entry_yaw = (sum(entry) / len(entry)) if entry else None
    # climb engagement
    eng = [s for s in rows if s.get("stairs_action_active")]
    eng_x = eng[0].get("x") if eng else None
    # on-stairs window (x past base)
    on = [s for s in rows if s.get("x", -99) >= STAIR_BASE_X - 0.05]
    on_stairs = bool(on)
    min_h_on = min((s.get("h", 9) for s in on), default=None)
    pitches = [s.get("pitch", 0) for s in on]
    rolls_on = [abs(s.get("roll", 0)) for s in on]
    mean_pitch_on = (sum(pitches) / len(pitches)) if pitches else None
    min_pitch_on = min(pitches) if pitches else None
    max_abs_roll_on = max(rolls_on) if rolls_on else None
    # Peak body tilt over the WHOLE run. A genuine FALL is a large roll OR pitch
    # (flip/topple). A stair COLLISION is the opposite: the body stays roughly
    # upright (tilt well under the fall line) but plows nose-first into the risers,
    # so its pitch is persistently nose-DOWN and it DRAGS low as it wedges.
    max_tilt = max(
        (max(abs(s.get("roll", 0)), abs(s.get("pitch", 0))) for s in rows), default=0.0
    )
    # follow / collision
    gaps = [s.get("gap_m") for s in rows if s.get("gap_m") is not None]
    min_gap = min(gaps) if gaps else None
    steps_climbed = max(0.0, (max_x - STAIR_BASE_X)) / STEP_RUN

    print(f"  rows={len(rows)}  sim_t={final.get('t')}")
    print(f"  max_x={max_x:.3f}  final_x={final.get('x')}  => ~{steps_climbed:.1f} step-runs past base (x=2.0)")
    print(f"  final h={final.get('h')}  pitch={final.get('pitch')}  roll={final.get('roll')}")
    print(f"  stairs_action_active first True at x={eng_x}  (engagements={len(eng)})")
    if on:
        print(f"  on-stairs(x>=1.95): min_h={min_h_on:.3f}  "
              f"pitch[min={min(pitches):.1f}, mean={mean_pitch_on:.1f}, max={max(pitches):.1f}]  "
              f"|roll|max={max_abs_roll_on:.1f}")
    print(f"  min gap_m to patient = {min_gap}  (collision floor 0.65; <0.65 = too close)")
    print(f"  max |y| off-axis = {max_abs_y:.2f} m  (>1.0 = spiralled/wandered)")
    print(f"  yaw span = {yaw_span:.0f} deg  (>180 = did a loop)")
    print(f"  mean |yaw| at stair entry (x 1.9-2.25) = {entry_yaw if entry_yaw is None else round(entry_yaw,1)} deg  (0=head-on, >20=crooked)")
    print(f"  max body tilt over run = {max_tilt:.1f} deg  (>{FALL_TILT_DEG:.0f} = flipped/toppled)")
    print("  ---")
    # climb height profile: world z gain (absolute base z) vs x, to see step-by-step ascent
    zs = [(s.get("x"), s.get("h")) for s in rows if s.get("x", -9) >= 1.9]
    if zs:
        # h is base height above terrain directly below; on a clean climb it stays ~0.3 while x advances
        # up the steps. Report base height ABSOLUTE via... (h is above-terrain, so use it as stability)
        hs = [h for _, h in zs if h is not None]
        if hs:
            print(f"  on-stairs height-above-terrain: stayed in [{min(hs):.2f},{max(hs):.2f}] "
                  f"(healthy ~0.30; <{COLLAPSE_H_M} = collapsed/dragging)")

    # ---- HONEST verdict: FELL vs COLLIDED vs CLEAN CLIMB ----------------------
    # The OLD verdict was `climbed = max_x >= base + 0.6` -- a pure forward-distance
    # check. So a dog that nose-dived into the steps and wedged (upright, never
    # flipped, never reached the top, carrying the O2 tank) scored CLIMBED=True. The
    # three outcomes below are physics-distinct and read straight off the fall_diag
    # trajectory (x, h, roll, pitch); no synthetic/demo telemetry is trusted.
    reached_stairs = max_x >= STAIR_BASE_X - 0.05
    # genuine flip/topple: body tilt blew past the fall threshold.
    fell = max_tilt > FALL_TILT_DEG
    # collision/wedge: stayed roughly upright but plowed into the risers --
    # persistently nose-down and/or the body dragged low on the steps.
    nose_diving = on_stairs and (
        (mean_pitch_on is not None and mean_pitch_on <= NOSE_DOWN_MEAN_DEG)
        or (min_pitch_on is not None and min_pitch_on <= NOSE_DOWN_MIN_DEG)
    )
    dragging = on_stairs and (min_h_on is not None and min_h_on < COLLAPSE_H_M)
    collided = (not fell) and on_stairs and (nose_diving or dragging)
    # final pose: standing upright at a healthy height (not nose-down, not dragging).
    final_h = float(final.get("h", 0) or 0)
    final_pitch = abs(float(final.get("pitch", 0) or 0))
    final_roll = abs(float(final.get("roll", 0) or 0))
    final_upright = (
        final_h >= CLEAN_STAND_H_M
        and final_pitch <= UPRIGHT_TILT_DEG
        and final_roll <= UPRIGHT_TILT_DEG
    )
    # CLEAN climb: advanced well past the base, never flipped, never plowed/dragged,
    # and ended standing upright at a healthy height.
    clean_climb = (
        reached_stairs
        and max_x >= STAIR_BASE_X + 2 * STEP_RUN
        and not fell
        and not collided
        and final_upright
    )
    spiralled = max_abs_y > 1.0 or yaw_span > 180

    if fell:
        verdict = "FELL (flipped/toppled -- body tilt exceeded the fall threshold)"
    elif collided:
        why = []
        if nose_diving:
            why.append(f"nose-down (mean pitch {mean_pitch_on:.1f} deg)")
        if dragging:
            why.append(f"dragging (min h {min_h_on:.2f} m)")
        verdict = "COLLIDED WITH STAIRS (upright, did NOT fall, did NOT cleanly climb): " + ", ".join(why)
    elif clean_climb:
        verdict = "CLEAN CLIMB (upright, healthy height, advanced up the steps)"
    elif not reached_stairs:
        verdict = "DID NOT REACH STAIRS"
    else:
        verdict = "INCOMPLETE (reached stairs, no clean top-out and no clear fall/collision)"

    print(f"  VERDICT: {verdict}")
    print(f"    fell(flip)={fell}  collided={collided}  clean_climb={clean_climb}  "
          f"final_upright_standing={final_upright}  spiralled={spiralled}")
    # collision during the climb only (ignore spawn-instant transient near x=-4.5)
    climb_gaps = [s.get("gap_m") for s in rows if s.get("gap_m") is not None and s.get("x", -99) > -3.5]
    min_climb_gap = min(climb_gaps) if climb_gaps else None
    print(f"  PATIENT collision risk (gap<0.65, x>-3.5): {bool(min_climb_gap is not None and min_climb_gap < 0.65)}  (min={min_climb_gap})")


if __name__ == "__main__":
    main()
