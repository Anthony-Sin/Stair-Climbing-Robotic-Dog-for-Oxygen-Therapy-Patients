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


def latest_run_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    min_h_on = min((s.get("h", 9) for s in on), default=None)
    pitches = [s.get("pitch", 0) for s in on]
    # follow / collision
    gaps = [s.get("gap_m") for s in rows if s.get("gap_m") is not None]
    min_gap = min(gaps) if gaps else None
    steps_climbed = max(0.0, (max_x - STAIR_BASE_X)) / STEP_RUN

    print(f"  rows={len(rows)}  sim_t={final.get('t')}")
    print(f"  max_x={max_x:.3f}  final_x={final.get('x')}  => ~{steps_climbed:.1f} step-runs past base (x=2.0)")
    print(f"  final h={final.get('h')}  pitch={final.get('pitch')}  roll={final.get('roll')}")
    print(f"  stairs_action_active first True at x={eng_x}  (engagements={len(eng)})")
    if on:
        print(f"  on-stairs(x>=1.95): min_h={min_h_on:.3f}  pitch[min={min(pitches):.1f}, max={max(pitches):.1f}]")
    print(f"  min gap_m to patient = {min_gap}  (collision floor 0.65; <0.65 = too close)")
    print(f"  max |y| off-axis = {max_abs_y:.2f} m  (>1.0 = spiralled/wandered)")
    print(f"  yaw span = {yaw_span:.0f} deg  (>180 = did a loop)")
    print(f"  mean |yaw| at stair entry (x 1.9-2.25) = {entry_yaw if entry_yaw is None else round(entry_yaw,1)} deg  (0=head-on, >20=crooked)")
    # verdict heuristics
    spiralled = max_abs_y > 1.0 or yaw_span > 180
    beached = (max_x < STAIR_BASE_X + 0.5) and (min_h_on is not None and min_h_on < 0.22)
    climbed = max_x >= STAIR_BASE_X + 0.6  # cleared ~2 step-runs
    print("  ---")
    # climb height profile: world z gain (absolute base z) vs x, to see step-by-step ascent
    zs = [(s.get("x"), s.get("h")) for s in rows if s.get("x", -9) >= 1.9]
    if zs:
        # h is base height above terrain directly below; on a clean climb it stays ~0.3 while x advances
        # up the steps. Report base height ABSOLUTE via... (h is above-terrain, so use it as stability)
        hs = [h for _, h in zs if h is not None]
        if hs:
            print(f"  on-stairs height-above-terrain: stayed in [{min(hs):.2f},{max(hs):.2f}] (healthy ~0.30; <0.18 = collapsed/dragging)")
    print(f"  CLIMBED(>=2 runs past base): {climbed}")
    print(f"  BEACHED at base: {beached}")
    print(f"  SPIRALLED/off-axis: {spiralled}")
    # collision during the climb only (ignore spawn-instant transient near x=-4.5)
    climb_gaps = [s.get("gap_m") for s in rows if s.get("gap_m") is not None and s.get("x", -99) > -3.5]
    min_climb_gap = min(climb_gaps) if climb_gaps else None
    print(f"  COLLISION risk after spawn (gap<0.65, x>-3.5): {bool(min_climb_gap is not None and min_climb_gap < 0.65)}  (min={min_climb_gap})")


if __name__ == "__main__":
    main()
