"""Trace the closed-loop stair climber from the latest run's fall_diag.

Shows, across the run, the climber engagement (scripted_climb), the gait state (swing leg, frozen,
phase), and the body response (x advance, height, roll/pitch) so we can see whether it engages,
propels, climbs, and stays level.
"""
import json
import os
import sys


def main():
    # this file lives at sim/analysis/, so repo root is three levels up
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rd = sys.argv[1] if len(sys.argv) > 1 else open(
        os.path.join(root, "log", "latest_run.txt"), encoding="utf-8-sig").read().strip()
    rows = []
    for line in open(os.path.join(rd, "debug", "isaac_env.jsonl"), encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ev = d.get("event")
        if isinstance(ev, dict) and ev.get("action") == "fall_diag":
            rows.append(d.get("sim", {}))
    print(f"run: {rd}  rows={len(rows)}")
    eng = [s for s in rows if s.get("scripted_climb")]
    print(f"climber engaged (scripted_climb): {len(eng)} / {len(rows)} frames")
    if not eng:
        # show where stairs_action_active was and why climber didn't engage
        act = [s for s in rows if s.get("stairs_action_active")]
        print(f"  stairs_action_active frames: {len(act)} (climber needs these)")
        return

    def g(v):
        return "--" if v is None else v
    print(f"{'t':>6}{'x':>7}{'h':>6}{'roll':>6}{'pitch':>6}{'bvx':>7}{'swing':>6}{'frz':>6}{'ch':>6}{'ph':>6}")
    last_x = None
    for i, s in enumerate(eng):
        if i % 3 and i < len(eng) - 1:
            continue
        c = s.get("climber") or {}
        print(f"{g(round(s.get('t', 0), 2)):>6}{g(s.get('x')):>7}{g(s.get('h')):>6}"
              f"{g(s.get('roll')):>6}{g(s.get('pitch')):>6}{g(s.get('body_vx')):>7}"
              f"{str(c.get('swing')):>6}{str(c.get('frozen'))[:1]:>6}"
              f"{str(c.get('contact_hold'))[:1]:>6}{g(c.get('phase')):>6}")
    dx = (eng[-1].get("x", 0) or 0) - (eng[0].get("x", 0) or 0)
    frozen_frac = sum(1 for s in eng if (s.get("climber") or {}).get("frozen")) / len(eng)
    pitches = [abs(s.get("pitch", 0)) for s in eng]
    rolls = [abs(s.get("roll", 0)) for s in eng]
    print("  ---")
    print(f"  body advance while climbing: dx = {dx:+.3f} m over {len(eng)} frames")
    print(f"  frozen (tilt-gate) fraction: {frozen_frac:.0%}")
    print(f"  |pitch| max={max(pitches):.1f} mean={sum(pitches)/len(pitches):.1f}  "
          f"|roll| max={max(rolls):.1f} mean={sum(rolls)/len(rolls):.1f}")


if __name__ == "__main__":
    main()
