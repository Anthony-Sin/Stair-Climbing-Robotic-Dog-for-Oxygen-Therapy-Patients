"""Ad-hoc analyzer for the follow-up-stairs goal.

Finds the newest run_sim_* log dir, parses debug/isaac_env.jsonl, and summarizes the
fall_diag stream so we can judge real motion (per project rule: judge from fall_diag,
not the synthetic demo reports). Focus: where/why the climb stops, collision into the
person (min gap), nose-dive/topple (pitch/roll), and person-tracking continuity.

Usage:  python analyze_follow_climb.py [run_dir]
"""
import sys, os, json, glob, math

SIM = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SIM)


def find_run_dir(arg):
    if arg and os.path.isdir(arg):
        return arg
    cands = []
    for base in (os.path.join(REPO, "log"), os.path.join(SIM, "log"), SIM, os.path.join(SIM, "logs")):
        cands += glob.glob(os.path.join(base, "run_sim_*"))
        cands += glob.glob(os.path.join(base, "**", "run_sim_*"), recursive=True)
    cands = [c for c in set(cands) if os.path.isdir(c)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def f(v, fmt="{:.3f}"):
    if v is None:
        return "-"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run = find_run_dir(arg)
    if not run:
        print("NO RUN DIR FOUND under sim/log or sim/")
        return
    print(f"RUN DIR: {run}")
    jl = os.path.join(run, "debug", "isaac_env.jsonl")
    if not os.path.exists(jl):
        # fall back: search anywhere in the run dir
        hits = glob.glob(os.path.join(run, "**", "isaac_env.jsonl"), recursive=True)
        jl = hits[0] if hits else None
    if not jl or not os.path.exists(jl):
        print(f"NO isaac_env.jsonl under {run}")
        print("dir contents:", os.listdir(run))
        return
    print(f"JSONL: {jl}  ({os.path.getsize(jl)} bytes)")

    diags = []
    other_events = {}
    with open(jl, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            act = (ev.get("event") or {}).get("action") if isinstance(ev.get("event"), dict) else ev.get("action")
            if act is None:
                act = ev.get("event") if isinstance(ev.get("event"), str) else None
            if act == "fall_diag":
                diags.append(ev.get("sim") or ev.get("data") or ev)
            else:
                other_events[act] = other_events.get(act, 0) + 1

    print(f"fall_diag samples: {len(diags)}")
    print("other event actions:", dict(sorted(other_events.items(), key=lambda kv: -kv[1])[:12]))
    if not diags:
        print("!! No fall_diag samples -- robot may not have started. Check console log.")
        return

    def g(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    # Timeline (downsampled to ~40 rows)
    n = len(diags)
    step = max(1, n // 40)
    print("\n  i   t      x      y      h     pitch  roll   yaw    vx   bvx   gap   stairsDet sActive pDet")
    for i in range(0, n, step):
        d = diags[i]
        print("{:4d} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5}   {:>3}      {:>3}    {:>3}".format(
            i,
            f(g(d, "t", "time"), "{:.1f}"),
            f(g(d, "x"), "{:.2f}"), f(g(d, "y"), "{:.2f}"), f(g(d, "h", "height"), "{:.2f}"),
            f(g(d, "pitch", "pitch_deg"), "{:.1f}"), f(g(d, "roll", "roll_deg"), "{:.1f}"),
            f(g(d, "yaw", "yaw_deg"), "{:.1f}"),
            f(g(d, "vx"), "{:.2f}"), f(g(d, "body_vx", "bvx"), "{:.2f}"),
            f(g(d, "gap_m", "gap"), "{:.2f}"),
            str(g(d, "stairs_detected")), str(g(d, "stairs_action_active", "sActive")),
            str(g(d, "person_detected", "pDet")),
        ))

    # Aggregates
    xs = [g(d, "x") for d in diags if g(d, "x") is not None]
    hs = [g(d, "h", "height") for d in diags if g(d, "h", "height") is not None]
    pitches = [abs(float(g(d, "pitch", "pitch_deg"))) for d in diags if g(d, "pitch", "pitch_deg") is not None]
    rolls = [abs(float(g(d, "roll", "roll_deg"))) for d in diags if g(d, "roll", "roll_deg") is not None]
    gaps = [float(g(d, "gap_m", "gap")) for d in diags if g(d, "gap_m", "gap") is not None and float(g(d, "gap_m", "gap")) > 1e-3]
    pdet = [bool(g(d, "person_detected", "pDet")) for d in diags if g(d, "person_detected", "pDet") is not None]

    print("\n=== AGGREGATES ===")
    print(f"  max_x={f(max(xs) if xs else None,'{:.2f}')}  (first riser x~2.0, top landing x~5.6)")
    print(f"  max_h={f(max(hs) if hs else None,'{:.3f}')}  final_h={f(hs[-1] if hs else None,'{:.3f}')}  (stand~0.31, +0.08/step, fallen<0.22)")
    print(f"  max|pitch|={f(max(pitches) if pitches else None,'{:.1f}')}  max|roll|={f(max(rolls) if rolls else None,'{:.1f}')}  (topple > ~30)")
    print(f"  min_gap_to_person={f(min(gaps) if gaps else None,'{:.2f}')}  (collision/overlap if < ~0.3)")
    if pdet:
        frac = sum(1 for p in pdet if p) / len(pdet)
        print(f"  person_detected fraction={frac:.2f} over {len(pdet)} samples")

    # Where did it stall/fall? find the peak-x index and what happened after
    if xs:
        peak_i = max(range(len(diags)), key=lambda i: (g(diags[i], "x") or -1))
        d = diags[peak_i]
        print(f"\n  PEAK-X at i={peak_i}: x={f(g(d,'x'))} h={f(g(d,'h','height'))} "
              f"pitch={f(g(d,'pitch','pitch_deg'),'{:.1f}')} roll={f(g(d,'roll','roll_deg'),'{:.1f}')} "
              f"pDet={g(d,'person_detected','pDet')} sActive={g(d,'stairs_action_active','sActive')}")

    # Person-loss episodes (contiguous pDet False runs) with x at loss
    if any(g(d, "person_detected", "pDet") is not None for d in diags):
        episodes = []
        cur = None
        for i, d in enumerate(diags):
            p = g(d, "person_detected", "pDet")
            if p is False:
                if cur is None:
                    cur = [i, i, g(d, "x")]
                else:
                    cur[1] = i
            else:
                if cur is not None:
                    episodes.append(cur); cur = None
        if cur is not None:
            episodes.append(cur)
        print(f"\n  person-loss episodes (count={len(episodes)}), longest by sample-span:")
        for ep in sorted(episodes, key=lambda e: -(e[1]-e[0]))[:6]:
            span = ep[1]-ep[0]+1
            print(f"    samples [{ep[0]}..{ep[1]}] span={span}  x_at_loss~{f(ep[2],'{:.2f}')}")


if __name__ == "__main__":
    main()
