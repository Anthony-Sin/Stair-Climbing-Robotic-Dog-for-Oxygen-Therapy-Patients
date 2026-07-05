# Follow / Landing / Climb — Session Summary (2026-07-05)

Work on the sim demo: patient walks up the stairs, the Go2 follows and climbs behind it,
then follows on the top landing. Everything below is sim (`run_sim.bat` / Isaac + Docker).

## How to run
```powershell
cd C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\src\sim
.\run_sim.bat
```
For automated/headless runs that must **auto-close** (so a background shell completes instead
of hanging on the `Press any key to continue` prompt): set `NO_PAUSE=1` first, e.g.
```powershell
$env:NO_PAUSE='1'; cd ...\src\sim; .\run_sim.bat
```
`run_sim.bat`'s final `pause` is gated by `if not "%NO_PAUSE%"=="1" pause`.

## Stair geometry (default preset)
- start_x_m = 2.0, end_x_m = 6.27, 14 steps, step_depth ≈ 0.305 m, step_height ≈ 0.15 m (~26° incline).
- The **last tread (≈5.97–6.27) is at the same height as the landing** (top_height). Landing slab
  extends past end_x_m (`SIM_LANDING_DEPTH_M`).
- Robot spawns at x = go2_x (≈ −4.5).

## FIXED this session
- **Patient "floating"** before Docker + at destination: the visible mannequin hovered because
  the hold/settle paths placed the root at bind-pose height without the foot-grounding the walking
  path uses, AND the gait was stuck in a mid-stride pose. Fixes: shared `_ground_patient_feet`,
  `SimPersonTarget._last_moving_time` default `0.0 → -1e9` (idle sentinel), pass the continuous
  `sim_clock_sec` to the pre-Docker hold so the gait crossfade settles to idle.
- **Startup smoothing**: `_startup_motion_ramp()` smoothstep over 0.8 s applied to BOTH the patient
  walk velocity and the robot's applied follow command, so they ease into motion together.
- **Auto-close**: use `NO_PAUSE=1` (above).
- **Gap-aware patient pacing** (`update_person_patrol`): the patient watches the robot's GT pose
  (`_robot_gt_xy`) and slows when the dog falls behind (COMFORT 1.6 → floor at MAX 2.7). Prevents
  the patient outrunning the climb and starving the climber (was: dog wedged at 4.79 while patient
  at 8.12).
- **Climb REGRESSION I introduced, then fixed**: a fixed-position "crest-wait" let the patient walk
  to a fixed point past the crest and pull ~2.3 m ahead of the still-climbing dog → climber lost its
  proxy → wedged mid-stairs at x~5.1. Replaced with a **lead-based hard-wait** (`PATIENT_HARD_WAIT_LEAD_M
  = 1.7`): patient STOPS if the dog is >1.7 m behind, capping the lead the whole climb.
- **Landing follow / no collision**: `_apply_no_reverse_follow_policy` suppresses reverse and the sim
  clamps `vx>=0`, so the dog can't hold its standoff against its own forward lean-on-creep on the flat
  landing. Added a **landing creep-brake** (isaac_env, gated on `stair_phase_now=="top_landing"` +
  perceived gap ≤ `LANDING_HOLD_GAP_MULT`×target) that zeroes forward creep so the dog holds ~1.0 m and
  keeps following the person. Plus a **top-landing stair-mode release** (stair_policy.py + main.py:
  `stair_demo.phase=="top_landing"`) so the follow standoff re-engages on the flat.
- **Exit condition** changed per user: removed the "robot reached top-landing waypoint" exit; the run
  now ends on **robot SETTLED** (stationary <8 cm for 15 s) **or a fall**. The settle exit is
  suppressed while the robot is still on the staircase (so it doesn't end mid-climb "without falling").
- **GT crest handback** (handoff_controller.py): the crest used to need BOTH the depth detector AND the
  GT terrain to read flat-ahead; a nose-down dog's depth never clears, so it never crested. Now the GT
  terrain read (`stairs_ahead_gt`) is trusted ALONE when available (GT-false is impossible on the incline,
  so incident 8.8 is preserved). Plus the crest/give-up exits now also require the dog to be **level**
  (`crest_level`, tilt ≤ ~0.17 rad) — keep the blind-RL climber until it is BOTH clear of stairs AND
  upright, THEN hand to PGTT (per user: don't switch to PGTT until fully off the stairs and not angled).

## OPEN BLOCKER — the climb itself (incident 8.9 climber-gait)
The blind-RL climber (`--handoff-climb-backend blind_rl`, proprioceptive) climbs **nose-down** the whole
staircase (pitch −16° to −24°) and **jams on a riser** partway up. Confirmed from run 002353_787 while
stuck at x=5.76 for ~76 s:
- cmd `vx = 0.383` (full stair speed) the whole time — command is HEALTHY, never drops.
- `gap_m ≈ 1.5 m`, `person_detected: True` — NOT a lost-person / distance problem.
- **`body_vx ≈ 0`** (jittering ±0.1) — legs cycling ("looks like climbing") but ZERO net forward progress.
- `pitch ≈ −20°` — nose-down front plowing into the riser, can't lift onto the step.
- **1 policy swap only** (PGTT→climber); it NEVER switched to PGTT on the stairs.

So it is NOT a command / distance / handoff issue — it is the climber-gait **posture**: it brute-forces
the lower steps nose-down, the pitch worsens, and by ~step 12 the nose-down front can't clear the next
riser. Stochastic in *how far* it gets: seen 5.79 / 6.26 / 7.8 across runs.

## Candidate next steps (not yet done)
- **Lower the riser height** so the nose-down climber can still clear each step (fit the stairs to the gait).
- Reduce the climber's forward pitch (harder — frozen RL policy).
- These are the only levers that change the *outcome*; re-running only changes how far it gets.
