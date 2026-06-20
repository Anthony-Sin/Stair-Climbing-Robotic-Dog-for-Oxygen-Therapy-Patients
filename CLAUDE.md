# Agent Operating Rules

## 1. Purpose of This File
This file contains only:
- Non-obvious project constraints
- Known failure patterns
- Recurring confusion points
- Lessons learned from prior agent mistakes

Do NOT treat this as general documentation. If something is obvious from reading the codebase, it should NOT be here.

---

## 2. Core Operating Principles
1. Do not assume conventions.
2. Do not refactor architecture unless explicitly instructed.
3. Prefer minimal, surgical changes.
4. Verify before destructive actions (overwrite, delete, replace).
5. When uncertain, ask instead of guessing.
6. Do not create fallbacks/safefails if not requested by the user; try to fix the core issue.

---

## 3. The "Surprise Rule" (Mandatory)
If you encounter:
- Behavior that contradicts common conventions
- Hidden coupling or side effects
- An unexpected failure after a seemingly correct change
- Any ambiguity that required developer clarification

You must:
1. Explicitly notify the developer.
2. Propose a concise addition to the Incident Ledger below.

---

## 4. Incident Ledger (Cross-Session Memory)
Each entry must follow this structure:
- **TRIGGER:** Condition or pattern that activates this rule.
- **LESSON:** What must or must not be done.
- **WHY:** Short explanation of failure mode.

---

## 5. Progressive Disclosure
If working in a specific domain or subdirectory:
- Check for a local `AGENTS.md` in that directory.
- Local rules override global ones.
- Do not load unrelated domain rules.

---

## 6. What Does NOT Belong Here
Do NOT include directory trees, tech stack summaries, style guides, obvious best practices, or long explanations. Keep this file short (<600 lines, ideally <500).

---

## 7. Special Rules
- This repository targets a **live remote robot system running on NVIDIA Jetson Orin** (currently working in the sim version).
- Operational commands for model export/conversion/inference must be run on the **robot**, inside the robot's **Docker container** used for runtime, unless explicitly stated otherwise.
- **Path mapping rule:**
  - Host repo `src/` is mounted as container working root `/workspace`.
  - When giving runnable commands for runtime tasks, prefer container-relative paths from `/workspace` (e.g., `python3 misc/convert_to_trt.py ...`), or explicitly state both host and container forms.
- Review the Jetson environment configuration located in the `/docker` directory to understand the system architecture, dependencies, and runtime environment.

---

## 8. Incident Ledger Entries

**TRIGGER:** Developer asks whether a runtime/export command should run on host vs container for robot deployment.
**LESSON:** Default to the robot runtime Docker container and state that context explicitly in the first command answer.
**WHY:** Host and container have different dependencies/paths; giving host-context commands causes execution confusion.

**TRIGGER:** Providing command paths without accounting for host-to-container mount remapping.
**LESSON:** Provide container-native paths from `/workspace` (or both mappings) for all executable instructions.
**WHY:** Ambiguous paths lead to incorrect execution locations (`repo/src` vs `/workspace`).

**TRIGGER:** Developer asks to remove a feature from the main loop while preserving future recovery.
**LESSON:** Prefer archive-by-move plus compatibility shims (warn + fallback) over hard deletion.
**WHY:** Keeps runtime behavior stable and minimizes reactivation effort.

**TRIGGER:** Developer asks to decouple logic into a separate reusable API/module.
**LESSON:** Do not leave compatibility wrappers for the decoupled logic in the original module unless requested.
**WHY:** Leftovers make ownership ambiguous and cause confusion during review.

**TRIGGER:** Person-follow behavior stops when target distance is satisfied but target is off-axis.
**LESSON:** Do not use distance-only completion; require bearing or heading to also be within tolerance.
**WHY:** Disables controller when a nearby target requires rapid turning to stay in view.

**TRIGGER:** MPPI yaw or speed tuning appears ineffective after updating controller limits.
**LESSON:** Check downstream velocity smoothing limits and accelerations.
**WHY:** The velocity smoother can silently clip controller outputs.

**TRIGGER:** Evaluating UsdSkel animations from remote CDN/Nucleus in code-driven sim scripts.
**LESSON:** Copy/export remote USD assets to a local directory (e.g., `assets/`) and reference local copies.
**WHY:** Remote assets load asynchronously, causing skeleton fallbacks (rest/T-pose) during initial frames.

**TRIGGER:** Reading gait/leg HUD or `stair_demo` telemetry in the sim.
**LESSON:** Source leg/gait telemetry from `parkour_locomotion_policy.ParkourLocomotionPolicy.leg_command_summary()`. Do NOT reintroduce `current_swing_legs`.
**WHY:** Procedural-gait scaffolding was removed; old fields silently display fake data disconnected from the active parkour policy.

**TRIGGER:** Removing "dead" gait fields from `Go2LocomotionState` (e.g. `gait_time`).
**LESSON:** Keep `gait_time` and `gait_period`. They are used by `set_front_camera_local_pose` for handheld walking shake.
**WHY:** Removing them breaks front-camera shake references in `isaac_env.py`.

**TRIGGER:** Changing XT16 LiDAR polar-profile wire format.
**LESSON:** Update encoder and decoder together, then re-run `tests/test_lidar_fusion.py`.
**WHY:** They live in different processes (Isaac host vs Docker). One-sided changes silently break BEV panel and LiDAR+YOLO fusion.

**TRIGGER:** Tuning the stair-climb forward command.
**LESSON:** Keep the `debug_info["stairs_action_active"]` bypass in the obstacle gate.
**WHY:** The downstream obstacle gate will silently zero/scale the forward command because stairs read as near obstacles.

**TRIGGER:** Judging if the robot climbed from demo reports (`evaluation_summary.txt` / `stair_demo_report.json`).
**LESSON:** Judge real motion from the `fall diagnostic` JSONL stream (`debug/isaac_env.jsonl`), not the reports.
**WHY:** Synthetic demo telemetry is decoupled from physics and can falsely report climbs.

**TRIGGER:** Reasoning about perception driving the sim's stair climb.
**LESSON:** Live stair trigger is sensor-derived (`stairs_detected`, `stairs_depth_m`). `_get_analytical_terrain_height` only feeds HUD labels and drives no physics.
**WHY:** Treating synthetic ground-truth as live control input leads to misdirected debugging.

**TRIGGER:** Diagnosing why the parkour dog "surges" on flat ground.
**LESSON:** The surge is caused by the person filling the depth cam at close range. Masking the person out of the depth (`mask_person_in_parkour_depth`) fixes it. Do NOT try to fix this by lowering `--trans-x-max` or tweaking frozen weights.
**WHY:** The policy over-drives because it misreads the near body as terrain.

**TRIGGER:** Changing `--parkour-heading-mode`.
**LESSON:** `vision` (depth self-steer) is the trained default. Both `vision` and `command` modes surge until the person is masked. `command` mode requires a non-zero bearing to remain stable.
**WHY:** The surge is depth-driven, not heading-driven.

**TRIGGER:** Believing a past commit "didn't surge" due to different code/weights.
**LESSON:** Diff the LAUNCHERS, not the weights. The policy/scene are byte-identical; only launch config defaults changed.
**WHY:** Chasing nonexistent code differences wastes time.

**TRIGGER:** Changing the parkour person-mask camera intrinsics or UDP field.
**LESSON:** Update `_BBOX_TO_DEPTH_SCALE_H/V` and the UDP fixed-size datagram together.
**WHY:** One-sided changes silently drop the mask across the process boundary.

**TRIGGER:** Robot overshoots target distance, walks past the person, and takes several seconds to stop.
**LESSON:** The cruise-and-brake model in `person_follower.py` must have THREE zones: far→cruise, at-target→stop (cmd=0), too-close→brake. The old two-zone condition (`distance_error >= -tolerance → cruise`) made the robot always walk forward even when exactly at the target. Also verify kp satisfies `kp >= cruise / (target - tolerance - min_safe_depth)` or braking can never reach zero. Pacing (`follow_pace_distance`) must be set large (10.0) in sim to avoid 1.5 s settle stalls.
**WHY:** Two-zone cruise-and-brake has no "hold at target" state. With target=0.45 m, kp=0.8, the brake formula can never produce zero output before the robot contacts the person.

**TRIGGER:** A sibling module under `sim/isaac/` needs the live `isaac_env` logger (or any `isaac_env` symbol) at runtime.
**LESSON:** Do NOT `from isaac_env import LOGGER`. `isaac_env.py` runs as `__main__` (launched as a script), so importing it by name RE-EXECUTES the whole module — a second `SimulationApp` boot. Use `logging.getLogger("isaac_env")` (loggers are singletons by name; it returns the same retargeted instance).
**WHY:** Importing the entry-point module by name creates a duplicate module object and re-runs its top-level code.

**TRIGGER:** Writing the warm/bench `command.json` (or any JSON a Python Kit reads) from PowerShell.
**LESSON:** Write UTF-8 WITHOUT a BOM (`[System.IO.File]::WriteAllText($p,$json,(New-Object System.Text.UTF8Encoding $false))`). PS 5.1 `Set-Content -Encoding UTF8` prepends a BOM that `json.load(open(path))` rejects. `terrain_bench/run_bench.ps1` does this; the legacy `run_sim.ps1 Write-WarmCommand` still uses `-Encoding UTF8` (latent BOM risk if warm IPC is ever exercised hard).
**WHY:** Python's plain `open()`/`read_text("utf-8")` does not strip a BOM; `json` then fails on the leading `﻿`.

**TRIGGER:** Recording per-episode videos in `--warm-isaac`/`--bench` mode.
**LESSON:** Do NOT pass `-RawVideoPath` (scene_view) at boot. `topdown/lidar/follow` derive from `args.log_dir` (retargeted per episode by `_warm_retarget_logger`), but `scene_view` uses `args.raw_video_path` if set and only falls back to the per-episode `log_dir` when it is empty. A fixed boot path makes every episode overwrite ONE scene_view.mp4.
**WHY:** `raw_video_path = args.raw_video_path or (log_dir-derived)`; a non-empty boot arg pins it.

**TRIGGER:** Driving the dog with a constant forward command (self-test / `terrain_bench`) with the person in the scene.
**LESSON:** Park the person OFF the forward lane (`--person-x -8 --person-y 8`, no `--person-move`). The default spawn `(-3.5, 0)` sits ~1 m ahead of the robot in its path, so an open-loop forward drive collides with it and corrupts the run.
**WHY:** Without a follow controller nothing steers around the person; constant `vx` walks straight into it.

---

## 9. Testing & Verification

**1. The Test Command & Execution**
- Tests must be executed on the **local computer** via CMD/Command Prompt or PowerShell.
- When you have completed a major task, made significant changes, or need to verify your results, **you must run the following sequence** to execute the simulation:

```powershell
  cd C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\sim
  .\run_sim.bat