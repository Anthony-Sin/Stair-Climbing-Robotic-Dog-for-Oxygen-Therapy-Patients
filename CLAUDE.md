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
  - The **repo root** is mounted as container working root `/workspace` (`docker/start_follow_system.sh`: `-v "$REPO_ROOT:/workspace" -w /workspace`). There is **no `src/`** dir; a container path is `/workspace/<repo-relative path>`.
  - When giving runnable commands for runtime tasks, prefer container-relative paths from `/workspace` (e.g., `python3 real/models/export_stairs_trt.py ...`), or explicitly state both host and container forms.
- Review the Jetson environment configuration located in the `/docker` directory to understand the system architecture, dependencies, and runtime environment.

---

## 8. Incident Ledger Entries

**TRIGGER:** Giving container-relative paths per the "`src/` -> `/workspace`" phrasing.
**LESSON:** This repo mounts the **repo root** (not a `src/` dir) at `/workspace` (`docker/start_follow_system.sh`); a container path is `/workspace/<repo-relative path>` (e.g. host `real/models/export_stairs_trt.py` -> `/workspace/real/models/export_stairs_trt.py`).
**WHY:** There is no `src/` dir; blindly prefixing `src/` produces wrong container paths.

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

**TRIGGER:** Changing how runs are persisted in `perf_tracker/update_table.py`.
**LESSON:** Ingest only through the public API (`update_table.record_run` / `rebuild_table`); `sim/isaac/terrain_bench/bench_metrics.py` imports `update_table` and breaks if you remove/rename its persistence functions. Bench rows are categorised `bench` and live in `archive.jsonl` only — they are kept OFF the lean `performance_table.csv` leaderboard (a ramp's `max_x_m` is not comparable to a climb).
**WHY:** `bench_metrics` reuses `update_table` internals across the host process; one-sided changes silently break the terrain-bench aggregation.

**TRIGGER:** A run is missing from `performance_table.csv` even though it ran.
**LESSON:** The CSV is the LEAN leaderboard (top actionable runs + all successes). Full history is `archive.jsonl`. `unknown`/`not_recorded`/self-test/bench/instant-fall runs are archived but excluded by design (see `classify_run`). Use `python perf_tracker/update_table.py --rebuild` to re-derive the table from the archive.
**WHY:** The leaderboard is intentionally filtered to actionable runs; the archive is the source of truth.

**TRIGGER:** Adding a subpackage under `core/` (the Docker controller) whose name matches one under `sim/isaac/` (e.g. `perception`).
**LESSON:** `sim/isaac/isaac_env.py` puts `core/` on the Isaac process `sys.path` AHEAD of `sim/isaac/`, so a same-named `core/` subpackage shadows the sim one and breaks Isaac's bare `from perception ... import` (Isaac dies before `world_ready`). Give core subpackages distinct names (core uses `vision`, NOT `perception`) and import core internals fully-qualified (`from core.<pkg>.<mod> import ...`).
**WHY:** Both `core/` and `sim/isaac/` are on the Isaac `sys.path`; a bare same-named subpackage import resolves to whichever dir is first (`core/`), so Isaac can no longer find its own submodule.

**TRIGGER:** Validating a `core/` change quickly via `run_sim.bat --headless`.
**LESSON:** Pass `--max-run-time-sec 90` (+ set `NO_PAUSE=1`) to cap the run (~3 min vs ~10), and judge PASS/FAIL from `log/run_sim_*/logs/launcher.log` (Isaac `world_ready` + controller started + no `Traceback` in `debug/docker_run.log`), NOT the exit code. A capped run exits non-zero (container killed) and `summary` reports `failed`/`sim_gate` because the stair demo never completes — neither is a regression. Logs live under `log/`, not `run_logs/`.
**WHY:** The launcher's timeout-kill returns non-zero and the completion gate can't be reached in a capped run, so the exit code/summary alone gives false failures.

**TRIGGER:** Touching the patient gait phase clock (`biped_anim/locomotion_controller.py`), the patient VISUAL body Z (`_person_visual_z` / `_PERSON_VISUAL_Z_TAU` in `update_person_patrol`), or the ground fn passed to the gait (`spawn_sim_person(ground_height_fn=get_terrain_height)`).
**LESSON:** The patient legs use FOOT-PLANTING IK (`biped_anim/foot_planting.py`). Two coupled invariants make the feet sit on the steps without skating: (1) the gait phase advances by distance-travelled/stride (never wall-clock) so the planted stance foot is world-fixed; (2) the VISUAL root rides the *smoothed DISCRETE tread* (`get_terrain_height`, eased by `_PERSON_VISUAL_Z_TAU`), NOT the smooth nosing ramp — the ramp sits above the treads and a foot cannot reach below the root, so a ramp-following body leaves the feet FLOATING. GT/recorded Z still uses the smooth ramp (`pz`); keep them decoupled. The gait ground-references each foot to `get_terrain_height` under it, so the body z and that fn must agree (both discrete).
**WHY:** Float = body above the treads (feet can't reach down past the root). Skate = phase not distance-synced. Both are kinematic; break either and the symptom returns.

**TRIGGER:** A limb animates the wrong way (mirrored/backward) after the foot-planting IK change.
**LESSON:** The IK outputs the SAME anatomical `JointPose` angles (hip forward +, knee bend +) the old open-loop gait used, so the fix is still a one-line sign flip in `rig._CHANNEL_SIGNS` (`hip`/`knee`/`ankle`), NOT an IK-math change. Leg proportions are auto-measured from standing-pose FK (`rig._measure_leg_geometry`); if geometry can't be measured the gait silently falls back to the old open-loop swing (`biped_rig_ready` log `leg_mode`).
**WHY:** Structure (planting, lift, stepping) is geometry-correct; only the rig's handedness (sign) is unknowable without a visual run.

**TRIGGER:** Importing the locomotion policies / handoff for sim OR real.
**LESSON:** They live in the repo-root `go2_locomotion/` package (moved out of `sim/isaac/locomotion/`), imported as `from go2_locomotion.X import ...` by BOTH the sim (`isaac_env.py`, with `REPO_ROOT` appended to `sys.path`) and the real ROS2 port. There are NO compat shims in the old location. A distinct top-level name (not `locomotion`) avoids the `core/` vs `sim/isaac/` shadowing trap.
**WHY:** The real port must import the shared control code without a `sim/isaac` sys.path hack; one copy means the sim is the regression test for the real robot's policies.

**TRIGGER:** Working on the real Go2 EDU deployment (`real/`), assuming it is empty/stale.
**LESSON:** `real/` is the native-ROS2 port (Foxy): `real/ros2/` nodes (thin rclpy shells), `real/control/` (pure, host-tested: lowstate adapter, dual_policy_runner, lowcmd_builder+CRC, watchdog), `real/perception`, `real/logging`, `real/verification`. `unitree_sdk2` is confined to `real/ros2/sport_startup_node` (Motion-Switcher release) — everything else is pure ROS2 `/lowstate`->`/lowcmd`. The OLD `real/bot/` sdk2py-DDS controller is superseded; `ros2_ws/` Nav2 sidecar is **Humble**, this stack is **Foxy**.
**WHY:** A lot already existed and the transport differs (native ROS2, not raw DDS); re-deriving it wastes time and risks regressions.

**TRIGGER:** Reasoning about the `blind_rl` climb backend as if it climbs stairs.
**LESSON:** `blind_rl` is the rl_sar `go2_robot_lab` policy (`sim/models/locomotion/go2_robot_lab_policy.pt`) — a GENERAL blind proprioceptive WALKER, NOT a stair-trained net. The follow + detect + handoff are solved; the ascent is genuinely unproven/best-effort. Do not represent it as a proven climber; a real blind-parkour net (DreamWaQ++-class) is unavailable.
**WHY:** Overstating the climb capability misleads HIL planning; the dog reliably reaches the stairs but reliable climbing needs a policy that does not yet exist.

**TRIGGER:** Asked to remove `omni.anim.people` / NavMesh from the sim's human/patient character.
**LESSON:** This repo never used `omni.anim.people`. The patient's "fake walking animation that glides through stairs" is the procedural UsdSkel gait in `sim/isaac/biped_anim/` driven along a SCRIPTED waypoint path in `isaac_env.update_person_patrol` (kinematic XY + per-bone gait). Change THAT, not a nonexistent anim.people/NavMesh setup.
**WHY:** Acting on the literal premise wastes time hunting code that isn't there.

**TRIGGER:** Driving the patient (or any humanoid) with the H1 policy from `isaacsim.robot.policy.examples` to climb stairs.
**LESSON:** `H1FlatTerrainPolicy` is the ONLY shipped humanoid policy and it is FLAT-terrain / not terrain-aware: commanded forward into a riser it walks its fixed gait, catches a foot, and face-plants (pelvis 0.91->0.27 m, confirmed on 0.15 m AND 0.06 m risers). It walks/steers fine on flat. Real climbing needs a stair-capable/perceptive humanoid policy that does not ship -- same frozen-policy-climb-limit class as the Go2.
**WHY:** The flat policy reaches the stairs but cannot ascend any riser; do not represent the H1-puppet patient (`world/h1_puppet.py`) as a proven climber.

**TRIGGER:** Running an `isaacsim.robot.policy.examples` robot (experimental warp/torch `Articulation`) inside the sim's classic `isaacsim.core.api.World`.
**LESSON:** They COEXIST without flipping the global backend -- do NOT call `SimulationManager.set_backend("torch")` (it would disturb the frozen Go2/parkour pipeline). Construct the policy before `world.reset()`, lazily `initialize()` it from a `POST_PHYSICS_STEP` callback (guard `is_physics_tensor_entity_valid()`), and read poses via `robot.get_world_poses()[0].numpy()`. Only the physics device is shared. Proven live (`h1_puppet_initialized`) + matches the shipped CPU/numpy H1 unit test.
**WHY:** Flipping the global backend is unnecessary and risks regressing the working classic-World sim.

**TRIGGER:** Adding a pre-policy phase to the Go2 main loop that must appear in the recorded videos (e.g. `--stand-up-from-ground`).
**LESSON:** Recorder capture is gated by `topdown_recording_released` (drives `_record_tick`), which only flips on `scene_motion_released` (first controller command) OR `--no-hold-motion`. In the default follow demo motion is HELD until the first YOLO command, so the whole pre-command window is NOT recorded. To record a pre-command phase you must also release recording for it (the stand-up adds `or _standing_up` to that gate). Also: the fall-watchdog (`robot_fallen_now`/`robot_fall_since_sim_sec`) and the motion clock live INSIDE `if scene_motion_allowed:`, so holding `scene_motion_allowed=False` during the phase safely suspends fall-detection + timeouts (and a low-but-upright body wouldn't trip the low-AND-tilted fall test anyway). The stand-up itself runs in the `not scene_motion_allowed` freeze branch via `_Go2StandUp.tick()`; it seats folded (`GO2_FOLDED_POSE`, stiff 800/40 gains), smoothstep-ramps targets to the standing pose, then hands gains to the policy via `_handoff_drive_gains_to_policy`.
**WHY:** The recorder-release and fall/clock gating are non-obvious couplings; a pre-policy phase silently goes unrecorded in the demo (and could false-trip the fall logic) unless both are accounted for.

**TRIGGER:** Gating any command-suppression (wz/vx kill, hold, etc.) on `debug_info["stairs_action_active"]` as if it means "we are on the stairs".
**LESSON:** `stairs_action_active` is NOT "on the stairs" -- it latches ~2.5 m early and reads the person / flat ground as stairs. In a flat-follow run (`run_sim_20260702_000504`) it was True on 2167/2200 frames while `stair_climb_committed` and `stairs_near` were True on ZERO frames and the dog never climbed. If you must suppress a command only during a genuine climb, gate on `stair_climb_committed` (the FSM climb latch) or `stairs_near` (physically at the riser), NOT `stairs_action_active`.
**WHY:** The Task-3.7 transport wz-clamp keyed on `stairs_action_active` and so zeroed the follower's turn during ordinary flat-ground person-follow: with a person approaching-and-turning the follower asked for a full turn (rotation_cmd=1.0) but wz was forced to 0, the bearing ran out to -58 deg, the person left the FOV and was lost, and the dog spiraled off route (x -4.5 -> 20, "collided with patient"). The committed-climb branch already owns wz=0 during the real climb, so the clamp only needs to cover the prepare / at-riser window.

---

## 9. Testing & Verification

**1. The Test Command & Execution**
- Tests must be executed on the **local computer** via CMD/Command Prompt or PowerShell.
- When you have completed a major task, made significant changes, or need to verify your results, **you must run the following sequence** to execute the simulation:

```powershell
  cd C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\sim
  .\run_sim.bat