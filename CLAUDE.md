# Agent Operating Rules

## 1. Purpose of This File
This file contains only:
- Non-obvious project constraints
- Known failure patterns
- Recurring confusion points
- Lessons learned from prior agent mistakes

Do NOT treat this as general documentation.
If something is obvious from reading the codebase, it should NOT be here.

---

## 2. Core Operating Principles

1. Do not assume conventions.
2. Do not refactor architecture unless explicitly instructed.
3. Prefer minimal, surgical changes.
4. Verify before destructive actions (overwrite, delete, replace).
5. When uncertain, ask instead of guessing.
6. do not create Fallbacks/safefials if not requested by the user try to fix the issue
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

This file evolves from real mistakes.

---

## 4. Incident Ledger (Cross-Session Memory)

Each entry must follow this structure:

TRIGGER:
Condition or pattern that activates this rule.

LESSON:
What must or must not be done.

WHY:
Short explanation of failure mode.

---

(Entries are added over time.)

---

## 5. Progressive Disclosure

If working in a specific domain or subdirectory:
- Check for a local AGENTS.md in that directory.
- Local rules override global ones.
- Do not load unrelated domain rules.

---

## 6. What Does NOT Belong Here

Do NOT include:
- Directory trees
- Tech stack summaries
- Style guides
- Obvious best practices
- Anything discoverable from reading the repository
- Long explanations

Keep this file short (<600 lines, ideally <500).

## 7. Special rules
- This repository targets a **live remote robot system running on NVIDIA Jetson Orin**.  right now we are working in the sim version
- Operational commands for model export/conversion/inference must be run on the **robot**, inside the robot's **Docker container** used for runtime, unless the developer explicitly says otherwise.

- Path mapping rule for command guidance:
  - Host repo `src/` is mounted as container working root `/workspace`.
  - When giving runnable commands for runtime tasks, prefer container-relative paths from `/workspace` (for example `python3 misc/convert_to_trt.py ...`), or explicitly state both host and container forms.

- To understand:
  - System architecture
  - Compatible dependencies
  - Runtime environment
  - Available libraries and drivers
  You must review the Jetson environment configuration located in the `/docker` directory.

The `/docker` folder defines the authoritative runtime environment for this project.

## 8. Incident Ledger Entries

TRIGGER:
Developer asks whether a runtime/export command should run on host vs container for robot deployment.

LESSON:
Default to the robot runtime Docker container and state that context explicitly in the first command answer.

WHY:
Host and container have different dependencies/paths; giving host-context commands causes execution confusion and failures.

---

TRIGGER:
Providing command paths without accounting for host-to-container mount remapping.

LESSON:
Provide container-native paths from `/workspace` (or both host+container mappings) for all executable instructions.

WHY:
The same file has different effective roots (`repo/src` on host vs `/workspace` in container), and ambiguous paths lead to incorrect execution location.

---

TRIGGER:
Developer asks to remove a feature from the main loop while preserving future recovery.

LESSON:
Prefer archive-by-move plus compatibility shims (warn + fallback) over hard deletion.

WHY:
This keeps runtime behavior stable now and minimizes reactivation effort later.

---

TRIGGER:
Developer asks to decouple logic into a separate reusable API/module.

LESSON:
Do not leave compatibility wrappers for the decoupled logic in the original module unless explicitly requested.

WHY:
Wrapper leftovers make ownership ambiguous and look like duplicated implementation, causing confusion during review.

---

TRIGGER:
Person-follow behavior stops when target distance is satisfied but the target is still off-axis.

LESSON:
Do not use distance-only completion or hold logic for person-following; require bearing or heading to also be within tolerance.

WHY:
Distance-only completion disables the controller exactly when a nearby target may still require rapid turning to stay in view.

---

TRIGGER:
MPPI yaw or speed tuning appears ineffective even after updating the controller limits.

LESSON:
Check downstream velocity smoothing limits and accelerations whenever changing MPPI velocity bounds.

WHY:
The velocity smoother can silently clip controller outputs, making controller tuning appear broken or ignored.

---

TRIGGER:
Evaluating UsdSkel animations from a remote CDN/Nucleus path asynchronously in code-driven simulation scripts.

LESSON:
Always copy/export remote USD assets to a local directory (e.g., `assets/`) and modify/reference the local copies.

WHY:
USD resolves referenced and nested assets asynchronously. For standalone scripts that query or step skeleton transforms immediately, remote assets result in loading lag where the skeleton falls back to a rest/T-pose during the initial frames of the simulation.

---

TRIGGER:
Reading or trusting the gait/leg HUD (Panels 3-4, the central gait reticle) or the stair_demo `blind_rl.leg_commands` / `swing_legs` telemetry in the sim.

LESSON:
Source leg/gait telemetry from the RL policy's real joint targets via `rl_locomotion_policy.RLLocomotionPolicy.leg_command_summary()` (stored on `Go2LocomotionState.rl_leg_summary` in `_step_go2_locomotion`). Do NOT reintroduce a `current_swing_legs`-style field that the locomotion controller never fills in.

WHY:
The procedural-gait scaffolding that once populated `Go2LocomotionState.current_swing_legs` was removed, but its consumers were left reading the now-dead field (always empty), so the leg/gait HUD silently displayed static/fake data disconnected from the RL policy.

---

TRIGGER:
Removing "dead" gait fields from `Go2LocomotionState` (e.g. `gait_time`, `gait_period`).

LESSON:
`gait_time` and `gait_period` are NOT locomotion gait state -- `set_front_camera_local_pose` uses them to add handheld walking shake to the robot-POV camera. Keep them when trimming procedural-gait fields; only the genuinely unreferenced ones are safe to delete.

WHY:
A field name containing "gait" looked like leftover procedural-gait code, but removing it would break the front-camera shake references in isaac_env.py.

---

TRIGGER:
Changing the XT16 LiDAR polar-profile wire format (the `lidar_profile` UDP sidecar field).

LESSON:
The encoder `sim_lidar_xt16.profile_from_scan` and the decoder `core/lidar_fusion.decode_lidar_profile` are a contract pair across the Isaac->controller UDP boundary; update both together and re-run `tests/test_lidar_fusion.py` (it round-trips the real encode+decode).

WHY:
The two live in different processes (Isaac host vs Docker controller); a one-sided format change silently breaks the in-preview BEV panel and the LiDAR+YOLO distance fusion without an import error.

---

TRIGGER:
Tuning the stair-climb forward command (e.g. adding a stair forward floor in `_apply_stair_command_policy`).

LESSON:
`_apply_front_obstacle_gate` runs immediately after the stair policy in the `core/main.py` loop and will zero/scale the forward command because the staircase reads as a near obstacle in the central depth ROI. It now early-returns when `debug_info["stairs_action_active"]` is set; keep that bypass or any stair forward floor is silently re-zeroed.

WHY:
The two gates are sequential and both write `trans_x_cmd`; the obstacle gate is downstream, so it wins unless it explicitly defers to the stair policy on the stairs.

---

TRIGGER:
Judging whether the robot climbed from `reports/evaluation_summary.txt` / `stair_demo_report.json` (phase, x_m, "drifted/flat_follow").

LESSON:
Those are the SYNTHETIC stair-demo overlay and can report `flat_follow` / a near-spawn `x_m` even when the physics robot actually climbed. Judge real motion from the `fall diagnostic` JSONL stream in `debug/isaac_env.jsonl` (`x`, `pitch`, `policy_cmd=[vx,vy,wz]`), not the reports.

WHY:
The demo telemetry is geometry-exact scene decoration decoupled from the RL physics; trusting it hid that the robot physically climbed ~2 steps then stalled at x≈2.38.

---

TRIGGER:
Reasoning about what perception drives the sim's stair climb (assuming the `_get_analytical_terrain_height` ground-truth signal is the control input).

LESSON:
The live stair trigger is sensor-derived: `stairs_detected` comes from `yolo_stairs_inference` (YOLO-World on RGB, latched in `core/main.py`) and `stairs_depth_m` from the depth camera (`_depth_from_bbox_excluding_person`); `_apply_stair_command_policy` reads only those `debug_info` values. The analytical `_get_analytical_terrain_height` feeds ONLY `_build_stair_demo_telemetry`'s `phase` / `blind_rl.mode` HUD labels (`vertical_assist_mps=0.0` / `body_height_target_m=None`) — it drives no command, RL, or physics. Do not treat it as the climb's perception or "replace" it expecting behavior to change.

NOTE (2026-06-17): the fabricated `demo_4d_elevation_raycast` block — the fake "lidar" the analytical probe used to populate in `stair_demo["lidar"]` (samples/detected/confidence/distance_to_next_riser_m) — was REMOVED. `stair_demo["lidar"]` is now filled only by the real PhysX-raycast XT16 (`sim_lidar_xt16.cast_scan`) in `isaac_env.py`. The synthetic `phase`/`blind_rl.mode` labels remain.

WHY:
The decorative overlay and the live control signal share "stair/raycast" vocabulary, so the synthetic ground-truth telemetry looks like the perception path and leads to redundant or misdirected work (e.g. "make the climb sensor-driven" when it already is).

## 9. Testing it

1. The Test Command
We run the simulation using Isaac Sim’s bundled Python interpreter via a command line instruction. We pass two special arguments to the environment script:

Headless Mode: This runs the simulation in the background without launching a full graphical user interface, which makes it fast and resource-efficient.
Verification Image Path: We specify a target file path for a PNG image.
Exit After Verification: This flag tells the script to capture the image and immediately close down the simulation app, so we don't leave the background process running forever.

2. How the Verification Capture Works
When the command runs, the simulation goes through the following sequence:

World Building: The virtual ground plane, friction settings, stairs, and boundaries are constructed.
Spawning the Actors: The robot dog and the person are imported and placed in the scene.
Settling the Scene: Instead of taking a photo immediately, the simulation runs for 70 steps. During these steps:
A stabilization loop continuously targets a stable standing posture for the robot dog, preventing it from collapsing under gravity.
The animation updater sets the person target's posture so they are in a natural stance instead of their default flat T-pose.
Saving the Image: The camera takes an RGB snapshot of the viewport and saves it directly to the designated PNG file path.

---
