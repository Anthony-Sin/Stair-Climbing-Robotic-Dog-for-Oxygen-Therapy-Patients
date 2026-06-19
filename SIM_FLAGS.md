# Isaac Sim — Flag Reference

> **Keep this file in sync.** Whenever a flag is added or removed in any of the three source files below, update the matching section here.
>
> **Source files:**
> - `sim/run_sim.ps1` — launcher / orchestration flags (PowerShell)
> - `sim/isaac/isaac_env.py` — Isaac Sim environment flags (Python argparse)
> - `core/args_parser.py` — Docker vision/control container flags (Python argparse)

---

## Quick-Reference: Common Launch Recipes

All recipes run from the project root (or the `sim/` subdirectory — `run_sim.bat` resolves paths relative to itself).

```powershell
# ─── STANDARD ────────────────────────────────────────────────────────────────

# Normal run — GUI, full RTX, Docker controller starts automatically after Isaac is ready
.\run_sim.bat

# Headless — no Isaac GUI window (fastest for CI / unattended runs)
.\run_sim.bat --headless

# Headless + fast render — lighter RaytracedLighting renderer, skips RTX pathtrace
.\run_sim.bat --headless --fast-render

# ─── WARM MODE (keep Isaac alive between runs, pay ~120s boot only once) ─────

# First run: boots Isaac in warm mode (pays the boot)
.\run_sim.bat --headless --fast-render --warm

# Subsequent runs: reuse the live Isaac (instant, ~0s boot)
.\run_sim.bat --headless --fast-render --warm

# Shut down the warm Isaac cleanly
.\run_sim.bat --warm-shutdown

# ─── DOCKER IMAGE MANAGEMENT ─────────────────────────────────────────────────

# Force rebuild the Docker image (e.g. after Python dependency changes)
.\run_sim.bat --force-build

# Skip image existence check and build entirely (use when image is known good)
.\run_sim.bat --skip-build

# ─── SELF-TEST (no Docker / controller needed) ────────────────────────────────

# Drive the parkour policy at 0.5 m/s for 15 s, then auto-exit
.\run_sim.bat --self-test-walk

# Same but with a custom speed and duration
.\run_sim.bat --self-test-walk --self-test-vx 0.3 --self-test-sec 20

# Self-test WITHOUT policy — verifies physics / PD gains can stand the robot alone
.\run_sim.bat --self-test-walk --self-test-no-policy

# ─── PAYLOAD / SCENE ─────────────────────────────────────────────────────────

# Attach the O2 concentrator cradle to the robot's back
.\run_sim.bat --with-o2-payload

# Upgraded hospital demo scene
.\run_sim.bat --final-scene

# ─── A/B COMPARISONS ─────────────────────────────────────────────────────────

# Disable person depth-mask (shows the old surge behavior for comparison)
.\run_sim.bat --no-parkour-person-mask

# Legacy far-fill mask instead of terrain-inpaint
.\run_sim.bat --parkour-mask-fill far

# Parkour (jumping) gait instead of default walk gait
.\run_sim.bat --no-parkour-walk-mode

# Disable speed governor (exposes raw policy speed, may surge/jump)
.\run_sim.bat --no-speed-governor

# ─── SIM-TO-REAL VALIDATION ──────────────────────────────────────────────────

# Full realism preset: D435 depth noise, proprio noise+latency, DR, LiDAR noise
.\run_sim.bat --sim2real-validation-cam

# Same, with 60 ms / ±20 ms latency (auto-applied by --sim2real-validation-cam)
.\run_sim.bat --sim2real-validation-cam --sim-latency-ms 60 --sim-latency-jitter-ms 20

# ─── DEBUGGING ───────────────────────────────────────────────────────────────

# Dry run — print what would be launched without actually running anything
.\run_sim.bat --dry-run

# Keep 5 past run-log folders instead of pruning to 1
.\run_sim.bat --keep-run-logs 5

# Open vision preview window in Docker (disabled by default in sim)
.\run_sim.bat --vision-preview

# Pause at the manual "Isaac is ready" gate before starting Docker
.\run_sim.bat --pause-after-isaac

# Stair square-up alignment on approach (experimental; validate mask first)
.\run_sim.bat --stair-square-up
```

---

## 1  Launcher Flags (`run_sim.bat` / `run_sim.ps1`)

These flags are handled by the PowerShell launcher. They control build, startup sequencing, warm mode, and which sub-systems run.

### 1.1  Build

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-build` | off | Skip Docker image build entirely — assume the image already exists. |
| `--force-build` | off | Always rebuild the Docker image even if it already exists. |

### 1.2  Execution Control

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Print every command that _would_ run; execute nothing. Useful for inspecting the full launch sequence. |
| `--no-isaac` | off | Skip launching Isaac Sim entirely (still runs Docker if enabled). |
| `--no-docker-run` | off | Skip launching the Docker vision/control container. Isaac still runs alone. |
| `--no-model-preflight` | off | Skip the TensorRT engine pre-flight check that normally gates Docker start. |
| `--no-isaac-ready-wait` | off | Don't wait for Isaac's `world_ready` event before starting Docker. |
| `--isaac-ready-timeout-sec <sec>` | `420` | Seconds to wait for Isaac `world_ready` before aborting. |
| `--max-run-time-sec <sec>` | `900` | Kill the Docker controller after this many seconds (15 min default). |
| `--sim-frame-timeout-exit-sec <sec>` | `30.0` | Docker exits if no Isaac camera frame arrives for this long. |

### 1.3  Warm Mode

Warm mode keeps one Kit/Isaac process alive across runs so the ~120 s RTX boot is paid only once. Each `.\run_sim.bat --warm` call that finds a live Kit reuses it instantly.

| Flag | Default | Description |
|------|---------|-------------|
| `--warm` / `--warm-isaac` | off | Enable warm-iteration mode. |
| `--warm-shutdown` | — | Send a shutdown command to a running warm Isaac, then exit immediately. |
| `--warm-max-runs <n>` | `10` | Force a fresh Kit boot after this many warm episodes to prevent GPU/stage leaks. |

### 1.4  Display / Rendering

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | off | Launch Isaac without a GUI window. Fastest for CI or unattended runs. |
| `--fast-render` | off | Use `RaytracedLighting` instead of `RealTimePathTracing`. Cuts RTX boot time and per-frame cost at the price of less photorealistic video. |

### 1.5  Network

| Flag | Default | Description |
|------|---------|-------------|
| `--frame-host <ip>` | `127.0.0.1` | IP Isaac pushes camera frames to (Docker Desktop UDP forwarding). |
| `--frame-port <port>` | `55002` | UDP port for camera frames (Isaac → Docker). |
| `--cmd-host <ip>` | `host.docker.internal` | IP Docker pushes velocity commands to (Isaac side). |
| `--cmd-port <port>` | `55001` | UDP port for velocity commands (Docker → Isaac). |

### 1.6  Docker Image & Container

| Flag | Default | Description |
|------|---------|-------------|
| `--image <name>` | `go2-pose-x86:latest` | Docker image name to build/run. |
| `--trt-engine <path>` | `/models/yolo11n-pose-fp16.trt` | TensorRT engine path inside the container. Pre-flight checks this file exists on the host before starting. |
| `--follow-backend <pid\|mppi>` | `pid` | Person-follow controller backend. `pid` = in-process PID; `mppi` = ROS2 sidecar. |

### 1.7  Sim-to-Real & Latency

| Flag | Default | Description |
|------|---------|-------------|
| `--sim2real-validation-cam` | off | Activates the full realism preset (D435 depth noise, proprio noise + 1-step latency, domain randomization, LiDAR range noise). Also sets `--sim-latency-ms 60` and `--sim-latency-jitter-ms 20` unless those are overridden. |
| `--sim-latency-ms <ms>` | `0.0` | Hold each sim frame this many ms before the perception loop sees it (models the real Jetson sense→act delay). |
| `--sim-latency-jitter-ms <ms>` | `0.0` | Uniform ±jitter (ms) on top of `--sim-latency-ms`. |

### 1.8  Self-Test

| Flag | Default | Description |
|------|---------|-------------|
| `--self-test-walk` | off | Inject a constant forward command into the policy and auto-exit. Disables Docker automatically. |
| `--self-test-vx <m/s>` | `0.5` | Forward speed for the self-test walk. |
| `--self-test-sec <s>` | `15.0` | Simulated seconds to walk before auto-exit. |
| `--self-test-no-policy` | off | Skip policy inference during self-test — stand on PD drives only. Isolates physics from the learned policy. |

### 1.9  Parkour Behaviour (forwarded to Isaac)

| Flag | Default | Description |
|------|---------|-------------|
| `--parkour-heading-mode <mode>` | `hybrid` | `vision` = policy self-steers from depth. `command` = always steer toward person bearing. `hybrid` = person-steer on flat, depth self-steer on stairs. |
| `--parkour-mask-fill <fill>` | `terrain` | `terrain` = terrain-inpaint (default, prevents stair-base falls). `far` = legacy flat far-fill (kept for A/B). |
| `--no-parkour-person-mask` | off | Disable the YOLO-bbox depth mask for the parkour policy. Person at close range causes surge. |
| `--no-parkour-walk-mode` | off | Switch from calm walk gait `[0,1]` to agile parkour gait `[1,0]`. |
| `--no-speed-governor` | off | Disable the two-stage speed governor (command backoff + action-norm cap). |
| `--with-o2-payload` | off | Attach the 3D-printed O2 concentrator cradle to the Go2's back. |
| `--stair-square-up` | off | Steer the heading to face the staircase head-on during the approach phase only (experimental). |
| `--final-scene` | off | Compose the upgraded hospital demo scene. |

### 1.10  Logs

| Flag | Default | Description |
|------|---------|-------------|
| `--keep-run-logs <n>` | `1` | Number of past `run_sim_*` log folders to keep. Oldest are pruned on each launch. |
| `--vision-preview` | off | Show the Docker controller's live OpenCV preview window (headless by default in sim). |
| `--pause-after-isaac` | off | Wait for a manual Enter keypress after Isaac signals `world_ready` before starting Docker. |
| `--no-pause-after-isaac` | off | Opposite: skip any pause even if `--pause-after-isaac` was set. |
| `--isaacsim-dir <path>` | `C:\isaac_sim_600` | Path to the Isaac Sim installation (or set `$env:ISAACSIM_DIR`). |

---

## 2  Isaac Sim Flags (`sim/isaac/isaac_env.py`)

These are Python argparse flags passed directly to the Isaac Sim python.bat entrypoint by `sim/run_isaac_window.ps1`. Many are forwarded by the launcher; some are Isaac-only and must be added to the launcher to expose them at the top level.

### 2.1  Display & Rendering

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | off | Run without an Isaac GUI window. |
| `--fast-render` | off | Use `RaytracedLighting` instead of `RealTimePathTracing` for faster iteration. |
| `--no-view-follow-camera` | off | Don't switch the Isaac viewport to the Go2 dynamic follow-camera. |
| `--view-camera-distance <m>` | `3.2` | Follow-camera distance behind the robot. |
| `--view-camera-height <m>` | `1.45` | Follow-camera height above the route. |
| `--view-camera-side-offset <m>` | `-0.85` | Follow-camera lateral offset relative to heading. |

### 2.2  Warm Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--warm-isaac` | off | Keep this Kit process alive across episodes (driven by file sentinel). |
| `--warm-command-file <path>` | `""` | Path to the JSON sentinel `run_sim.ps1` writes to drive episodes. |
| `--warm-max-runs <n>` | `10` | Self-reboot after N warm episodes to prevent GPU/stage memory leaks. |

### 2.3  Network

| Flag | Default | Description |
|------|---------|-------------|
| `--frame-host <ip>` | `0.0.0.0` | Destination IP for camera frame UDP (WSL2 IP when Docker is on WSL). |
| `--frame-port <port>` | `55002` | UDP port for outgoing camera frames. |
| `--cmd-port <port>` | `55001` | UDP port for incoming velocity commands. |

### 2.4  Physics & Simulation Rate

| Flag | Default | Description |
|------|---------|-------------|
| `--physics-hz <hz>` | `200` | Physics simulation rate. 200 Hz gives integer 4× decimation against the 50 Hz RL control rate. |
| `--render-every <n>` | `7` | Render + publish camera frame every N physics steps (~28 fps with default 200 Hz). |
| `--record-every <n>` | `3` | Record top-down + scene_view cameras every N physics steps (~66 fps recording). |
| `--spawn-settle-steps <n>` | `50` | Zero-command hold steps after spawn before `world_ready` is emitted. |

### 2.5  Scene / Environment

| Flag | Default | Description |
|------|---------|-------------|
| `--final-scene` | off | Compose the upgraded hospital demo scene (reuses Go2, patient, cameras, stair pipeline). |
| `--final-scene-env <name>` | `hospital` | Final-scene backdrop. Only `hospital` is currently supported. |
| `--no-hold-motion-until-command` | off | Let autonomous scene motion start _before_ the Docker command stream arrives (default: hold). |
| `--log-dir <path>` | `<repo>/log` | Directory for per-run Isaac JSONL event logs. |
| `--quiet-console-log` | off | Write JSONL only; suppress pretty console log lines. |

### 2.6  Person & Robot Placement

| Flag | Default | Description |
|------|---------|-------------|
| `--person-x <m>` | `-3.5` | Initial X position of the person. Default gives ~15 s of flat following before the stairs at x≈2.0. |
| `--person-y <m>` | `0.0` | Initial Y position of the person. |
| `--go2-x <m>` | `-4.5` | Initial X position of the robot. Default keeps ~1 m separation from the person. |
| `--person-move` | off | Make the person walk a patrol path (always on when launched via `run_sim.ps1`). |

### 2.7  Parkour Locomotion Policy

| Flag | Default | Description |
|------|---------|-------------|
| `--parkour-base-model <path>` | `assets/policies/parkour/base_jit.pt` | TorchScript actor+estimator checkpoint. |
| `--parkour-vision-model <path>` | `assets/policies/parkour/vision_weight.pt` | Depth-encoder state dict. |
| `--parkour-depth-hz <hz>` | `10.0` | Rate (Hz) the rigid depth camera is rendered and submitted to the policy. |
| `--parkour-depth-noise-mult <x>` | `0.0` | RealSense D435 depth-noise multiplier on the ML input (0 = clean; 1.0 = nominal D435). Set automatically by `--sim2real-validation-cam`. |
| `--parkour-heading-mode <mode>` | `hybrid` | Steering mode: `vision`, `command`, or `hybrid` (see launcher §1.9). |
| `--stair-follow-bearing-scale <x>` | `0.4` | Scale factor for follow bearing injected in `hybrid` mode on stairs. |
| `--hold-ramp-sec <s>` | `0.25` | Ramp time (s) to blend policy action to stance pose during a soft hold. |

### 2.8  Person Depth Mask

| Flag | Default | Description |
|------|---------|-------------|
| `--no-parkour-person-mask` | off | Disable YOLO bbox depth masking. Person at close range causes the policy to surge. Leave ON. |
| `--parkour-mask-fill <fill>` | `terrain` | Fill method: `terrain` (terrain-inpaint; prevents stair-base falls) or `far` (legacy flat far-fill for A/B). |

### 2.9  Speed Governor

| Flag | Default | Description |
|------|---------|-------------|
| `--no-speed-governor` | off (governor ON) | Disable both the command-backoff and action-norm cap. Exposes raw policy speed. |
| `--parkour-walk-mode` | on | (explicit) Set policy one-hot to walk `[0,1]` — calm, lower-clearance gait. |
| `--no-parkour-walk-mode` | off | Switch to parkour one-hot `[1,0]` — agile/jumping gait. |
| `--speed-governor-overspeed-ratio <x>` | `1.8` | Command-backoff trigger: backs off if `est_vel > vx_cmd × ratio`. |
| `--speed-governor-action-norm-max <x>` | `8.0` | Action-norm cap. Normal walk ≈ 4–6; surging/jumping > 10. `0` disables. |

### 2.10  Staircase Geometry

| Flag | Default | Description |
|------|---------|-------------|
| `--stair-preset <name>` | `demo_gentle` | Preset: `demo_gentle`, `residential`, `commercial`, `steep`. Single source of truth for tread height, depth, count, and patient path. |
| `--stair-step-height <m>` | (preset) | Override the preset tread rise in metres. |
| `--stair-step-depth <m>` | (preset) | Override the preset tread run in metres. |
| `--stair-step-count <n>` | (preset) | Override the number of steps. |
| `--stair-handrail` | (preset) | Force-add coarse handrail collision volumes. |
| `--no-stair-handrail` | (preset) | Force-disable handrail volumes. |

### 2.11  O2 Payload

| Flag | Default | Description |
|------|---------|-------------|
| `--with-o2-payload` | off | Attach the 3D-printed rail cradle + P2-E6 O2 concentrator to the Go2's back. |

### 2.12  Sim-to-Real Realism Overrides

All off / nominal by default. `--sim2real-validation-cam` turns on the full suite automatically.

| Flag | Default | Description |
|------|---------|-------------|
| `--sim2real-validation-cam` | off | **Preset.** Enables D435 depth noise (`parkour-depth-noise-mult=1.0`), proprio obs noise + 1-step latency, domain randomization + lighting, joint-limit clamp, XT16 LiDAR range noise (0.02 m). |
| `--obs-noise` | off | Inject Gaussian IMU/encoder noise into the proprioceptive observation. |
| `--obs-latency-steps <n>` | `0` | Act on proprio from N control steps ago (models sense→actuate delay). |
| `--torque-rate <Nm/step>` | `0.0` | Actuator torque slew-rate limit (0 = unlimited). |
| `--domain-rand` | off | Enable friction, PD gain, and push-disturbance domain randomization. |
| `--dr-seed <n>` | `0` | Seed for DR random draws (reproducible runs). |
| `--dr-friction-pct <x>` | `0.3` | ±fraction randomization of ground/stair friction (0.3 = ±30%). |
| `--dr-gain-pct <x>` | `0.2` | ±fraction randomization of PD gains kp/kd (0.2 = ±20%). |
| `--dr-push-interval-sec <s>` | `4.0` | Seconds between random push disturbances (≤0 disables). |
| `--dr-push-vel <m/s>` | `0.4` | Magnitude of each random horizontal push disturbance. |
| `--dr-lighting-pct <x>` | `0.0` | ±fraction randomization of scene light intensity (0 = off). |
| `--joint-limit-clamp` | off | Clamp joint-position targets to the articulation's reported hard stops. |
| `--backlash-rad <rad>` | `0.0` | Actuator backlash/deadband half-width on PD position error. |
| `--torque-derate <x>` | `1.0` | Multiplier on commanded torque to model thermal/voltage sag. |

### 2.13  Fall Recovery

| Flag | Default | Description |
|------|---------|-------------|
| `--fall-recovery` | off | On a sustained fall, kinematically re-stand the robot in place and continue (not a learned getup — snaps to stand pose). |
| `--max-fall-recoveries <n>` | `3` | Maximum re-stand attempts before the run ends anyway. |

### 2.14  Self-Test

| Flag | Default | Description |
|------|---------|-------------|
| `--self-test-walk` | off | Inject constant forward velocity into the policy; auto-exit (no controller/Docker needed). |
| `--self-test-vx <m/s>` | `0.5` | Forward speed for the self-test. |
| `--self-test-sec <s>` | `15.0` | Simulated seconds before auto-exit. |
| `--self-test-no-policy` | off | Skip policy inference; hold default pose via PD drives only. |

### 2.15  LiDAR (Simulated Hesai XT16)

| Flag | Default | Description |
|------|---------|-------------|
| `--no-lidar-preview` | off | Disable the XT16 LiDAR raycast + preview video. |
| `--lidar-hz <hz>` | `10.0` | XT16 scan rate (real sensor spins at 10/20 Hz). |
| `--lidar-range-noise-m <m>` | `0.0` | 1-sigma Gaussian range noise per return (0 = exact; real XT16 ≈ 0.02 m). |
| `--lidar-dropout-prob <p>` | `0.0` | Per-ray probability of a missing return. |
| `--lidar-azimuth-step-deg <deg>` | `3.0` | Horizontal angular step between rays. Smaller = denser but more costly (scales as 16 × 360/step). |
| `--lidar-view-range-m <m>` | `6.0` | BEV/range-image colour-scale radius. |
| `--lidar-max-range-m <m>` | `50.0` | Max ray distance before a return is dropped as no-hit. |

### 2.16  ROS2 Bridge

| Flag | Default | Description |
|------|---------|-------------|
| `--ros2-bridge` | off | Emit XT16 point cloud + robot pose over UDP to the `sim_lidar_bridge` ROS2 node. |
| `--ros2-bridge-host <ip>` | `127.0.0.1` | Destination host for the ROS2 bridge UDP sidecar. |
| `--ros2-bridge-port <port>` | `55003` | Destination UDP port for the ROS2 bridge sidecar. |

### 2.17  Video Recording

| Flag | Default | Description |
|------|---------|-------------|
| `--raw-video-path <path>` | `<log-dir>/scene_view.mp4` | MP4 path for the external Isaac scene Left view. |

### 2.18  Camera Debug

| Flag | Default | Description |
|------|---------|-------------|
| `--front-cam-out <path>` | `""` | Save the front D435 RGB to this PNG after `--front-cam-after` steps, then exit. Used to verify the person is visible to YOLO. |
| `--front-cam-after <steps>` | `120` | Steps to run before capturing `--front-cam-out`. |
| `--front-cam-pitch-deg <deg>` | `0.0` | Upward tilt (deg) of the manual fallback camera. |

### 2.19  Verification / Debug

| Flag | Default | Description |
|------|---------|-------------|
| `--verification-image <path>` | `""` | Write a wide scene PNG showing robot, person, and stairs. |
| `--exit-after-verification` | off | Exit after writing `--verification-image`. |

---

## 3  Docker Controller Flags (`core/args_parser.py`)

These are parsed by the Docker container (`sim/main.py` → `core/main.py`). Many are set by `run_sim.ps1` in the `$visionArgs` array; see that section for the effective sim defaults.

### Effective Sim Defaults (as set by `run_sim.ps1`)

The launcher overrides many Python defaults for the sim use-case. If a value isn't listed here, the Python default applies.

| Argument | Sim value | Python default |
|----------|-----------|----------------|
| `--target-distance` | `0.45 m` | `1.5 m` |
| `--trans-x-max` | `0.35 m/s` | `0.6 m/s` |
| `--trans-x-tolerance` | `0.12 m` | `0.1 m` |
| `--trans-x-alpha` | `0.65` | `0.4` |
| `--kp` | `2.0` | `0.9` |
| `--kd` | `0.0` | `0.3` |
| `--follow-pace-distance` | `10.0 m` | `2.0 m` |
| `--sim-latency-ms` | `0.0` (60 if sim2real) | `0.0` |
| `--sim-latency-jitter-ms` | `0.0` (20 if sim2real) | `0.0` |
| `--preview-save-fps` | `5` | `0` |
| `--headless` | on | off |
| `--no-raw-video` | on | off |

### 3.1  Isaac Sim Connection

| Flag | Default | Description |
|------|---------|-------------|
| `--sim` | off | Use Isaac Sim backend instead of real hardware. |
| `--frame-port <port>` | `55002` | UDP port the controller listens on for Isaac camera frames. |
| `--cmd-host <ip>` | `$SIM_CMD_HOST` or `192.168.1.91` | Host where Isaac listens for velocity commands. |
| `--cmd-port <port>` | `55001` | UDP port Isaac listens on for commands. |
| `--sim-frame-timeout-exit-sec <s>` | `30.0` | Exit if no frame arrives for this long (0 = disable). |
| `--sim-latency-ms <ms>` | `0.0` | Artificial frame hold before the perception loop sees it. |
| `--sim-latency-jitter-ms <ms>` | `0.0` | Uniform ±jitter on top of `--sim-latency-ms`. |

### 3.2  Inference / Model

| Flag | Default | Description |
|------|---------|-------------|
| `--trt-engine <path>` | `models/yolo11n-pose-fp16.trt` | TensorRT engine path for YOLO11 pose detection. |
| `--debug` | off | Enable DEBUG-level log messages. |
| `--preprocess-backend <cpu\|gpu>` | `gpu` | Image preprocessing backend before TensorRT inference. |

### 3.3  Camera

| Flag | Default | Description |
|------|---------|-------------|
| `--rotate <0\|90\|180\|270>` | `0` | Clockwise rotation of the input image. |
| `--camera-mode single` | `single` | Camera mode. Only `single` is supported. |
| `--camera-offset-x-m <m>` | `0.0` | Forward offset from camera optical center to base_link. |
| `--camera-offset-y-m <m>` | `0.0` | Lateral offset from camera optical center to base_link. |

### 3.4  Follow Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--follow` | off | Enable person-following mode. |
| `--follow-backend <pid\|mppi>` | `pid` | Follow backend: `pid` (in-process) or `mppi` (ROS2 sidecar). |
| `--network-interface <iface>` | `eth0` | Network interface for real-hardware robot control. |
| `--motion-lock-frames <n>` | `10` | Consecutive matched detections required before motion is allowed. |
| `--no-auto-reacquire` | off (reacquire ON) | Skip automatic person re-selection after the tracked ID is lost. |
| `--tracker-area-weight <x>` | `1.0` | Weight for selecting larger/nearer person boxes. |
| `--tracker-center-weight <x>` | `0.6` | Weight for selecting horizontally centered person boxes. |
| `--follow-start-delay <s>` | `0.0` | Hold all follow commands at zero for this many seconds after first detection. |

### 3.5  MPPI Target Export

| Flag | Default | Description |
|------|---------|-------------|
| `--target-export-host <ip>` | `0.0.0.0` | UDP export host for the MPPI sidecar. |
| `--target-export-port <port>` | `41234` | UDP export port for the MPPI sidecar. |
| `--target-export-rate-hz <hz>` | `15.0` | Export rate cap for the MPPI sidecar. |

### 3.6  Logging & Preview

| Flag | Default | Description |
|------|---------|-------------|
| `--log-components <list>` | `none` | Comma-separated vision ECS log allowlist: `none`, `all`, `vision.main`, `vision.exporter`. |
| `--headless` | off | Disable OpenCV preview windows. |
| `--preview-fps <hz>` | `30.0` | Maximum preview refresh rate. |
| `--preview-save-dir <path>` | `""` | Directory for OpenCV preview output (cleaned at startup). |
| `--preview-save-fps <hz>` | `0.0` | Saved preview frame rate (0 = use `--preview-fps`). |
| `--preview-save-images` | off | Also save individual JPEG frames alongside the video. |
| `--preview-video-path <path>` | `""` | Direct MP4 path for the OpenCV preview (avoids `--preview-save-dir` rmtree). |
| `--ecs-log-dir <path>` | `logs` | Directory for ECS JSONL analytics logs. |
| `--debug-trace-dir <path>` | `""` | Directory for debug-trace JSONL logs (empty = disable). |
| `--debug-trace-every-n-frames <n>` | `1` | Emit debug-trace every N frames. |
| `--rotation-debug` | off | Show a rotation debug visualization window. |
| `--raw-video-path <path>` | `""` | MP4 for the controller-side raw camera frames (no overlays). |
| `--no-raw-video` | off | Disable the controller-side raw video writer (used in sim where Isaac records scene_view). |

### 3.7  PID — X-axis Translation

> Rule: `kp ≥ cruise / (target − tolerance − min_safe_depth)` to guarantee the brake can reach zero before contact.  
> Sim values: cruise=0.35, target=0.45, tolerance=0.12 → `kp ≥ 1.52`; sim uses `kp=2.0`.

| Flag | Default | Sim value | Description |
|------|---------|-----------|-------------|
| `--kp <x>` | `0.9` | `2.0` | Proportional gain on distance error. |
| `--kd <x>` | `0.3` | `0.0` | Derivative gain. |
| `--ki <x>` | `0.0` | `0.0` | Integral gain. |
| `--trans-x-max <m/s>` | `0.6` | `0.35` | Maximum forward command (cruise speed). |
| `--trans-x-tolerance <m>` | `0.1` | `0.12` | Stop band half-width around the target distance. |
| `--trans-x-antiwindup <x>` | `0.0` | — | Integral anti-windup clamp. |
| `--trans-x-alpha <x>` | `0.4` | `0.65` | Low-pass filter on the forward command. |
| `--max-trans-x-accel <m/s²>` | `0.7` | — | Max forward command slew (0 = disable). |
| `--target-distance <m>` | `1.5` | `0.45` | Target following standoff distance. |
| `--follow-pace-distance <m>` | `2.0` | `10.0` | Distance where advance/settle pacing engages (set large in sim to disable pacing). |
| `--follow-pace-speed <m/s>` | `0.4` | — | Max speed during pacing advance phase. |
| `--follow-pace-advance-time <s>` | `2.0` | — | Duration of the pacing advance phase. |
| `--follow-pace-settle-time <s>` | `1.5` | — | Duration of the pacing settle/hold phase. |

### 3.8  PID — Rotation

| Flag | Default | Description |
|------|---------|-------------|
| `--rot-kp <x>` | `0.8` | Proportional gain on yaw error. |
| `--rot-kd <x>` | `0.15` | Derivative gain. |
| `--rot-ki <x>` | `0.0` | Integral gain. |
| `--rot-max <rad/s>` | `1.0` | Maximum yaw command. |
| `--rot-tolerance <deg>` | `3.0` | Yaw dead-band (do not command below this error). |
| `--rot-antiwindup <x>` | `0.0` | Integral anti-windup clamp. |
| `--rot-alpha <x>` | `0.35` | Low-pass filter on the yaw command. |
| `--rot-velocity-ff <x>` | `0.01` | Feed-forward gain from target lateral pixel velocity into yaw command. |
| `--max-rot-accel <rad/s²>` | `1.5` | Max yaw command slew (0 = disable). |

### 3.9  Lost-Target Recovery

| Flag | Default | Description |
|------|---------|-------------|
| `--no-prediction` | off (prediction ON) | Disable short-horizon position prediction after track loss. |
| `--prediction-time-limit <s>` | `3.0` | Max seconds to use the predicted position before giving up. |
| `--min-tracking-time <s>` | `4.0` | Stable tracking seconds required before prediction is trusted. |
| `--lost-search-yaw-speed <rad/s>` | `0.25` | Yaw search speed after the target leaves frame. |
| `--lost-search-timeout-sec <s>` | `2.5` | Max seconds to yaw-search. |
| `--lost-search-min-error-deg <deg>` | `3.0` | Min last-known bearing error before yaw-search is issued. |

### 3.10  Standoff, Gait, and Pacing

| Flag | Default | Description |
|------|---------|-------------|
| `--follow-standoff-speed-gain <x>` | `0.4` | Gain mapping leader speed to standoff distance offset. |
| `--follow-standoff-band-in <m>` | `-0.15` | Hysteresis stop-band offset relative to standoff target. |
| `--follow-standoff-band-out <m>` | `0.15` | Hysteresis start-band offset relative to standoff target. |
| `--no-follow-gait-gate` | off (gate ON) | Disable keypoint/speed gait-based follow gating. |
| `--follow-gait-history-len <n>` | `30` | Rolling history frames for gait estimation. |
| `--follow-gait-walk-threshold <x>` | `0.5` | Walk-classification threshold for the gait estimator. |

### 3.11  Stairs & Obstacle Gating

| Flag | Default | Description |
|------|---------|-------------|
| `--stairs-model <path>` | `yolov8x-worldv2.pt` | YOLO-World model for stair detection. |
| `--stairs-consistency-frames <n>` | `5` | Temporal-consistency window size. |
| `--stairs-consistency-required <n>` | `3` | Positive detections required within the window. |
| `--stairs-latch-frames <n>` | `40` | Frames to keep `stairs_detected` true after a consistent positive. |
| `--stair-near-distance <m>` | `1.2` | Depth threshold where speed/centering are tightened. |
| `--stair-speed-scale <x>` | `0.45` | Forward command scale _while climbing_ (stairs near). |
| `--stair-approach-speed-scale <x>` | `0.5` | Forward command scale during _approach_ (stairs detected but not yet near). |
| `--stair-centering-scale <x>` | `0.6` | Yaw scale while on stairs (suppresses the centering saw). |
| `--stair-forward-floor <m/s>` | `0.35` | Minimum forward command while climbing (prevents policy stall at base). |
| `--stair-rot-max <rad/s>` | `0.6` | Yaw cap while on stairs (lower than `--rot-max`). |
| `--stair-yaw-deadband-deg <deg>` | `4.0` | Zero the yaw while on stairs when centering error is within this band. |
| `--stair-target-distance <m>` | `1.2` | Follow standoff while stairs are detected. |
| `--stair-follow-bearing-scale <x>` | `0.4` | Bearing scale for `hybrid` heading mode on stairs. |
| `--hold-ramp-sec <s>` | `0.25` | Ramp time for blending policy action to stance pose during a soft hold. |

### 3.12  Stair Square-Up (Experimental)

| Flag | Default | Description |
|------|---------|-------------|
| `--stair-square-up` | off | During approach only, steer the parkour heading to face the staircase head-on. Frozen when climb engages. |
| `--stair-square-up-gain <x>` | `0.5` | Heading gain (rad per normalized bbox offset [-1, 1]). |
| `--stair-square-up-max <rad>` | `0.4` | Cap on the square-up heading command. |

### 3.13  Parkour Heading (Controller Side)

| Flag | Default | Description |
|------|---------|-------------|
| `--parkour-yaw-deadband-deg <deg>` | `2.0` | Bearing errors within this band are sent as zero (suppresses bbox jitter). |
| `--parkour-yaw-slew-rad-s <rad/s>` | `3.0` | Max rate of change on the parkour heading command. `0` disables. |

### 3.14  Obstacle Stop

| Flag | Default | Description |
|------|---------|-------------|
| `--no-obstacle-stop` | off (stop ON) | Disable central-depth front-obstacle speed gating. |
| `--obstacle-stop-distance <m>` | `0.55` | Stop forward motion at/below this obstacle depth. |
| `--obstacle-slow-distance <m>` | `1.20` | Begin scaling forward motion below this obstacle depth. |
| `--obstacle-target-clearance <m>` | `0.25` | Only treat as blocking if obstacle is this much closer than the person. |
| `--obstacle-roi-width-ratio <x>` | `0.24` | Central ROI width fraction for obstacle depth sampling. |
| `--obstacle-roi-height-ratio <x>` | `0.42` | Lower-center ROI height fraction for obstacle depth sampling. |

### 3.15  Rotation Error Penalties

| Flag | Default | Description |
|------|---------|-------------|
| `--edge-penalty-k <x>` | `10.0` | Penalty gain for target near the frame edge (increases yaw command). |
| `--size-penalty-k <x>` | `8.0` | Penalty gain for large bounding boxes (person very close). |
| `--large-bbox-thresh <x>` | `0.5` | Fraction of frame height above which the size penalty activates. |

---

## 4  Log Output Structure

Each run writes to `log/run_sim_<timestamp>/`:

```
run_sim_<timestamp>/
├── 00_READ_ME_FIRST.txt      ← human timeline; open this first
├── videos/
│   ├── opencv_preview.mp4    ← YOLO + LiDAR BEV + fused distance (Docker controller)
│   ├── scene_view.mp4        ← Isaac external scene Left view (768×432)
│   ├── topdown.mp4           ← Isaac overhead camera (768×432)
│   └── lidar_preview.mp4     ← Simulated XT16 BEV scan animation
├── reports/
│   ├── evaluation_summary.txt
│   ├── stair_demo_report.json
│   └── verification_*.png
├── logs/
│   ├── launcher.log          ← plain-text launcher timeline
│   ├── status.jsonl          ← machine-readable stage events
│   └── isaac_console.log     ← filtered Isaac important messages
└── debug/                    ← open only when stuck
    ├── isaac_raw.log         ← raw Kit stdout
    ├── isaac_env.jsonl       ← structured Isaac events (fall diagnostics here)
    ├── docker_build.log
    ├── docker_run.log
    ├── ecs/                  ← ECS analytics JSONL
    └── debug_trace/          ← per-frame timing JSONL
```

> **Diagnosis rule:** judge real motion from `debug/isaac_env.jsonl` (fall diagnostics, `patient_reached_destination`), not from `reports/` which can be synthetic.

---

## 5  Changelog

| Date | Change |
|------|--------|
| 2026-06-18 | Initial document created. All flags extracted from `run_sim.bat`, `run_sim.ps1`, `sim/isaac/isaac_env.py`, `core/args_parser.py`. |
