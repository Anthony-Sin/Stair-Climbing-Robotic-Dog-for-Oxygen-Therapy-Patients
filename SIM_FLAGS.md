# Isaac Sim — Flag Reference

This guide lists the most common configuration flags for the simulation. For full definitions, inspect the source files:
- **Launcher**: [sim/run_sim.ps1](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim/run_sim.ps1)
- **Isaac CLI args**: [sim/isaac/isaac_args.py](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim/isaac/isaac_args.py)
- **Isaac Sim Env**: [sim/isaac/isaac_env.py](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim/isaac/isaac_env.py)
- **Docker Controller**: [core/args_parser.py](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/args_parser.py)

---

## 🚀 Common Launch Recipes

Run these commands from the project root using `.\sim\run_sim.bat`:
- **Standard (GUI)**: `.\sim\run_sim.bat` (Starts GUI and Docker controller automatically)
- **Headless & Fast**: `.\sim\run_sim.bat --headless --fast-render` (Fastest for iteration/CI)
- **Warm Boot**: `.\sim\run_sim.bat --headless --warm` (Keeps Isaac alive across runs to bypass boot lag)
- **Warm Shutdown**: `.\sim\run_sim.bat --warm-shutdown` (Cleans up the background warm process)
- **Locomotion Self-Test**: `.\sim\run_sim.bat --self-test-walk` (Test robot locomotion without Docker)
- **Stair Waypoint Test**: `.\sim\run_sim.bat --stair-waypoint-test` (Test stair climbing without Docker controller)
- **Sim-to-Real Validation**: `.\sim\run_sim.bat --sim2real-validation-cam` (Applies camera/LiDAR/IMU noise)

---

## 🧪 Terrain Benchmark (`run_bench`)

Fast, Docker-free locomotion benchmark across a battery of terrains (flat, ramps, stair presets). Boots Isaac **once** (headless) and drives the locomotion policy across every terrain back-to-back, recording results in `perf_tracker` plus a `benchmark_summary`.

- **Run the whole battery**: `.\sim\run_bench.bat`
- **Subset / reorder**: `.\sim\run_bench.bat --only stairs_steep,ramp_20deg`
- **Dry run**: `.\sim\run_bench.bat --dry-run` (lists battery without booting)

---

## 🔍 Verification & Testing

Verify stage build integrity, joint limits, USD variants, and rendering capabilities using these test paths:

### 1. Pytest Host-Safe Suite (Offline)
Runs pure-Python offline tests (estimators, tracker, sensors, contracts) that do not require GPU/Isaac:
```bash
pytest tests/
```

### 2. Standalone Robot Simulation Test
A dedicated script verifying USD loading, variant selections, joint limitations, and viewport capture:
```powershell
& "$env:ISAACSIM_DIR\python.bat" tests/test_robot_simulation.py
```

### 3. Headless Verification Capture (`isaac_env.py`)
Spawns actors, settles the scene, captures a viewport screenshot, and exits cleanly:
```powershell
& "$env:ISAACSIM_DIR\python.bat" sim/isaac/isaac_env.py --headless --verification-image tests/test_robot_simulation.png --exit-after-verification
```

---

## ⚙️ Core Configuration Flags

### 1. Environment & Locomotion Policy
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--headless` | off | Runs simulation in the background without UI |
| `--fast-render` | off | Uses RaytracedLighting to speed up boot/rendering |
| `--locomotion-policy <type>` | `pgtt` | Low-level Go2 controller: `pgtt` (default phase-guided) or `parkour` (legacy depth) |
| `--pgtt-level <level>` | `level17` | PGTT curriculum checkpoint: `level03` to `level20` |
| `--handoff-climb-backend <type>`| `blind_rl`| Climb backend: `blind_rl` (default proprioceptive net), `parkour`, or `ik` |
| `--with-o2-payload` | off | Attaches the 3D-printed O2 payload cradle to the dog |
| `--stair-preset <name>` | `demo_gentle`| Stair preset: `demo_gentle`, `residential`, `commercial`, `steep` |
| `--stair-step-height <m>` | `0.178` | Override riser height (residential defaults to 0.178m) |

### 2. Locomotion Self-Test Configuration
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--self-test-walk` | off | Constant forward velocity walk test (Docker bypassed) |
| `--self-test-vx <vx>` | `0.5` | Target speed (m/s) during self-test walk |
| `--self-test-sec <sec>` | `15.0` | Duration (seconds) before self-test auto-exit |
| `--self-test-stairs` | off | Enable stair-climb check on self-test under legacy `parkour` |
| `--self-test-heading-hold` | off | Enable heading-hold to keep the robot walking straight (+X) |

### 3. Follower, Control & Safety (Docker Controller)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--target-distance <m>` | `1.5` | Standoff target distance behind the patient |
| `--trans-x-max <m/s>` | `0.6` | Max cruise speed in person following |
| `--kp <val>` | `0.9` | Proportional gain for translation (distance error) |
| `--rot-max <rad/s>` | `1.0` | Max yaw rotation speed |
| `--rot-kp <val>` | `0.8` | Proportional gain for steering (bearing error) |
| `--follow-backend <pid\|mppi>` | `pid` | Follower controller backend |
| `--no-speed-governor` | off | Disable speed governor (norm cap and vx backoff) |

### 4. Sim-to-Real & Validation
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--sim2real-validation-cam`| off | Enable RealSense depth noise, joint clamps, domain rand, LiDAR noise |
| `--sim-latency-ms <ms>` | `0.0` | Frame delivery delay to model processing latency |
| `--sim-latency-jitter-ms <ms>`| `0.0`| Uniform +/- jitter added to latency per frame |
| `--domain-rand` | off | Randomize friction, gains, and apply random pushes |
