# Isaac Sim — Flag Reference

This guide lists the most common configuration flags for the simulation. For full definitions, inspect the source files:
- **Launcher**: `sim/run_sim.ps1`
- **Isaac Sim Env**: `sim/isaac/isaac_env.py`
- **Docker Controller**: `core/args_parser.py`

---

## 🚀 Common Launch Recipes

Run these commands from the project root using `.\sim\run_sim.bat`:
- **Standard (GUI)**: `.\sim\run_sim.bat` (Starts GUI and Docker controller automatically)
- **Headless & Fast**: `.\sim\run_sim.bat --headless --fast-render` (Fastest for iteration/CI)
- **Warm Boot**: `.\sim\run_sim.bat --headless --warm` (Keeps Isaac alive across runs to bypass boot lag)
- **Warm Shutdown**: `.\sim\run_sim.bat --warm-shutdown` (Cleans up the background warm process)
- **Self-Test Walk**: `.\sim\run_sim.bat --self-test-walk` (Test robot locomotion without Docker controller)
- **Sim-to-Real Validation**: `.\sim\run_sim.bat --sim2real-validation-cam` (Applies camera/LiDAR/IMU noise)

---

## 🧪 Terrain Benchmark (`run_bench`)

Fast, Docker-free locomotion benchmark across a battery of terrains (flat, ramps, stair
presets). Boots Isaac **once** (headless) and drives the frozen parkour policy across every
terrain back-to-back in that warm Kit, recording the same headless videos per terrain and
rolling results into `perf_tracker` plus a `benchmark_summary`. No vision/Docker/TensorRT.

- **Run the whole battery**: `.\sim\run_bench.bat`
- **Subset / reorder**: `.\sim\run_bench.bat --only stairs_steep,ramp_20deg`
- **Full-fidelity render**: `.\sim\run_bench.bat --no-fast-render`
- **Dry run (list battery, no boot)**: `.\sim\run_bench.bat --dry-run`

| Output | Location |
| :--- | :--- |
| Per-terrain run folders (videos/reports/debug) | `log/run_bench_<stamp>/runs/<stamp>_<terrain_id>/` |
| Benchmark summary (PASS/FAIL + articulation) | `log/run_bench_<stamp>/benchmark_summary.{md,csv,json}` |
| Performance table (one row per terrain, `terrain_id` column) | `perf_tracker/data/performance_table.csv` |

Terrain catalogue + drive commands live in `sim/isaac/terrain_bench/terrain_registry.py`
(edit `BATTERY` to add/tune terrains). Per-episode wiring is behind `isaac_env.py --bench`;
the default `run_sim` path is unchanged.

---

## 🔍 Verification & Testing

Verify stage build integrity, joint limits, USD variants, and rendering capabilities using headless simulation tests that capture verification images and exit automatically:

- **Headless Verification Capture (isaac_env)**:
  Runs the environment script in headless mode, builds the world, spawns the actors, settles the scene, captures a viewport screenshot, and exits cleanly:
  ```powershell
  # Using the default ISAACSIM_DIR path:
  & "C:\isaac_sim_600\python.bat" sim/isaac/isaac_env.py --headless --verification-image tests/test_robot_simulation.png --exit-after-verification

  # Using the ISAACSIM_DIR environment variable:
  & "$env:ISAACSIM_DIR\python.bat" sim/isaac/isaac_env.py --headless --verification-image tests/test_robot_simulation.png --exit-after-verification
  ```

- **Standalone Robot Simulation Test Script**:
  A dedicated testing script that verifies USD loading, variant selections, joint limitations, and saves a viewport snapshot:
  ```powershell
  & "$env:ISAACSIM_DIR\python.bat" tests/test_robot_simulation.py
  ```

---

## ⚙️ Core Configuration Flags

### 1. Environment & Rendering (Launcher / Isaac)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--headless` | off | Runs simulation in the background without UI |
| `--fast-render` | off | Uses RaytracedLighting to speed up boot/rendering |
| `--final-scene` | off | Composes the final hospital patient-tracking scene |
| `--with-o2-payload` | off | Attaches the 3D-printed O2 concentrator cradle to the dog |
| `--stair-preset <name>` | `demo_gentle` | Stair geometries: `demo_gentle`, `residential`, `commercial`, `steep` |
| `--self-test-vx <m/s>` | `0.5` | Target forward speed during `--self-test-walk` |

### 2. Follower & Control PID (Docker Controller)
| Flag | Default | Sim Value | Description |
| :--- | :--- | :--- | :--- |
| `--target-distance <m>` | `1.5` | `0.45` | Standoff target distance behind the patient |
| `--trans-x-max <m/s>` | `0.6` | `0.35` | Max cruise speed in person following |
| `--kp <val>` | `0.9` | `2.0` | Proportional gain for translation (distance error) |
| `--rot-max <rad/s>` | `1.0` | `1.0` | Max yaw rotation speed |
| `--rot-kp <val>` | `0.8` | `0.8` | Proportional gain for steering (bearing error) |
| `--follow-backend <pid\|mppi>` | `pid` | `pid` | Follower controller backend |

### 3. Parkour, Locomotion & Safety (Forwarded to Isaac)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--parkour-heading-mode <mode>`| `hybrid` | `vision` (depth self-steer), `command` (steer to person), or `hybrid` |
| `--no-parkour-person-mask` | off | Disable YOLO depth-camera masking (allows close range person surge) |
| `--parkour-mask-fill <fill>` | `terrain` | How to fill masked person depth: `terrain` (inpaint) or `far` (legacy) |
| `--no-parkour-walk-mode` | off | Switch from walking gait `[0,1]` to jumping parkour gait `[1,0]` |
| `--no-speed-governor` | off | Disable speed governor (cap on actions & backoff scaling) |
| `--stair-square-up` | off | Steer heading head-on to the staircase on approach (experimental) |

### 4. Sim-to-Real & Validation
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--sim2real-validation-cam` | off | Enable RealSense depth noise, joint clamps, domain rand, LiDAR noise |
| `--sim-latency-ms <ms>` | `0` | Artificial frame delivery delay to model Jetson processing latency |
| `--domain-rand` | off | Randomize friction, gains, and apply random pushes |
