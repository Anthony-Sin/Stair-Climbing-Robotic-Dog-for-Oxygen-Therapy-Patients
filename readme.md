# Stair-Climbing Robotic Dog for Oxygen Therapy Patients

This project enables a quadruped robot (**Unitree Go2**) to serve as an autonomous mobile support companion for oxygen therapy patients. The robot carries a custom 3D-printed payload cradle containing a P2-E6 oxygen concentrator, follows the patient at a safe standoff distance, and autonomously navigates and climbs staircases alongside or behind them.

The system is designed to run in two environments:
1. **Simulation**: Using **NVIDIA Isaac Sim** for high-fidelity physics, vision rendering, and terrain generation.
2. **Real Hardware**: Deployed on a physical **Unitree Go2** robotic dog powered by an onboard **NVIDIA Jetson Orin** system.

---

## 🚀 Key Features

* **Person Tracking**: Integrates YOLOv11/YOLOv8-pose detection with a single-target ByteTrack-based tracker to lock onto and robustly follow the designated patient.
* **Intelligent Person Following**: A custom PID-based tracking controller featuring safety braking zones, low-pass command filtering, and dynamic centering.
* **Stair Climb & Locomotion Policy**: Leverages a learned Parkour Locomotion Policy (walk/jump gaits) to dynamically adapt to flat ground and climb stairs.
* **Perception-Driven Control**: Active depth camera processing with target person bounding-box masking (`terrain-inpaint`) to prevent the robot from misreading the patient's close-range body as high terrain (eliminates surging).
* **Sensor Fusion & Obstacle Gating**: Merges simulated Hesai XT16 LiDAR profiles with depth/camera perception to slow down, square up on stairs, or trigger obstacle stops.
* **Modular Simulation Sandbox**: An Isaac Sim-based environment with configurations for custom stair presets (step height, depth, count), final hospital scene, domain randomization, and sim-to-real noise validation.

---

## 🏗️ System Architecture

### 1. System Overview

The runtime execution is split across two main components communicating over a UDP network interface:

```mermaid
graph TD
    subgraph Launcher [Launcher · run_sim.bat / run_sim.ps1]
        L[Orchestrator\nLogs to log/ · Judge from launcher.log not exit code]
    end

    subgraph Isaac [Isaac Sim · GPU Host · 50 Hz Physics Loop]
        Go2[Unitree Go2 Articulation\n12-DOF · Kp=40 Kd=1.0 · DR per episode]
        Patient[Patient System\nH1 Puppet physics + BipedMannequin visual]
        O2[O2 Payload\nRails + Tank · 600 N strap · CoM monitor]
        Handoff[HandoffController\nstall + riser trigger · policy select · 50 Hz]
        FD[Fall Diagnostics\ndebug/isaac_env.jsonl · authoritative verdict]
        UDPOut[UDP 55002\nRGB 1280x720 + Depth 58x87 out]
        UDPIn[UDP 55001\nvx · wz · yaw_err · hold in]
    end

    subgraph Docker [Docker Controller · Core Process · Per-Frame Perception + Control]
        YOLO[YOLOv8-Pose TRT\nperson bboxes + 17 COCO keypoints]
        YS[YOLO-World Stairs TRT\nopen-vocab · riser count · 3-frame debounce]
        DP[DepthProcessor\nbimodal histogram · LiDAR fusion weight=0.6]
        SPT[SinglePersonTracker\nIoU re-ID · predict up to 3.0 s]
        PF[PersonFollower\n3-zone PID · target 0.45 m · kp=0.8]
        FSM[ClimbFSM + Speed Governor\n6-state · follow_shaping.py · stairs_action_active bypass]
    end

    subgraph Policies [go2_locomotion/ · Shared by Isaac Sim + real/ros2/ ROS2 Port]
        PGTT[PGTT MLP · default\n153-D heightmap]
        Parkour[Parkour CNN\n114-D + depth GRU]
        BlindRL[blind-RL\n45-D proprio · climb backend]
        IKC[IK Climber\n2-link sagittal · deterministic]
    end

    L -->|StairSpec preset forwarded| Isaac
    Go2 -->|Camera Frames| UDPOut
    UDPOut -->|RGB-D Frames| YOLO
    UDPOut -->|RGB| YS
    UDPOut -->|Depth 58x87| DP
    YOLO -->|Bboxes / Keypoints| SPT
    SPT -->|Target Coordinate / Distance / Bearing| PF
    DP -->|Fused Gap m| PF
    YS -->|stairs_detected / riser_count| FSM
    PF -->|Locomotion Commands| FSM
    FSM -->|Final Command Packets| UDPIn
    UDPIn -->|Velocity & Yaw Commands| Go2
    Handoff -->|Policy Select| Policies
    Policies -->|Joint Position Targets 50 Hz| Go2
```

---

### 2. Locomotion Policies

All policies live in `go2_locomotion/` — imported by both Isaac Sim and the real ROS2 port. Do **not** import as `locomotion/` (name collision).

```mermaid
graph TD
    subgraph Root [go2_locomotion/ · Shared Policies · import as go2_locomotion NOT locomotion]
        Root_Note[Kp=40 position drive · shared Isaac Sim + real/ros2/ ROS2 Foxy]
    end

    subgraph PG [PGTT MLP · DEFAULT · pgtt_locomotion_policy.py]
        PG_A[Phase-Guided Terrain Traversal\nHeightmap MLP · no depth camera]
        PG_B[Obs 153-D · Heightscan 11x9=99-D\nPhase trot 0-pi-pi-0 · action scale 0.5]
        PG_C[Kp=40 Kd=0.5 · stall detect vx < 0.06 for 0.6 s\nHandoff: stall + riser count >= 2 at < 1.6 m]
        PG_D[Walk: excellent · Climb: pinned at 0.15 m riser\nsim/models/locomotion/pgtt/]
        PG_A --> PG_B --> PG_C --> PG_D
    end

    subgraph PK [Parkour CNN · parkour_locomotion_policy.py]
        PK_A[Extreme-Parkour-Onboard Go2\nDepth CNN encoder + GRU backbone]
        PK_B[Obs 114-D · 53-D proprio + 58x87 to 32-D depth latent\n10-frame GRU history · action scale 0.25]
        PK_C[Kp=40 Kd=1.0 · heading: vision default or command\nSurge fix: person mask required]
        PK_D[Terrain: perceptive adapts · Person fills depth = surge\nsim/models/locomotion/parkour/]
        PK_A --> PK_B --> PK_C --> PK_D
    end

    subgraph BR [blind-RL · go2_robot_lab_policy.pt]
        BR_A[rl_sar go2_robot_lab policy\nBlind MLP · proprioception only · no depth]
        BR_B[Obs 45-D · proprio only\nKp=40 Kd=1.0 · --handoff-climb-backend blind_rl]
        BR_C[Walk: PGTT handles unchanged\nClimb: general walker · NOT stair-trained]
        BR_D[Do NOT represent as proven stair climber\nsim/models/locomotion/go2_robot_lab_policy.pt]
        BR_A --> BR_B --> BR_C --> BR_D
    end

    subgraph IK [IK Climber · closed_loop_stair_climber.py]
        IK_A[ClosedLoopStairClimber · deterministic\n2-link sagittal IK + PD balance FSM]
        IK_B[L1=L2=0.213 m · 2-bone IK per leg\nFSM: lift to place to settle]
        IK_C[Riser 0.15 m commercial · --closed-loop-stair-climb ON\nDiagnose via trace_climber.py + fall_diag]
        IK_D[Safety: fails safe upright no roll-off\nBlocker: static crawl nose-dives 20-30 deg]
        IK_A --> IK_B --> IK_C --> IK_D
    end

    Root_Note --> PG_A
    Root_Note --> PK_A
    Root_Note --> BR_A
    Root_Note --> IK_A
```

---

### 3. Perception Pipeline

```mermaid
graph TD
    subgraph Camera [Intel D435 RealSense · FRONT_D435_MOUNT at 0.24 0 0.12 m]
        D435[Single co-located RGB + Depth sensor\nRGB + Depth via UDP 55002 to Docker]
    end

    subgraph RGB [RGB Stream · 1280 x 720 · 69H x 42.6V deg]
        YoloPose[YOLOv8-Pose TRT\nbboxes + 17 COCO keypoints · NMS · Jetson TRT engine]
        Tracker[SinglePersonTracker\nIoU re-ID · consistent track_id · lost predict 3.0 s]
        YoloStair[YOLO-World Stairs TRT\nopen-vocab · bbox + confidence + riser_count\nGate: >= 2 risers at < 1.6 m]
        YoloPose -->|Bboxes + Score + Keypoints| Tracker
        Tracker -->|Track ID / Bearing / Distance| YoloStair
    end

    subgraph Depth [Depth Stream · 58 x 87 · float32 · 0 to 2 m]
        DepthProc[DepthProcessor\nperson gap via foreground bimodal histogram\nperson_dist_m + gap_m]
        LiDAR[LiDAR Fusion\nfuse_distance · lidar_weight=0.6\nfused gap_m to PersonFollower]
        PersonMask[Person Depth Mask\nmask_person_in_parkour_depth\nbbox to fill region with terrain depth]
        DepthProc -->|person_dist_m| LiDAR
        LiDAR -->|fused gap_m| PersonMask
    end

    subgraph Control [Control + Policy Outputs]
        PFollow[PersonFollower\n3-zone PID · target 0.45 m · kp=0.8\nzones: far cruise · at-target hold · close brake]
        ClimbFSM[ClimbFSM\n6-state · follow_shaping.py\nstairs_detected triggers STAIR_APPROACH]
        PolicyObs[Policy Obs 114-D\nParkour CNN input vector\n32-D depth latent + 53-D proprio + 3-D cmd]
    end

    subgraph Output [UDP 55001 to Isaac Sim]
        UDP[vx · vy · wz · yaw_err · stairs_detected\ngap_m · person_bbox · hold · stairs_action_active]
    end

    D435 -->|RGB 1280x720| YoloPose
    D435 -->|RGB 1280x720| YoloStair
    D435 -->|Depth 58x87| DepthProc
    PersonMask -->|clean 58x87 depth| PolicyObs
    Tracker -->|Target Bearing + Distance| PFollow
    LiDAR -->|Fused Gap m| PFollow
    YoloStair -->|stairs_detected| ClimbFSM
    PFollow -->|Locomotion Commands| ClimbFSM
    ClimbFSM -->|vx + wz + yaw_err| UDP
    PolicyObs -->|depth latent| UDP
```

---

### 4. Policy Handoff & Climb State Machines

```mermaid
graph TD
    subgraph Handoff [HandoffController · pgtt_stair_handoff.py · Isaac Sim · 50 Hz]
        WALK[WALK\nPGTT or Parkour active\nstall monitor: vx < 0.06 for 0.6 s\nriser gate: >= 2 at < 1.6 m]
        CLIMB[CLIMB\nblind-RL or Parkour or IK backend\nhandoff dist 0.90 m · abort tilt 0.70 rad\ntimeout 8.0 s · stairs_action_active = true]
        TOPEGRESS[TOP EGRESS\nwalk clear of crest\ndepth-clear AND GT-terrain-clear\nwalk top_egress_distance_m forward]
        WALK_R[WALK resumed\nPGTT or Parkour continues\nstall detector resets · can re-trigger]
        ABORT[ABORT to WALK\ntimeout or tilt > 0.70 rad]

        WALK -->|stall + riser conditions met| CLIMB
        CLIMB -->|crest clear debounced 3-frame| TOPEGRESS
        CLIMB -->|timeout or tilt exceeded| ABORT
        ABORT -->|reset| WALK
        TOPEGRESS -->|egress distance walked| WALK_R
        WALK_R -->|stall + riser re-trigger| CLIMB
    end

    subgraph ClimbFSM [ClimbFSM · core/control/follow_shaping.py · Docker · Per-Frame]
        FF[FLAT_FOLLOW\nnormal PID follow · start state\nperson visible and tracked]
        SA[STAIR_APPROACH\nstairs visible\nheading hold + speed hold\ncreep to riser commit]
        SN[STAIR_NEAR\nstair base reached\nsquare-up + vx floor\nobstacle gate bypassed]
        CC[COMMITTED_CLIMB\nactive climb\nstairs_action_active = true\nheading-lock yaw to 0]
        LF[STAIR_LOSS_FLOOR\nstairs lost mid-climb\nhold last bearing + gap\npredict 3.0 s window]
        STOP[STOP\ncmd = 0 all axes\nwait reacquire\nor operator override]

        FF -->|stairs_detected = true| SA
        FF -->|person lost| LF
        SA -->|range close| SN
        SA -->|stairs lost| FF
        SN -->|HandoffController active| CC
        SN -->|stairs gone| LF
        CC -->|stairs gone mid-climb| LF
        CC -->|climb complete| FF
        LF -->|stairs seen again| CC
        LF -->|timeout| STOP
        STOP -->|person seen again| FF
    end

    subgraph FallDiag [Fall Diagnostics · debug/isaac_env.jsonl · Authoritative]
        FDetect[Fall Detection\nflipped: tilt > 1.05 rad\ncollapsed: height < 0.18 m AND tilt > 0.52 rad\nROBOT_COLLAPSE_TILT_RAD=0.52 guards upright-wedge]
        Verdict[Climb Verdict · analyze_climb\nFELL: sustained fall detected\nCOLLIDED: contact without forward progress\nCLEAN CLIMB: sustained upright progress]
        Source[Authoritative Source\ndebug/isaac_env.jsonl per-step JSONL\nDo NOT use evaluation_summary.txt or max_x alone\nperf_tracker archive.jsonl = full history]
        FDetect --> Verdict --> Source
    end
```

---

### 5. Isaac Sim — World Components

```mermaid
graph TD
    subgraph Entry [Entry Point · isaac_env.py · GPU Host Process]
        EP[isaac_env.py\n50 Hz physics loop · do NOT import by name from siblings\nuse logging.getLogger not from isaac_env import LOGGER]
    end

    subgraph Boot [Boot Sequence · ~167 s total]
        B1[SimulationApp\nheadless or windowed · ~123 s Kit/RTX boot]
        B2[configure_stairs\nStairSpec preset · riser geometry\npass --stair-preset or silently uses demo_gentle 0.08 m OOD]
        B3[build_world 50 Hz\nisaacsim.core.api.World · classic numpy backend\ndo NOT call set_backend torch]
        B4[load_go2 + O2 Payload\nURDF Articulation · O2 attached before world.reset\n600 N strap · CoM shift monitor per step]
        B5[spawn_sim_person\nSimPersonTarget · H1Puppet + BipedMannequin\n~22 s per-run · USD cache helps]
        B6[world.reset\nH1Puppet.initialize via POST_PHYSICS callback\nguard is_physics_tensor_entity_valid]
        B7[Physics Loop 50 Hz\nread cmd · HandoffController · policy.step\nO2 monitor · patient patrol · fall watchdog\nSTOP_ISAAC sentinel to graceful shutdown]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
    end

    subgraph Patient [Patient System · biped_anim/]
        H1[H1Puppet physics\ninvisible Unitree H1 · flat-terrain policy\nface-plants any riser · NOT stair-capable]
        Biped[BipedMannequin visual\nUsdSkel Biped_Setup_modified.v2.usd\nLOCAL asset not remote CDN · avoids T-pose]
        ClipPlay[clip_player.py\nbaked mocap walk · flat terrain default]
        FootPlant[foot_planting.py\n2-bone IK stair gait\nstance foot world-locked · no skating]
        LocoCtrl[locomotion_ctrl.py\ndistance-driven phase · no wall-clock skating]
        Rig[rig.py\nchannel map + signs\nauto-measure leg geometry]
        BodyZ[body_z drives discrete tread NOT smooth ramp\nramp-following body = feet float off steps]
        H1 --> Biped
        ClipPlay --> FootPlant
        FootPlant --> LocoCtrl --> Rig
        Rig --> BodyZ
    end

    subgraph Bench [Bench + perf_tracker]
        Sweep[run_stair_sweep.ps1\nwarm boot-once riser sweep\nsweep_present.py to slides]
        TB[terrain_bench/\nflat + ramps + stairs battery\nbench rows off CSV leaderboard]
        Archive[archive.jsonl\nfull run history · source of truth\nrecord_run and rebuild_table public API]
        CSV[performance_table.csv\nlean leaderboard · top actionable runs\nunknown + bench excluded by design]
        Sweep --> Archive
        TB --> Archive
        Archive --> CSV
    end

    subgraph Rules [Import Rules · Critical]
        R1[core/ namespace: use vision NOT perception\nshadows sim/isaac/ on Isaac sys.path]
        R2[go2_locomotion/ NOT locomotion/\nshared sim + real/ros2/]
        R3[Warm IPC command.json\nwrite UTF-8 no-BOM · PS5.1 Set-Content bug]
        R4[Judge runs from log/launcher.log\nnot exit code · capped run exits non-zero]
    end

    EP --> Boot
    Boot --> Patient
    Boot --> Bench
    Boot --> Rules
```

---

## 📂 Project Structure

* **[`core/`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core)**: Contains the core perception, tracking, and control logic.
  * [`main.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/main.py): Main runtime loop of the controller.
  * [`person_follower.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/person_follower.py): PID-based tracking control logic.
  * [`depth_processor.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/depth_processor.py): Depth image filtering and patient body masking.
  * [`lidar_fusion.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/lidar_fusion.py): Merges camera detections and LiDAR distance metrics.
  * [`visualization.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/core/visualization.py): Visualizes system outputs (Bird's Eye View panel, tracking info, LiDAR points).
* **[`sim/`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim)**: Contains Isaac Sim scripts, scene definitions, and launcher files.
  * [`isaac/isaac_env.py`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim/isaac/isaac_env.py): Main Isaac Sim environment script.
  * [`run_sim.bat`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/sim/run_sim.bat) / `run_sim.ps1`: Primary simulation orchestrator scripts.
* **[`real/`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/real)**: Execution scripts and interfaces for physical deployment on the real robot.
* **[`docker/`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/docker)**: Configuration files (Dockerfiles, compose files) for creating the containerized environment.
* **[`ros2_ws/`](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/ros2_ws)**: ROS 2 workspace workspace for communication nodes and sidecar processes.
* **`models/` / `weights/`**: Model checkpoints and pre-compiled TensorRT engines for inference.

---

## 🚦 Getting Started in Simulation

Before running, ensure you have **NVIDIA Isaac Sim** installed and your **Docker Desktop** running.

### Common Launch Recipes

All commands should be executed from the project root or the `sim/` directory:

1. **Standard Launch** (Launches Isaac Sim with GUI + Docker controller container):
   ```powershell
   .\sim\run_sim.bat
   ```

2. **Headless & Fast Rendering** (Runs simulation in the background without graphical windows):
   ```powershell
   .\sim\run_sim.bat --headless --fast-render
   ```

3. **Warm Start Mode** (Keeps Isaac Sim alive between runs to bypass the 120s RTX boot cost):
   ```powershell
   # Boot warm instance
   .\sim\run_sim.bat --headless --fast-render --warm
   
   # Stop warm instance cleanly
   .\sim\run_sim.bat --warm-shutdown
   ```

4. **Self-Test Walk** (Bypasses Docker / controller, directly driving the locomotion policy at a constant speed to test physics):
   ```powershell
   .\sim\run_sim.bat --self-test-walk --self-test-vx 0.5 --self-test-sec 15
   ```

5. **Full Realism Validation** (Sim-to-real preview: applies RealSense depth camera noise, latency, domain randomization, and LiDAR noise):
   ```powershell
   .\sim\run_sim.bat --sim2real-validation-cam
   ```

6. **Multi-Terrain Locomotion Benchmark** (Boots headless Isaac Sim once and sequentially evaluates the policy across flat, ramp, and stair terrains back-to-back, saving output results to `perf_tracker`):
   ```powershell
   # Run the full battery
   cd sim
   .\run_bench.bat

   # Run a subset of terrains
   .\run_bench.bat --only stairs_steep,ramp_20deg

   # Dry-run to list the terrain battery without booting Isaac Sim
   .\run_bench.bat --dry-run
   ```

7. **Verification & Testing (Headless Image Capture)** (Runs headless simulation to verify USD variant loading, joint limits, actor spawning, and camera rendering by saving a snapshot and exiting):
   ```powershell
   # Run the headless verification capture from the environment script:
   & "$env:ISAACSIM_DIR\python.bat" sim/isaac/isaac_env.py --headless --verification-image tests/test_robot_simulation.png --exit-after-verification

   # Or run the standalone robot simulation test script:
   & "$env:ISAACSIM_DIR\python.bat" tests/test_robot_simulation.py
   ```

For the complete flag options, detailed explanation of CLI switches, and logging layout, see the [**`SIM_FLAGS.md`**](file:///c:/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/SIM_FLAGS.md) file.
