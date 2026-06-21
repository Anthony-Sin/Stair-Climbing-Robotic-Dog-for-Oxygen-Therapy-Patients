param(
    [Parameter(Mandatory = $true)][string]$IsaacSimDir,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$RunLogDir,
    [string]$RawVideoPath = "",
    [Parameter(Mandatory = $true)][string]$FrameHost,
    [int]$FramePort = 52002,
    [int]$CmdPort = 52001,
    # Low-level controller selection + PGTT curriculum checkpoint (see run_sim.ps1).
    [string]$LocomotionPolicy = "pgtt",
    [string]$PgttLevel = "level17",
    [double]$PgttActionScale = 0.5,
    [double]$PgttHeightscanScale = 1.0,
    [double]$Go2X = -4.5,
    [string]$ParkourHeadingMode = "hybrid",
    [switch]$Sim2RealValidationCam,
    [switch]$SelfTestWalk,
    [double]$SelfTestVx = 0.5,
    [double]$SelfTestSec = 15.0,
    [switch]$SelfTestNoPolicy,
    [switch]$SelfTestHeadingHold,
    [switch]$SelfTestStairs,
    [switch]$NoParkourPersonMask,
    # 'terrain' (restored): the near-fill triggers the policy's climb charge; 'far' killed
    # the climb. run_sim.ps1 passes this explicitly anyway. See [[project_parkour_stair_base_fall_mask]].
    [string]$ParkourMaskFill = "terrain",
    # Staircase geometry. 'residential' (0.178 m riser, US IRC home stairs) is the realistic
    # oxygen-patient scenario AND in-distribution for the frozen Extreme-Parkour policy, which is
    # trained on real obstacle heights and triggers its climb gait on a realistic riser. The old
    # 'demo_gentle' 0.08 m step is OOD-shallow -- the policy reads it as a near-flat ramp,
    # under-reacts, and face-plants at the first riser (confirmed across runs ..190725..200400).
    [string]$StairPreset = "residential",
    # Optional per-step RISE override (m); 0 keeps the preset value. Plumbed to
    # isaac_env --stair-step-height so a run can shorten the risers without a new preset.
    [double]$StairStepHeight = 0,
    [switch]$FinalScene,
    [switch]$NoParkourWalkMode,
    [switch]$NoSpeedGovernor,
    # Attach the oxygen-concentrator payload (mounting rails + O2 tank) on the Go2's back.
    # run_sim.ps1 forwards -WithO2Payload here; without this param it errored the launch.
    [switch]$WithO2Payload,
    [switch]$Headless,
    [switch]$FastRender,
    [switch]$WarmIsaac,
    [string]$WarmCommandFile = "",
    [int]$WarmMaxRuns = 10,
    [switch]$Bench,
    # Dual-policy handoff CLIMB backend: 'parkour' (depth/vision RL, default), 'blind_rl'
    # (proprioceptive rl_sar RL net), or 'ik' (deterministic ClosedLoopStairClimber).
    [string]$HandoffClimbBackend = "parkour",
    # Isolated stair-climb test: drive straight forward up the stairs (no Docker/person-follow)
    # and exit when the robot reaches (StairWaypointX, StairWaypointY) = the top landing.
    [switch]$StairWaypointTest,
    [double]$StairWaypointX = 6.2,
    [double]$StairWaypointY = 0.0
)

$ErrorActionPreference = "Stop"
# Per-run logs are bucketed: verbose raw Kit output under debug/, the filtered
# human console under logs/. (Isaac itself sorts videos/reports/jsonl into buckets.)
$DebugDir = Join-Path $RunLogDir "debug"
$LogsDir = Join-Path $RunLogDir "logs"
New-Item -ItemType Directory -Force -Path $DebugDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$RawLog = Join-Path $DebugDir "isaac_raw.log"
$ConsoleLog = Join-Path $LogsDir "isaac_console.log"
$IsaacEnv = Join-Path $RepoRoot "sim\isaac\isaac_env.py"

function Write-ConsoleLog {
    param([string]$Message)
    Write-Host $Message
    Add-Content -LiteralPath $ConsoleLog -Encoding UTF8 -Value $Message
}

function Should-ShowIsaacLine {
    param([string]$Line)

    if ($Line -match '^\d{2}:\d{2}:\d{2}\s+(INFO|WARNING|ERROR)\s+\[isaac_env\]') {
        return $true
    }
    if ($Line -match '\[Error\]') {
        return $true
    }
    if ($Line -match 'Traceback|Exception|Isaac world is ready|Frame publisher|velocity command receiver') {
        return $true
    }
    return $false
}

Write-ConsoleLog "Isaac Sim launcher"
Write-ConsoleLog "  Raw Kit output:      $RawLog"
Write-ConsoleLog "  Filtered console:    $ConsoleLog"
Write-ConsoleLog "  Isaac JSONL events:  $(Join-Path $DebugDir 'isaac_env.jsonl')"
Write-ConsoleLog "  Isaac Sim dir:       $IsaacSimDir"
Write-ConsoleLog "  Frame target:        ${FrameHost}:${FramePort}"
Write-ConsoleLog "  Command receiver:    0.0.0.0:${CmdPort}"
Write-ConsoleLog "  Locomotion policy:   parkour (depth/vision)"
Write-ConsoleLog "  Parkour heading:     $ParkourHeadingMode"
Write-ConsoleLog "  Parkour person mask: $(-not $NoParkourPersonMask)"
Write-ConsoleLog "  Parkour mask fill:   $ParkourMaskFill"
Write-ConsoleLog "  Final scene:         $([bool]$FinalScene)"
Write-ConsoleLog "  Real-sim env preset: $([bool]$Sim2RealValidationCam)"
Write-ConsoleLog ""

if (-not (Test-Path -LiteralPath $IsaacSimDir)) {
    Write-ConsoleLog "ERROR: Isaac Sim directory does not exist: $IsaacSimDir"
    exit 1
}
$IsaacBat = Join-Path $IsaacSimDir "python.bat"
if (-not (Test-Path -LiteralPath $IsaacBat)) {
    Write-ConsoleLog "ERROR: Missing Isaac Sim launcher: $IsaacBat"
    exit 1
}
if (-not (Test-Path -LiteralPath $IsaacEnv)) {
    Write-ConsoleLog "ERROR: Missing Isaac environment script: $IsaacEnv"
    exit 1
}

Set-Location -LiteralPath $IsaacSimDir
$env:PYTHONUNBUFFERED = "1"

# Initialize RawLog file first
New-Item -ItemType File -Path $RawLog -Force | Out-Null

# PGTT is the default low-level controller; --pgtt-level picks the curriculum checkpoint.
# (--parkour-heading-mode etc. below are parsed but unused on the PGTT path.)
$locomotionArgs = "--locomotion-policy $LocomotionPolicy --pgtt-level $PgttLevel"
$locomotionArgs += " --pgtt-action-scale $PgttActionScale --pgtt-heightscan-scale $PgttHeightscanScale"
$locomotionArgs += " --parkour-heading-mode $ParkourHeadingMode"
# Person-mask is ON by default in isaac_env.py; pass the disable flag through for A/B.
if ($NoParkourPersonMask) { $locomotionArgs += " --no-parkour-person-mask" }
# Terrain-preserving mask fill is the default; pass it through so 'far' is selectable for A/B.
$locomotionArgs += " --parkour-mask-fill $ParkourMaskFill"
# Staircase geometry preset (default residential = realistic + in-distribution for the policy).
$locomotionArgs += " --stair-preset $StairPreset"
# Optional per-step rise override (0 = keep the preset). Lets a run halve/shorten the risers.
if ($StairStepHeight -gt 0) { $locomotionArgs += " --stair-step-height $StairStepHeight" }
if ($FinalScene) { $locomotionArgs += " --final-scene" }
# Walk mode and speed governor are ON by default in isaac_env.py; pass disable flags for A/B.
if ($NoParkourWalkMode) { $locomotionArgs += " --no-parkour-walk-mode" }
if ($NoSpeedGovernor) { $locomotionArgs += " --no-speed-governor" }
# Faster test iteration: run Isaac without the GUI window and/or with the lighter
# RaytracedLighting renderer. Both off by default (live window, full-fidelity render).
if ($Headless) { $locomotionArgs += " --headless" }
if ($FastRender) { $locomotionArgs += " --fast-render" }

# Isaac records the external scene Left view to scene_view.mp4 (beside opencv_preview.mp4).
$rawArg = ""
if ($RawVideoPath) {
    $rawArg = "--raw-video-path `"$RawVideoPath`""
}

# Real-simulated-env preset: turns on the full sim-to-real realism suite inside
# isaac_env.py -- RealSense D435 noise on the depth-camera ML AND the YOLO RGB/depth
# stream, proprio obs noise + 1-step latency, domain randomization + lighting,
# joint-limit clamp, and XT16 LiDAR range noise. Off by default => the clean
# "perfect env".
$validationCamArg = ""
if ($Sim2RealValidationCam) {
    $validationCamArg = "--sim2real-validation-cam"
}

# Controller-free locomotion self-test: drive a constant forward command straight
# into the policy (headless, auto-exits after --self-test-sec) so the gait can be
# isolated from the vision/follow controller. --self-test-no-policy skips inference
# to confirm the robot stands on the position-hold drives alone.
$selfTestArg = ""
if ($SelfTestWalk) {
    $selfTestArg = "--self-test-walk --self-test-vx $SelfTestVx --self-test-sec $SelfTestSec"
    # Gate --headless on the switch (like the normal path at line ~128) so dropping -Headless
    # opens the RTX viewport to WATCH the self-test live, instead of always forcing headless.
    if ($Headless) { $selfTestArg += " --headless" }
    if ($SelfTestNoPolicy) { $selfTestArg += " --self-test-no-policy" }
    if ($SelfTestHeadingHold) { $selfTestArg += " --self-test-heading-hold" }
    if ($SelfTestStairs) { $selfTestArg += " --self-test-stairs" }
    Write-ConsoleLog "  Self-test:           vx=$SelfTestVx, ${SelfTestSec}s, no-policy=$SelfTestNoPolicy, headless=$Headless (no controller)"
}

# Warm-iteration mode: keep this Kit process alive and rebuild the scene per episode
# on command from run_sim.ps1 (the ~120s RTX boot is paid once). Off => one-shot.
$warmArg = ""
if ($WarmIsaac) {
    $warmArg = "--warm-isaac --warm-command-file `"$WarmCommandFile`" --warm-max-runs $WarmMaxRuns"
    Write-ConsoleLog "  Warm mode:           command-file=$WarmCommandFile max-runs=$WarmMaxRuns"
}

# Terrain-benchmark mode: pass --bench so isaac_env reads the per-episode terrain + drive
# from the warm command-file. The person is a pure locomotion-test distractor here, so keep
# it STATIC and well off the robot's forward lane (default -3.5,0 sits directly in the path,
# and a constant-forward drive would collide with it on every terrain).
$benchArg = ""
$personArgs = "--person-move"
if ($Bench) { $benchArg = "--bench" }
if ($Bench -or $SelfTestWalk -or $StairWaypointTest) {
    # Open-loop constant-forward drive (bench / self-test / waypoint-test) would collide
    # with the default person spawn (-3.5,0) sitting in the forward lane, so park the
    # person static and well off-lane. See the CLAUDE.md incident ledger entry. (The
    # waypoint test ALSO parks the person off-lane inside isaac_env, but drop --person-move
    # here so it never patrols into the lane.)
    $personArgs = "--person-x -8.0 --person-y 8.0"
    Write-ConsoleLog "  Person parked off-lane (open-loop bench/self-test/waypoint drive)"
}

# Dual-policy handoff CLIMB backend (parkour | blind_rl | ik). Always pass it so the
# selected backend (e.g. the blind RL climb net) is in effect.
$handoffArg = "--handoff-climb-backend $HandoffClimbBackend"

# Isolated stair WAYPOINT test: Docker-free straight drive up the stairs, exit at the
# target waypoint. Pair with -HandoffClimbBackend blind_rl to test the blind RL climb.
$waypointArg = ""
if ($StairWaypointTest) {
    $waypointArg = "--stair-waypoint-test --stair-waypoint-x $StairWaypointX --stair-waypoint-y $StairWaypointY"
    if ($Headless) { $waypointArg += " --headless" }
    Write-ConsoleLog "  Stair waypoint test: target=($StairWaypointX, $StairWaypointY), climb backend=$HandoffClimbBackend (no controller)"
}

# Oxygen-concentrator payload (mounting rails + O2 tank) on the Go2's back. isaac_env
# attaches it in load_go2() and runs the O2PayloadMonitor (mass/CoM, tank-detach watchdog).
$o2Arg = ""
if ($WithO2Payload) {
    $o2Arg = "--with-o2-payload"
    Write-ConsoleLog "  O2 payload:          ON (oxygen tank + mounting rails attached to the Go2)"
}

$cmdArgs = "/c `"`"$IsaacBat`" `"$IsaacEnv`" $personArgs --go2-x $Go2X --frame-host $FrameHost --frame-port $FramePort --cmd-port $CmdPort --log-dir `"$RunLogDir`" --no-view-follow-camera $rawArg $locomotionArgs $validationCamArg $selfTestArg $warmArg $benchArg $handoffArg $waypointArg $o2Arg > `"$RawLog`" 2>&1`""

# Start the process with direct OS redirection to prevent pipeline blocking
$process = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgs -PassThru -NoNewWindow

# Read and filter the log file in real-time
$reader = New-Object System.IO.StreamReader([System.IO.File]::Open($RawLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite))
try {
    while (-not $process.HasExited) {
        $line = $reader.ReadLine()
        if ($line -ne $null) {
            if (Should-ShowIsaacLine -Line $line) {
                Write-ConsoleLog $line
            }
        } else {
            Start-Sleep -Milliseconds 100
        }
    }
    # Read remaining lines
    while (($line = $reader.ReadLine()) -ne $null) {
        if (Should-ShowIsaacLine -Line $line) {
            Write-ConsoleLog $line
        }
    }
} finally {
    $reader.Close()
}

$exitCode = $process.ExitCode
Write-ConsoleLog ""
Write-ConsoleLog "Isaac process exited with code $exitCode"
exit $exitCode
