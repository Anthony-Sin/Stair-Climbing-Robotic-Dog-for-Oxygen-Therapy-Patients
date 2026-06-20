param(
    [switch]$SkipBuild,
    [switch]$ForceBuild,
    [switch]$DryRun,
    [switch]$NoPauseAfterIsaac,
    [switch]$PauseAfterIsaac,
    [switch]$NoIsaac,
    [switch]$NoDockerRun,
    [switch]$FinalScene,
    [string]$IsaacSimDir = $(if ($env:ISAACSIM_DIR) { $env:ISAACSIM_DIR } else { "C:\isaac_sim_600" }),
    [string]$Image = "go2-pose-x86:latest",
    [string]$FrameHost = "",
    [string]$CmdHost = "",
    [int]$CmdPort = 52001,
    [int]$FramePort = 52002,
    [string]$FollowBackend = "pid",
    [string]$TrtEngine = "/models/yolo11n-pose-fp16.trt",
    [double]$SimFrameTimeoutExitSec = 30.0,
    [switch]$VisionPreview,
    [switch]$NoModelPreflight,
    [switch]$NoIsaacReadyWait,
    [int]$IsaacReadyTimeoutSec = 420,
    [int]$KeepRunLogs = 1,
    [int]$MaxRunTimeSec = 900,
    # Low-level Go2 controller: 'pgtt' (default, phase-guided heightmap stair policy)
    # or 'parkour' (legacy depth/vision policy + climbers) for A/B. PgttLevel selects
    # the curriculum checkpoint (level03..level20; higher = trained on taller stairs).
    [string]$LocomotionPolicy = "pgtt",
    [string]$PgttLevel = "level17",
    # Go2 spawn X (m). Default -4.5 (full follow demo). Set near the stair base
    # (e.g. 1.0; stairs start at x=2.0) for a Docker-free climb self-test that
    # spends the sim-time budget climbing rather than walking to the stairs.
    [double]$Go2X = -4.5,
    [string]$ParkourHeadingMode = "hybrid",
    # 'terrain' (restored): the terrain-inpaint near-fill in the patient region is what
    # TRIGGERS the policy's climb charge at the staircase -- perf history shows terrain
    # mask + governor reached the top landing (max_x 5.61 m), while 'far' opened that
    # region and the dog stopped climbing entirely (beach/face-plant at the first riser,
    # runs 040900-050159). 'far' does kill the flat-ground person-ram surge, but it also
    # kills the climb, so it is the wrong trade. The flat-approach surge is bounded by the
    # speed governor; square-up remains an explicit A/B option. A/B mask with --parkour-mask-fill far.
    # See [[project_parkour_stair_base_fall_mask]].
    [string]$ParkourMaskFill = "terrain",
    [switch]$StairSquareUp,
    # Optional approach square-up. The recorded top-landing configuration kept this OFF and let
    # the person bearing own the approach; pass --stair-square-up only for an explicit A/B run.
    [switch]$NoStairSquareUp,
    [switch]$Sim2RealValidationCam,
    [switch]$SelfTestWalk,
    [double]$SelfTestVx = 0.5,
    [double]$SelfTestSec = 15.0,
    [switch]$SelfTestNoPolicy,
    [switch]$SelfTestHeadingHold,
    [switch]$NoParkourPersonMask,
    [switch]$WithO2Payload,
    [switch]$NoParkourWalkMode,
    [switch]$NoSpeedGovernor,
    [double]$SimLatencyMs = 0.0,
    [double]$SimLatencyJitterMs = 0.0,
    [switch]$Headless,
    [switch]$FastRender,
    [switch]$WarmIsaac,
    [switch]$WarmShutdown,
    [int]$WarmMaxRuns = 10
)

$ErrorActionPreference = "Stop"
if ($ParkourHeadingMode -notin @("vision", "command", "hybrid")) {
    throw "ParkourHeadingMode must be 'vision' (policy self-steers from depth), 'command' (always steer toward the person-follow bearing), or 'hybrid' (person-steer on flat, depth self-steer on the stairs)."
}
if ($ParkourMaskFill -notin @("terrain", "far")) {
    throw "ParkourMaskFill must be 'terrain' (inpaint the masked person with surrounding terrain depth -- default) or 'far' (legacy flat far-fill, for A/B)."
}
if ($SelfTestWalk) {
    # The locomotion self-test drives a constant forward command straight into the
    # parkour policy inside Isaac -- no vision/Docker controller is involved -- so
    # skip Docker and its TensorRT preflight. Isaac auto-exits after -SelfTestSec.
    $NoDockerRun = $true
    $NoModelPreflight = $true
    Write-Host "Self-test mode: driving the policy directly (vx=$SelfTestVx, ${SelfTestSec}s, no-policy=$SelfTestNoPolicy); Docker controller disabled."
}
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$RunLogDir = Join-Path $RepoRoot ("log\run_sim_" + $Stamp)
# Each run folder is bucketed for humans: videos/ (mp4s), reports/ (summaries,
# verification PNGs, JSON), logs/ (launcher + console timeline), debug/ (verbose
# raw/JSONL/ECS/trace -- open only when stuck).
$VideosDir = Join-Path $RunLogDir "videos"
$ReportsDir = Join-Path $RunLogDir "reports"
$LogsDir = Join-Path $RunLogDir "logs"
$DebugDir = Join-Path $RunLogDir "debug"
$LauncherLog = Join-Path $LogsDir "launcher.log"
$StatusLog = Join-Path $LogsDir "status.jsonl"
$SummaryLog = Join-Path $RunLogDir "00_READ_ME_FIRST.txt"
$LatestRunFile = Join-Path (Join-Path $RepoRoot "log") "latest_run.txt"
$DockerContainerName = "go2-pose-sim-" + ($Stamp -replace '[^A-Za-z0-9_.-]', '-')

# --- Warm-iteration mode (-WarmIsaac) ---------------------------------------
# Keep ONE booted Kit process alive across runs and drive episodes via file
# sentinels in log/warm_isaac/, so the ~120s RTX boot is paid once. The default
# one-shot path ignores everything in this block.
$WarmStateDir    = Join-Path (Join-Path $RepoRoot "log") "warm_isaac"
$WarmCommandFile = Join-Path $WarmStateDir "command.json"
$WarmStatusFile  = Join-Path $WarmStateDir "warm_status.json"

function Test-WarmIsaacAlive {
    param([string]$StatusFile)
    if (-not (Test-Path -LiteralPath $StatusFile)) { return $false }
    try {
        $st = Get-Content -LiteralPath $StatusFile -Raw -ErrorAction Stop | ConvertFrom-Json
    } catch { return $false }
    if (-not $st.pid) { return $false }
    if (-not (Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue)) { return $false }
    # Heartbeat freshness (Python time.time() epoch seconds). Stale => hung/dead.
    try {
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
        if (($now - [double]$st.heartbeat_ts) -gt 20.0) { return $false }
    } catch {}
    return $true
}

function Get-NextWarmSeq {
    param([string]$CommandFile)
    $seq = 0
    if (Test-Path -LiteralPath $CommandFile) {
        try { $seq = [int]((Get-Content -LiteralPath $CommandFile -Raw | ConvertFrom-Json).seq) } catch { $seq = 0 }
    }
    return ($seq + 1)
}

function Write-WarmCommand {
    param([string]$CommandFile, [int]$Seq, [string]$Action, [string]$RunDir)
    $dir = Split-Path -Parent $CommandFile
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $obj = [ordered]@{ seq = $Seq; action = $Action; run_dir = $RunDir; stamp = (Get-Date -Format o) }
    $tmp = "$CommandFile.tmp"
    $obj | ConvertTo-Json -Compress | Set-Content -LiteralPath $tmp -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $CommandFile -Force
}

if ($WarmShutdown) {
    if (-not (Test-Path -LiteralPath $WarmStateDir)) { New-Item -ItemType Directory -Force -Path $WarmStateDir | Out-Null }
    $seq = Get-NextWarmSeq -CommandFile $WarmCommandFile
    Write-WarmCommand -CommandFile $WarmCommandFile -Seq $seq -Action "shutdown" -RunDir ""
    Write-Host "Warm Isaac shutdown requested (seq=$seq); the warm Kit process will close."
    exit 0
}

# Clear stale simulation runs (Docker containers and local processes) to release file locks
Write-Host "Cleaning up stale Docker containers and Isaac processes..."
if (-not $DryRun) {
    # Stop and remove any Docker containers labeled com.cable.run_sim=true
    try {
        $staleContainers = & wsl.exe -e docker ps -a --filter "label=com.cable.run_sim=true" --format "{{.Names}}" 2>$null
        if ($LASTEXITCODE -eq 0 -and $staleContainers) {
            foreach ($container in ($staleContainers -split "`n")) {
                $container = $container.Trim()
                if ($container) {
                    Write-Host "Stopping stale Docker container: $container"
                    & wsl.exe -e docker rm -f $container 2>$null | Out-Null
                }
            }
        }
    } catch {}

    # Terminate any running local processes associated with the Isaac environment.
    # In warm mode we deliberately keep the live warm Isaac, so skip this kill.
    if (-not $WarmIsaac) {
        try {
            Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
                $_.CommandLine -and $_.CommandLine.Contains("isaac_env.py")
            } | ForEach-Object {
                try {
                    Write-Host "Stopping stale Isaac process PID: $($_.ProcessId)"
                    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                } catch {}
            }
        } catch {}
    }

    # Terminate any running Omniverse hub.exe processes that might hold file locks
    try {
        Get-Process -Name "hub" -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "Stopping Omniverse hub process PID: $($_.Id)"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}

# Clear the log folder at startup to prevent old runs from clashing. In warm mode the
# folder is preserved (the warm Isaac is mid-session and owns log/warm_isaac/ plus the
# active run folder); old run_sim_* folders are still bounded by Prune-OldRunLogs.
$LogRoot = Join-Path $RepoRoot "log"
if (-not $WarmIsaac -and (Test-Path -LiteralPath $LogRoot)) {
    # Try deleting via WSL to bypass any WSL/Docker mount locks
    try {
        $wslLogRoot = ConvertTo-WslPath -WindowsPath $LogRoot
        & wsl.exe -e rm -rf $wslLogRoot 2>$null
    } catch {}

    # Try deleting via Windows to clean up anything remaining
    try {
        Remove-Item -LiteralPath $LogRoot -Recurse -Force -ErrorAction SilentlyContinue
    } catch {}

    # Double-check: if it still exists, try to empty it
    if (Test-Path -LiteralPath $LogRoot) {
        Get-ChildItem -LiteralPath $LogRoot | ForEach-Object {
            try {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            } catch {}
        }
    }
}
# Ensure the log folder exists
if (-not (Test-Path -LiteralPath $LogRoot)) {
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
}


New-Item -ItemType Directory -Force -Path $RunLogDir | Out-Null
New-Item -ItemType Directory -Force -Path $VideosDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $DebugDir | Out-Null
Set-Content -LiteralPath $LatestRunFile -Encoding UTF8 -Value $RunLogDir
Set-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value @(
    "run_sim log guide",
    "Run folder: $RunLogDir",
    "Started: $(Get-Date -Format o)",
    "",
    "Open this file first. It is the human timeline for this run.",
    "",
    "What the stages mean:",
    "  setup        launcher setup and selected options",
    "  network      WSL and Windows UDP endpoints",
    "  isaac        Isaac Sim PowerShell window launch",
    "  isaac_wait   automatic wait for Isaac world_ready",
    "  operator     optional manual Enter gate before Docker",
    "  models       required TensorRT engine preflight",
    "  build        Docker image build",
    "  docker       robot vision/control container",
    "  logs         old run-folder cleanup",
    "  summary      final launcher result",
    "",
    "What the folders mean:",
    "  videos/    all recorded mp4s (opencv_preview, scene_view, topdown, lidar_preview)",
    "  reports/   evaluation_summary.txt, stair_demo_report.json, verification_*.png",
    "  logs/      human timeline: launcher.log, status.jsonl, isaac_console.log",
    "  debug/     verbose dumps -- open only when stuck:",
    "             isaac_raw.log, isaac_env.jsonl, docker_build.log, docker_run.log, ecs/, debug_trace/",
    "",
    "Key files:",
    "  logs/status.jsonl                           machine-readable launcher stage events",
    "  logs/launcher.log                           plain-text launcher timeline",
    "  logs/isaac_console.log                      filtered important Isaac messages",
    "  videos/opencv_preview.mp4                   OpenCV preview (YOLO + LiDAR BEV + fused distance)",
    "  videos/topdown.mp4 / scene_view.mp4         768x432 overhead + external scene cameras (~66 fps; downscaled to fit the mpeg4 encoder)",
    "",
    "Fast diagnosis:",
    "  If isaac_wait is complete, Isaac emitted world_ready and scene loading finished.",
    "  Docker starts automatically after Isaac is ready; pass --pause-after-isaac to restore the manual gate.",
    "  Isaac now holds autonomous person/distractor motion until the controller sends a nonzero command.",
    "  Docker will not command the robot until both TensorRT engine files exist in models/.",
    "  OpenCV preview video is saved even when GUI preview is disabled.",
    "  If build fails, docker_run.log will not exist because the controller never started.",
    "  Existing Docker images are reused automatically.",
    "  To force a rebuild, run: .\run_sim.bat --force-build",
    "  To skip image checks/builds entirely, run: .\run_sim.bat --skip-build",
    "  By default, old run_sim_* folders are pruned and only the latest run is kept.",
    "  To keep more history, run: .\run_sim.bat --keep-run-logs 5",
    "",
    "Timeline:"
)

function ConvertTo-WslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert Windows path to WSL path: $resolved"
    }
    $drive = $matches[1].ToLowerInvariant()
    $rest = $matches[2].Replace('\', '/')
    return "/mnt/$drive/$rest"
}

function Format-CommandLine {
    param([string]$FilePath, [string[]]$Arguments)

    $parts = @($FilePath) + $Arguments
    return ($parts | ForEach-Object {
        if ($_ -match '\s') {
            '"' + $_.Replace('"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

function Write-Stage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$State,
        [Parameter(Mandatory = $true)][string]$Message,
        [hashtable]$Data = @{}
    )

    $now = Get-Date
    $row = [ordered]@{
        timestamp = $now.ToString("o")
        stage = $Stage
        state = $State
        message = $Message
    }
    foreach ($key in $Data.Keys) {
        $row[$key] = $Data[$key]
    }
    $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $StatusLog -Encoding UTF8

    $line = "[{0:HH:mm:ss}] {1,-12} {2,-9} {3}" -f $now, $Stage, $State, $Message
    Write-Host $line
    Add-Content -LiteralPath $LauncherLog -Encoding UTF8 -Value $line

    $summaryPairs = @()
    foreach ($key in @(
        "run_log_dir",
        "summary_log",
        "latest_run_file",
        "frame_host",
        "frame_port",
        "cmd_host",
        "cmd_port",
        "raw_log",
        "console_log",
        "event_log",
        "log",
        "exit_code",
        "pid",
        "reason",
        "deleted_count",
        "keep_count",
        "missing_models",
        "model_path",
        "timeout_sec",
        "command",
        "container",
        "image"
    )) {
        if ($Data.ContainsKey($key)) {
            $summaryPairs += ("{0}={1}" -f $key, $Data[$key])
        }
    }
    $summaryLine = $line
    if ($summaryPairs.Count -gt 0) {
        $summaryLine = $summaryLine + " | " + ($summaryPairs -join "; ")
    }
    Add-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value $summaryLine
}

function ConvertTo-CleanText {
    param([object]$Value)

    return (($Value -join "`n") -replace "`0", "").Trim()
}

function Resolve-HostRuntimePath {
    param([Parameter(Mandatory = $true)][string]$RuntimePath)

    $normalized = $RuntimePath.Replace('/', '\')
    if ($normalized -match '^\\workspace\\(.+)$') {
        return Join-Path $RepoRoot $matches[1]
    }
    if ($normalized -match '^\\models\\(.+)$') {
        return Join-Path (Join-Path $RepoRoot "models") $matches[1]
    }
    if ([System.IO.Path]::IsPathRooted($normalized)) {
        return $normalized
    }
    return Join-Path $RepoRoot $normalized
}

function Test-ModelPreflight {
    if ($DryRun) {
        Write-Stage "models" "dry-run" "Model preflight skipped because --dry-run is set"
        return $true
    }
    if ($NoDockerRun) {
        Write-Stage "models" "skipped" "Model preflight skipped because Docker run is disabled"
        return $true
    }
    if ($NoModelPreflight) {
        Write-Stage "models" "skipped" "Model preflight skipped by --no-model-preflight"
        return $true
    }

    $checks = @(
        @{ name = "YOLO pose TensorRT engine"; runtime = $TrtEngine }
    )
    $missing = @()
    foreach ($check in $checks) {
        $hostPath = Resolve-HostRuntimePath -RuntimePath $check.runtime
        if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf)) {
            $missing += ("{0}: {1}" -f $check.name, $hostPath)
        }
    }

    if ($missing.Count -eq 0) {
        Write-Stage "models" "ready" "Required TensorRT engine files are present" @{
            model_path = (Join-Path $RepoRoot "models")
        }
        return $true
    }

    Write-Stage "models" "failed" "Required TensorRT engine files are missing; Docker controller would exit before sending robot commands" @{
        missing_models = ($missing -join " | ")
        model_path = (Join-Path $RepoRoot "models")
        command = "Build the engines from readme.md Part 8, then rerun .\run_sim.bat"
    }
    Write-Host ""
    Write-Host "Required model engines are missing:"
    foreach ($item in $missing) {
        Write-Host "  $item"
    }
    Write-Host ""
    Write-Host "Build these TensorRT engines on this GPU using readme.md Part 8, then rerun .\run_sim.bat."
    Write-Host "Rebuilding Docker alone will not create files in the mounted models folder."
    return $false
}

function Write-DockerFailureDiagnosis {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }
    $text = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
    if (-not $text) {
        return $false
    }

    if ($text -match "FileNotFoundError:.*models[\\/][^'`"]+\.trt" -or $text -match "TensorRT engine not found") {
        Write-Stage "docker_diag" "failed" "Docker exited before control because a TensorRT engine file is missing" @{
            log = $LogPath
            command = "Build the engines from readme.md Part 8, then rerun .\run_sim.bat"
        }
        return $true
    }
    if ($text -match "ModuleNotFoundError: No module named '([^']+)'") {
        Write-Stage "docker_diag" "failed" "Docker image is missing a Python module" @{
            log = $LogPath
            command = ".\run_sim.bat --force-build"
        }
        return $true
    }
    if ($text -match "libcuda\.so\.1") {
        Write-Stage "docker_diag" "failed" "Docker could not access the NVIDIA driver/CUDA runtime" @{
            log = $LogPath
            command = "Check Docker Desktop GPU/WSL integration, then rerun .\run_sim.bat"
        }
        return $true
    }
    if ($text -match "Bind for 0\.0\.0\.0:(\d+) failed: port is already allocated") {
        $blockedPort = $matches[1]
        Write-Stage "docker_diag" "failed" "Docker could not bind the sim frame UDP port because another container or process already owns it" @{
            log = $LogPath
            command = "Stop the stale process using UDP $blockedPort, then rerun .\run_sim.bat"
        }
        return $true
    }
    if ($text -match "Sim camera did not receive Isaac frames") {
        Write-Stage "docker_diag" "failed" "Docker started but did not receive Isaac camera frames" @{
            log = $LogPath
            command = "Check the frame_host/frame_port route; this launcher defaults Isaac frame_host to 127.0.0.1 for Docker Desktop UDP forwarding"
        }
        return $true
    }
    if ($text -match "ReID reacquire timeout triggered safe exit") {
        Write-Stage "docker_diag" "failed" "Docker exited from the old ReID reacquire timeout path before the stair demo finished" @{
            log = $LogPath
            command = "Rerun with the current ReID-free controller path; the launcher no longer passes an OSNet engine"
        }
        return $true
    }
    if ($text -match "(?m)^\./src/main\.py\s*$" -and $text -notmatch "\[main\] Sim mode") {
        Write-Stage "docker_diag" "failed" "Docker listed the mounted workspace and exited before main.py started" @{
            log = $LogPath
            command = "The launcher now passes Docker arguments directly through wsl.exe; rerun .\run_sim.bat"
        }
        return $true
    }
    return $false
}

function Get-IsaacLastAction {
    param([Parameter(Mandatory = $true)][string]$EventLog)

    if (-not (Test-Path -LiteralPath $EventLog)) {
        return "event_log_missing"
    }
    try {
        $matchInfo = Select-String -LiteralPath $EventLog -Pattern '"action":"([^"]+)"' | Select-Object -Last 1
        if ($matchInfo -and $matchInfo.Line -match '"action":"([^"]+)"') {
            return $matches[1]
        }
        return "not_found"
    } catch {
        return "event_log_unreadable"
    }
}

function Test-PatientReachedTop {
    param([Parameter(Mandatory = $true)][string]$EventLog)

    if (-not (Test-Path -LiteralPath $EventLog)) {
        return $false
    }
    try {
        $matches = Select-String -LiteralPath $EventLog -Pattern '"action":"patient_reached_destination"' -Quiet
        return [bool]$matches
    } catch {
        return $false
    }
}

function Write-SimCompletionGateDiagnosis {
    param(
        [Parameter(Mandatory = $true)][string]$EventLog,
        [Parameter(Mandatory = $true)][string]$DockerLog
    )

    if (Test-PatientReachedTop -EventLog $EventLog) {
        return $false
    }

    $lastAction = Get-IsaacLastAction -EventLog $EventLog
    $reason = "controller_exit_before_patient_top"
    if (Test-Path -LiteralPath $DockerLog) {
        $dockerText = Get-Content -LiteralPath $DockerLog -Raw -ErrorAction SilentlyContinue
        if ($dockerText -match "ReID reacquire timeout triggered safe exit") {
            $reason = "reid_timeout_before_patient_top"
        } elseif ($dockerText -match "Sim camera did not receive Isaac frames") {
            $reason = "sim_frame_timeout_before_patient_top"
        }
    }

    Write-Stage "sim_gate" "failed" "Controller stopped before the patient reached the top of the stairs" @{
        reason = $reason
        last_isaac_action = $lastAction
        event_log = $EventLog
        docker_log = $DockerLog
    }
    return $true
}

function Prune-OldRunLogs {
    param([int]$KeepCount = 1)

    $keep = [Math]::Max(1, $KeepCount)
    $logRoot = Join-Path $RepoRoot "log"
    if (-not (Test-Path -LiteralPath $logRoot)) {
        return
    }

    $runDirs = Get-ChildItem -LiteralPath $logRoot -Directory -Filter "run_sim_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending
    $toDelete = @($runDirs | Select-Object -Skip $keep)
    if ($toDelete.Count -eq 0) {
        Write-Stage "logs" "ready" "No old run log folders to prune" @{ keep_count = $keep }
        return
    }

    $deleted = 0
    foreach ($dir in $toDelete) {
        try {
            if ($dir.FullName -ne $RunLogDir) {
                Remove-Item -LiteralPath $dir.FullName -Recurse -Force
                $deleted += 1
            }
        } catch {
            Write-Stage "logs" "warning" "Could not delete an old run log folder" @{
                log = $dir.FullName
                error = ConvertTo-CleanText $_
            }
        }
    }

    Write-Stage "logs" "pruned" "Old run log folders pruned" @{
        deleted_count = $deleted
        keep_count = $keep
    }
}

function Test-ConsoleLineQuiet {
    param([string]$Line)

    $Line = $Line.Trim()
    if (-not $Line) { return $true }

    # Convert to lowercase for easier matching
    $lower = $Line.ToLower()

    # Filter CUDA banner / NGC info
    if ($Line -eq "===========" -or $Line -eq "== CUDA ==") { return $true }
    if ($lower -like "*cuda version*") { return $true }
    if ($lower -like "*nvidia corporation*") { return $true }
    if ($lower -like "*governed by the nvidia deep learning*") { return $true }
    if ($lower -like "*deep learning container license*") { return $true }
    if ($lower -like "*by pulling and using the container*") { return $true }
    if ($lower -like "*ngc-dl-container-license*") { return $true }

    # Filter CMake / Make compile / install noise
    if ($lower -like "*installing:*") { return $true }
    if ($lower -like "*up-to-date:*") { return $true }
    if ($lower -like "*built target*") { return $true }
    if ($lower -like "*building c object*") { return $true }
    if ($lower -like "*linking c shared library*") { return $true }
    if ($lower -like "*linking c static library*") { return $true }
    if ($lower -like "*linking c executable*") { return $true }
    if ($lower -like "*building cxx object*") { return $true }
    if ($lower -like "*linking cxx shared library*") { return $true }
    if ($lower -like "*linking cxx static library*") { return $true }
    if ($lower -like "*linking cxx executable*") { return $true }

    # Filter pip / progress bars / ultralytics package downloads / logs
    if ($Line -match '^[0-9]+%?\s+[━╸─]+') { return $true }
    if ($lower -like "*downloading*") { return $true }
    if ($lower -like "*collecting*") { return $true }
    if ($lower -like "*cloning*") { return $true }
    if ($lower -like "*resolved*") { return $true }
    if ($lower -like "*installing build dependencies*") { return $true }
    if ($lower -like "*getting requirements to build wheel*") { return $true }
    if ($lower -like "*preparing metadata*") { return $true }
    if ($lower -like "*building wheels for*") { return $true }
    if ($lower -like "*successfully built*") { return $true }
    if ($lower -like "*installing collected packages*") { return $true }
    if ($lower -like "*successfully installed*") { return $true }
    if ($lower -like "*running pip as the 'root' user*") { return $true }
    if ($lower -like "*requirements: ultralytics requirement*") { return $true }
    if ($lower -like "*requirements: autoupdate success*") { return $true }
    if ($lower -like "*restart runtime or rerun command*") { return $true }
    if ($lower -like "*pip3 install*") { return $true }
    if ($lower -like "*pip install*") { return $true }
    if ($lower -like "*obtaining file:*") { return $true }
    if ($lower -like "*running setup.py develop*") { return $true }
    if ($lower -like "*checking if build backend*") { return $true }

    # Filter lines that are progress indicators or download speeds
    if ($Line -match '[━╸─]+\s+\d+\.\d+/[0-9.]+\s+[KMG]B') { return $true }
    # Filter general progress lines in docker buildkit (e.g. #11 DONE 51.6s)
    if ($Line -match '^#\d+\s+DONE\s+[0-9.]+s') { return $true }
    # Filter progress bars in download/buildkit output
    if ($Line -match '━+') { return $true }

    # Keep important step headers, e.g. #12 [ 9/13] RUN ...
    # but discard general buildkit verbose stdout/stderr lines that start with #<num> followed by float/done
    if ($Line -match '^#\d+\s+\d+\.\d+\s+') { return $true }

    return $false
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $commandLine = Format-CommandLine -FilePath $FilePath -Arguments $Arguments
    Write-Stage $Stage "start" "Running command" @{ command = $commandLine; log = $LogPath }

    if ($DryRun) {
        Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value "[dry-run] $commandLine"
        Write-Stage $Stage "dry-run" "Command skipped because --dry-run is set"
        return 0
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $previousNativeErrorPreference = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
        $previousNativeErrorPreference = $global:PSNativeCommandUseErrorActionPreference
    }
    try {
        $ErrorActionPreference = "Continue"
        if ($null -ne $previousNativeErrorPreference) {
            $global:PSNativeCommandUseErrorActionPreference = $false
        }

        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $line = ConvertTo-CleanText $_
            if ($line) {
                Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value $line
                if (-not (Test-ConsoleLineQuiet -Line $line)) {
                    Write-Host $line
                }
            }
        }
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    } catch {
        $exitCode = 1
        $message = ConvertTo-CleanText $_
        Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value $message
        Write-Stage $Stage "failed" "Command launcher failed before exit code was available" @{
            exit_code = $exitCode
            log = $LogPath
            error = $message
        }
        return $exitCode
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($null -ne $previousNativeErrorPreference) {
            $global:PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
    }

    if ($exitCode -eq 0) {
        Write-Stage $Stage "complete" "Command completed" @{ exit_code = $exitCode; log = $LogPath }
    } else {
        Write-Stage $Stage "failed" "Command failed" @{ exit_code = $exitCode; log = $LogPath }
    }
    return $exitCode
}

function Test-DockerImageExists {
    param([Parameter(Mandatory = $true)][string]$ImageName)

    if ($DryRun) {
        return $false
    }

    $raw = & wsl.exe -e docker image inspect $ImageName --format "{{.Id}}" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Stage "build" "ready" "Docker image already exists; build will be skipped" @{ image = $ImageName }
        return $true
    }

    Write-Stage "build" "notice" "Docker image was not found locally; build is required" @{ image = $ImageName }
    return $false
}

function Stop-DockerContainer {
    param(
        [Parameter(Mandatory = $true)][string]$ContainerName,
        [string]$Reason = "launcher cleanup"
    )

    if ($DryRun -or -not $ContainerName) {
        return $false
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $previousNativeErrorPreference = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
        $previousNativeErrorPreference = $global:PSNativeCommandUseErrorActionPreference
    }
    try {
        $ErrorActionPreference = "Continue"
        if ($null -ne $previousNativeErrorPreference) {
            $global:PSNativeCommandUseErrorActionPreference = $false
        }
        # Send SIGTERM first (docker stop gives Python time to flush VideoWriter),
        # then hard-remove. The --time flag sets seconds to wait before SIGKILL.
        & wsl.exe -e bash -c "docker stop --time 8 '$ContainerName' 2>/dev/null; docker rm -f '$ContainerName' 2>/dev/null" 2>&1
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    } catch {
        Write-Stage "docker" "warning" "Could not stop Docker container" @{
            container = $ContainerName
            reason = $Reason
            error = ConvertTo-CleanText $_
        }
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($null -ne $previousNativeErrorPreference) {
            $global:PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
    }

    if ($exitCode -eq 0) {
        Write-Stage "docker" "cleanup" "Stopped Docker container" @{
            container = $ContainerName
            reason = $Reason
        }
        return $true
    }
    return $false
}

function Stop-StaleSimContainers {
    param(
        [Parameter(Mandatory = $true)][string]$ImageName,
        [Parameter(Mandatory = $true)][int]$PublishedUdpPort,
        [string]$ExpectedContainerName = ""
    )

    if ($DryRun -or -not $ImageName -or $PublishedUdpPort -le 0) {
        return
    }

    $filterImage = "ancestor=$ImageName"
    $filterPort = "publish=$PublishedUdpPort/udp"
    $raw = & wsl.exe -e docker ps --filter $filterImage --filter $filterPort --format "{{.Names}}" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Stage "docker" "warning" "Could not inspect stale sim containers before launch" @{
            error = ConvertTo-CleanText $raw
        }
        return
    }

    $containers = @(
        $raw |
            ForEach-Object { ConvertTo-CleanText $_ } |
            Where-Object { $_ -and $_ -ne $ExpectedContainerName }
    )
    if ($containers.Count -eq 0) {
        return
    }

    foreach ($container in $containers) {
        $null = Stop-DockerContainer `
            -ContainerName $container `
            -Reason "stale sim controller publishing UDP $PublishedUdpPort"
    }
}

function Start-IsaacExitMonitorJob {
    param(
        $Process = $null,
        [Parameter(Mandatory = $true)][string]$ContainerName,
        [int]$TimeoutSec = 0
    )

    if ($DryRun -or -not $Process -or -not $ContainerName) {
        return $null
    }

    try {
        $watchedPid = [int]$Process.Id
    } catch {
        return $null
    }

    return Start-Job -ScriptBlock {
        param([int]$WatchedPid, [string]$ContainerName, [int]$TimeoutSec)

        $elapsed = 0
        while ($true) {
            Start-Sleep -Seconds 1
            $elapsed++
            if ($TimeoutSec -gt 0 -and $elapsed -ge $TimeoutSec) {
                # Send SIGTERM first so Python can flush VideoWriter before SIGKILL
                & wsl.exe -e bash -c "docker stop --time 8 '$ContainerName' 2>/dev/null; docker rm -f '$ContainerName' 2>/dev/null" 2>$null | Out-Null
                break
            }
            $watched = Get-Process -Id $WatchedPid -ErrorAction SilentlyContinue
            if (-not $watched) {
                # Send SIGTERM first so Python can flush VideoWriter before SIGKILL
                & wsl.exe -e bash -c "docker stop --time 8 '$ContainerName' 2>/dev/null; docker rm -f '$ContainerName' 2>/dev/null" 2>$null | Out-Null
                break
            }
        }
    } -ArgumentList $watchedPid, $ContainerName, $TimeoutSec
}

function Stop-IsaacProcess {
    param(
        $Process = $null,
        [string]$Reason = "launcher cleanup"
    )

    if ($DryRun) {
        return $false
    }

    $stopped = $false

    try {
        if ($Process -and -not $Process.HasExited) {
            $pidText = [string][int]$Process.Id
            & taskkill.exe /PID $pidText /T /F 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Stage "isaac" "cleanup" "Stopped Isaac launcher process tree" @{
                    pid = [int]$Process.Id
                    reason = $Reason
                }
                $stopped = $true
            }
        }
    } catch {
        $pidValue = if ($Process) { [int]$Process.Id } else { 0 }
        Write-Stage "isaac" "warning" "Could not stop Isaac launcher process tree" @{
            pid = $pidValue
            error = ConvertTo-CleanText $_
        }
    }

    try {
        $kitProcesses = @(
            Get-CimInstance Win32_Process -Filter "Name = 'kit.exe'" -ErrorAction Stop |
                Where-Object {
                    $_.CommandLine -and
                    $_.CommandLine.Contains("isaac_env.py") -and
                    $_.CommandLine.Contains($RunLogDir)
                }
        )
        foreach ($kitProcess in $kitProcesses) {
            $kitPid = [string][int]$kitProcess.ProcessId
            & taskkill.exe /PID $kitPid /T /F 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Stage "isaac" "cleanup" "Stopped Isaac Kit process" @{
                    pid = [int]$kitProcess.ProcessId
                    reason = $Reason
                }
                $stopped = $true
            }
        }
    } catch {
        Write-Stage "isaac" "warning" "Could not inspect Isaac Kit processes for cleanup" @{
            reason = $Reason
            error = ConvertTo-CleanText $_
        }
    }

    return $stopped
}

function Invoke-PerfTracker {
    $perfScript = Join-Path $RepoRoot "perf_tracker\update_table.py"
    if ($DryRun -or -not (Test-Path -LiteralPath $perfScript)) {
        return
    }
    try {
        $gitBranch = ""
        try { $gitBranch = ((& git rev-parse --abbrev-ref HEAD 2>&1) -join "").Trim() } catch {}
        $wslPerfScript = "$WslRepoRoot/perf_tracker/update_table.py"
        & wsl.exe -e python3 $wslPerfScript $WslRunLogDir --git-branch $gitBranch 2>&1 |
            ForEach-Object { Write-Host $_ }
    } catch {
        Write-Host "[perf_table] Warning: performance table update skipped: $_"
    }
}

function Write-IsaacFailureDiagnosis {
    param(
        [Parameter(Mandatory = $true)][string]$EventLogPath,
        [string]$RawLogPath = ""
    )

    $eventText = ""
    if (Test-Path -LiteralPath $EventLogPath) {
        $eventText = Get-Content -LiteralPath $EventLogPath -Raw -ErrorAction SilentlyContinue
    }
    $rawText = ""
    if ($RawLogPath -and (Test-Path -LiteralPath $RawLogPath)) {
        $rawText = Get-Content -LiteralPath $RawLogPath -Raw -ErrorAction SilentlyContinue
    }

    if ($eventText -match "person_asset_missing") {
        Write-Stage "isaac_diag" "failed" "Isaac could not find a real Isaac People character asset; install/configure the matching Isaac Sim Assets pack" @{
            event_log = $EventLogPath
            raw_log = $RawLogPath
        }
        return $true
    }
    if (
        $rawText -match "Failed to prepare local modified Biped_Setup copy" -or
        ($rawText -match "Biped_Setup_modified" -and $rawText -match "Access is denied")
    ) {
        Write-Stage "isaac_diag" "failed" "Isaac could not write the generated Biped_Setup person asset; a previous Isaac process may still be holding the file" @{
            event_log = $EventLogPath
            raw_log = $RawLogPath
        }
        return $true
    }
    if ($eventText -match "person_animation_not_ready" -or $eventText -match "registered_character_count=0") {
        Write-Stage "isaac_diag" "failed" "Person AnimGraph did not register; refusing to run a sliding/fallback person animation" @{
            event_log = $EventLogPath
            raw_log = $RawLogPath
        }
        if ($rawText -match "Python import process in omni\.anim\.graph\.core failed") {
            Write-Stage "isaac_diag" "failed" "Bundled omni.anim.graph.core Python node registration failed during Isaac startup" @{
                raw_log = $RawLogPath
            }
        }
        return $true
    }
    if ($rawText -match "Python import process in omni\.anim\.graph\.core failed") {
        Write-Stage "isaac_diag" "failed" "Bundled omni.anim.graph.core Python node registration failed during Isaac startup" @{
            raw_log = $RawLogPath
        }
        return $true
    }
    return $false
}

function Wait-IsaacReady {
    param(
        [Parameter(Mandatory = $true)][string]$EventLogPath,
        $Process = $null,
        [int]$TimeoutSec = 420
    )

    if ($DryRun) {
        Write-Stage "isaac_wait" "dry-run" "Isaac ready wait skipped because --dry-run is set"
        return $true
    }

    Write-Stage "isaac_wait" "start" "Waiting for Isaac world_ready event" @{
        event_log = $EventLogPath
        timeout_sec = [int]$TimeoutSec
    }

    $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $EventLogPath) {
            $ready = Select-String -LiteralPath $EventLogPath -SimpleMatch '"action":"world_ready"' -Quiet
            if ($ready) {
                Write-Stage "isaac_wait" "complete" "Isaac world_ready event observed" @{ event_log = $EventLogPath }
                return $true
            }
        }

        try {
            if ($Process -and $Process.HasExited) {
                $null = Write-IsaacFailureDiagnosis -EventLogPath $EventLogPath -RawLogPath $IsaacRawLog
                Write-Stage "isaac_wait" "failed" "Isaac process exited before world_ready" @{
                    exit_code = [int]$Process.ExitCode
                    event_log = $EventLogPath
                }
                return $false
            }
        } catch {
            # Best-effort process status. The event log remains the source of truth.
        }

        Start-Sleep -Seconds 2
    }

    $null = Write-IsaacFailureDiagnosis -EventLogPath $EventLogPath -RawLogPath $IsaacRawLog
    Write-Stage "isaac_wait" "failed" "Timed out waiting for Isaac world_ready" @{
        event_log = $EventLogPath
        timeout_sec = [int]$TimeoutSec
    }
    return $false
}

function Get-WslHostnameIp {
    if ($DryRun) {
        return "<WSL_IP_FROM_hostname_-I>"
    }

    $raw = & wsl.exe -e hostname -I 2>&1
    if ($LASTEXITCODE -ne 0) {
        $message = ConvertTo-CleanText $raw
        Write-Stage "network" "failed" "Could not read WSL IP with hostname -I" @{ error = $message }
        throw "Could not read WSL IP with hostname -I: $message"
    }
    $ip = (($raw -join " ") -split '\s+' | Where-Object { $_ } | Select-Object -First 1)
    if (-not $ip) {
        throw "WSL hostname -I did not return an IP address"
    }
    return $ip
}

function Get-WslWindowsHostIp {
    if ($DryRun) {
        return "<WINDOWS_HOST_IP_FROM_WSL_resolv.conf>"
    }

    $raw = & wsl.exe -e cat /etc/resolv.conf 2>&1
    if ($LASTEXITCODE -ne 0) {
        $message = ConvertTo-CleanText $raw
        Write-Stage "network" "failed" "Could not read /etc/resolv.conf in WSL" @{ error = $message }
        throw "Could not read /etc/resolv.conf in WSL: $message"
    }
    foreach ($line in $raw) {
        if ($line -match '^nameserver\s+(\S+)') {
            return $matches[1]
        }
    }
    throw "Could not find a nameserver entry in WSL /etc/resolv.conf"
}

Write-Stage "setup" "start" "Preparing run_sim launch" @{
    run_log_dir = $RunLogDir
    summary_log = $SummaryLog
    latest_run_file = $LatestRunFile
    repo_root = $RepoRoot
    dry_run = [bool]$DryRun
    skip_build = [bool]$SkipBuild
    force_build = [bool]$ForceBuild
    no_isaac_ready_wait = [bool]$NoIsaacReadyWait
    no_model_preflight = [bool]$NoModelPreflight
    vision_preview = [bool]$VisionPreview
    headless = [bool]$Headless
    no_docker_run = [bool]$NoDockerRun
    self_test_walk = [bool]$SelfTestWalk
    final_scene = [bool]$FinalScene
    pause_after_isaac = [bool]$PauseAfterIsaac
    keep_run_logs = [int]$KeepRunLogs
    trt_engine = $TrtEngine
    sim_frame_timeout_exit_sec = [double]$SimFrameTimeoutExitSec
    locomotion_mode = "parkour"
    parkour_heading_mode = $ParkourHeadingMode
    parkour_mask_fill = $ParkourMaskFill
    sim2real_validation_cam = [bool]$Sim2RealValidationCam
}
Write-Host "Read first: $SummaryLog"
Prune-OldRunLogs -KeepCount $KeepRunLogs

if (-not (Test-ModelPreflight)) {
    Write-Stage "summary" "failed" "Stopping before Isaac because required TensorRT engine files are missing"
    exit 1
}

$WslRepoRoot = ConvertTo-WslPath -WindowsPath $RepoRoot
$WslRunLogDir = ConvertTo-WslPath -WindowsPath $RunLogDir

if ($NoIsaac -and $NoDockerRun) {
    if (-not $FrameHost) {
        $FrameHost = "not-used"
    }
    if (-not $CmdHost) {
        $CmdHost = "not-used"
    }
} else {
    if (-not $FrameHost) {
        # Default to 127.0.0.1 for robust Docker Desktop UDP port forwarding
        $FrameHost = "127.0.0.1"
        Write-Host "Using default 127.0.0.1 for frame_host (Docker Desktop UDP forwarding)"
    }
    if (-not $CmdHost) {
        $CmdHost = "host.docker.internal"
    }
}

Write-Stage "network" "ready" "Resolved sim network endpoints" @{
    frame_host = $FrameHost
    frame_port = $FramePort
    cmd_host = $CmdHost
    cmd_port = $CmdPort
}

# Build Docker before Isaac launches so the container is ready to start
# the moment Isaac reports world_ready.
$dockerImageExists = $false
if (-not $NoDockerRun -and -not $SkipBuild -and -not $ForceBuild) {
    $dockerImageExists = Test-DockerImageExists -ImageName $Image
}

if ($NoDockerRun) {
    Write-Stage "build" "skipped" "Docker image build skipped because Docker run is disabled"
} elseif ($SkipBuild) {
    Write-Stage "build" "skipped" "Docker image build skipped by --skip-build"
} elseif ($dockerImageExists) {
    Write-Stage "build" "skipped" "Using existing Docker image; pass --force-build to rebuild" @{
        image = $Image
        command = ".\run_sim.bat --force-build"
    }
} else {
    $buildLog = Join-Path $DebugDir "docker_build.log"
    $buildCommand = "cd '$WslRepoRoot' && bash docker/docker_build_x86_sim.sh"
    Write-Stage "build" "notice" "Building Docker image before Isaac launches; use --skip-build when the image is already built" @{
        log = $buildLog
        command = ".\run_sim.bat --skip-build"
    }
    $buildExit = Invoke-LoggedCommand -Stage "build" -FilePath "wsl.exe" -Arguments @("-e", "bash", "-lc", $buildCommand) -LogPath $buildLog
    if ($buildExit -ne 0) {
        Write-Stage "summary" "failed" "Stopping before Isaac because Docker build failed" @{
            exit_code = $buildExit
            log = $buildLog
        }
        exit $buildExit
    }
}

if ($NoIsaac) {
    Write-Stage "isaac" "skipped" "Isaac launch skipped by --no-isaac"
} else {
    $IsaacWindowScript = Join-Path $RepoRoot "sim\run_isaac_window.ps1"
    $IsaacRawLog = Join-Path $DebugDir "isaac_raw.log"
    $IsaacFilteredLog = Join-Path $LogsDir "isaac_console.log"
    $IsaacEventLog = Join-Path $DebugDir "isaac_env.jsonl"
    $isaacArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $IsaacWindowScript,
        "-IsaacSimDir", $IsaacSimDir,
        "-RepoRoot", $RepoRoot,
        "-RunLogDir", $RunLogDir,
        "-RawVideoPath", (Join-Path $VideosDir "scene_view.mp4"),
        "-FrameHost", $FrameHost,
        "-FramePort", [string]$FramePort,
        "-CmdPort", [string]$CmdPort,
        "-LocomotionPolicy", $LocomotionPolicy,
        "-PgttLevel", $PgttLevel,
        "-Go2X", [string]$Go2X,
        "-ParkourHeadingMode", $ParkourHeadingMode,
        "-ParkourMaskFill", $ParkourMaskFill,
        # Staircase preset (now selectable via -StairPreset). The launcher previously passed NOTHING,
        # so isaac_env's argparse default 'demo_gentle' (0.08 m) ran silently -- the OOD-shallow toy
        # step the frozen policy face-plants on (regression vs the prior-session intent of 'commercial').
        # Restored to 'commercial' (0.150 m, ~6"/12" ADA): the documented climbable-realistic-middle and
        # the closest-to-in-distribution preset for the frozen Extreme-Parkour policy. NOTE: the step-UP
        # itself is still a frozen-policy limit on every preset (closed-loop control / fine-tune needed);
        # see project_follow_up_stairs_goal + project_stair_approach_commit.
        "-StairPreset", "commercial"
    )
    if ($Sim2RealValidationCam) {
        $isaacArgs += "-Sim2RealValidationCam"
    }
    if ($NoParkourPersonMask) {
        $isaacArgs += "-NoParkourPersonMask"
    }
    if ($WithO2Payload) {
        $isaacArgs += "-WithO2Payload"
    }
    if ($NoParkourWalkMode) {
        $isaacArgs += "-NoParkourWalkMode"
    }
    if ($NoSpeedGovernor) {
        $isaacArgs += "-NoSpeedGovernor"
    }
    if ($Headless) {
        $isaacArgs += "-Headless"
    }
    if ($FastRender) {
        $isaacArgs += "-FastRender"
    }
    if ($FinalScene) {
        $isaacArgs += "-FinalScene"
    }
    if ($SelfTestWalk) {
        $isaacArgs += "-SelfTestWalk"
        $isaacArgs += "-SelfTestVx";  $isaacArgs += [string]$SelfTestVx
        $isaacArgs += "-SelfTestSec"; $isaacArgs += [string]$SelfTestSec
        if ($SelfTestNoPolicy) { $isaacArgs += "-SelfTestNoPolicy" }
        if ($SelfTestHeadingHold) { $isaacArgs += "-SelfTestHeadingHold" }
    }

    # Warm mode: reuse a live warm Isaac if present (skip the ~120s boot); otherwise
    # add the warm flags and clear stale sentinels so the new Kit ignores old commands.
    $warmReuse = $false
    if ($WarmIsaac) {
        $isaacArgs += "-WarmIsaac"
        $isaacArgs += "-WarmCommandFile"; $isaacArgs += $WarmCommandFile
        $isaacArgs += "-WarmMaxRuns";     $isaacArgs += [string]$WarmMaxRuns
        if (Test-WarmIsaacAlive -StatusFile $WarmStatusFile) {
            $warmReuse = $true
        } else {
            Remove-Item -LiteralPath $WarmCommandFile -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $WarmStatusFile -Force -ErrorAction SilentlyContinue
        }
    }

    $isaacCommandLine = Format-CommandLine -FilePath "powershell.exe" -Arguments $isaacArgs
    if ($DryRun) {
        Write-Stage "isaac" "dry-run" "Would open Isaac Sim PowerShell window" @{
            command = $isaacCommandLine
            raw_log = $IsaacRawLog
            console_log = $IsaacFilteredLog
        }
    } elseif ($WarmIsaac -and $warmReuse) {
        # Live warm Isaac: do not launch a new Kit; just command it into this run folder.
        $proc = $null
        Write-Stage "isaac" "warm_reuse" "Reusing live warm Isaac; skipping the ~120s Kit boot" @{
            event_log = $IsaacEventLog
            status_file = $WarmStatusFile
        }
        $seq = Get-NextWarmSeq -CommandFile $WarmCommandFile
        Write-WarmCommand -CommandFile $WarmCommandFile -Seq $seq -Action "begin" -RunDir $RunLogDir
        Write-Stage "isaac" "warm_begin" "Posted warm begin to the live Isaac" @{ command = "seq=$seq"; run_log_dir = $RunLogDir }
    } else {
        $startParams = @{
            FilePath = "powershell.exe"
            ArgumentList = $isaacArgs
            PassThru = $true
        }
        if ($Headless) {
            $startParams["NoNewWindow"] = $true
        }
        $proc = Start-Process @startParams
        $launchMsg = if ($WarmIsaac) { "Booted warm Isaac (this first run pays the ~120s Kit boot)" } else { "Opened Isaac Sim PowerShell window" }
        Write-Stage "isaac" "launched" $launchMsg @{
            pid = $proc.Id
            isaac_sim_dir = $IsaacSimDir
            raw_log = $IsaacRawLog
            console_log = $IsaacFilteredLog
            event_log = $IsaacEventLog
        }
        if ($WarmIsaac) {
            $seq = Get-NextWarmSeq -CommandFile $WarmCommandFile
            Write-WarmCommand -CommandFile $WarmCommandFile -Seq $seq -Action "begin" -RunDir $RunLogDir
            Write-Stage "isaac" "warm_begin" "Posted warm begin to the new Isaac" @{ command = "seq=$seq"; run_log_dir = $RunLogDir }
        }
    }
}

if (-not $NoIsaac -and -not $NoIsaacReadyWait) {
    $ready = Wait-IsaacReady -EventLogPath $IsaacEventLog -Process $proc -TimeoutSec $IsaacReadyTimeoutSec
    if (-not $ready) {
        Write-Stage "summary" "failed" "Stopping before WSL/Docker because Isaac did not report world_ready" @{
            event_log = $IsaacEventLog
        }
        $null = Stop-IsaacProcess -Process $proc -Reason "Isaac did not report world_ready"
        exit 1
    }
} else {
    Write-Stage "isaac_wait" "skipped" "Isaac world_ready wait skipped"
}

if ($PauseAfterIsaac -and -not $NoPauseAfterIsaac -and -not $NoIsaac -and -not $DryRun) {
    Write-Host ""
    Write-Host "Isaac reported world_ready."
    Write-Host "Docker will NOT start until you press Enter here because --pause-after-isaac is set."
    Read-Host "Press Enter to continue to WSL/Docker"
    Write-Stage "operator" "confirmed" "User confirmed Isaac is ready; continuing to WSL/Docker"
} else {
    Write-Stage "operator" "skipped" "Manual Isaac-ready pause skipped; Docker will start automatically"
}

if ($NoDockerRun) {
    Write-Stage "docker" "skipped" "Docker run skipped by --no-docker-run"
    if (-not $NoIsaac -and $proc) {
        Write-Host "Waiting for Isaac Sim process (PID $($proc.Id)) to complete..."
        $totalWait = 0
        $checkInterval = 1
        $timeout = if ($SelfTestWalk) { [int]$SelfTestSec + 60 } else { 300 }
        while (-not $proc.HasExited -and $totalWait -lt $timeout) {
            Start-Sleep -Seconds $checkInterval
            $totalWait += $checkInterval
        }
        if (-not $proc.HasExited) {
            Write-Warning "Isaac Sim process did not exit within $timeout seconds. Terminating..."
            Stop-IsaacProcess -Process $proc -Reason "self-test timeout"
        }
    }
} else {
    Stop-StaleSimContainers `
        -ImageName $Image `
        -PublishedUdpPort $FramePort `
        -ExpectedContainerName $DockerContainerName

    $dockerLog = Join-Path $DebugDir "docker_run.log"
    # Controller-side sense->act latency (core/main.py SimCameraCapture delay buffer).
    # The real-simulated-env preset adds a realistic default (60 ms +/- 20 ms -- a
    # Jetson camera->inference->command pipeline estimate) unless the flags are set
    # explicitly. Tune these once the real pipeline latency is measured.
    $effLatencyMs = $SimLatencyMs
    $effLatencyJitterMs = $SimLatencyJitterMs
    if ($Sim2RealValidationCam -and -not $PSBoundParameters.ContainsKey('SimLatencyMs')) {
        $effLatencyMs = 60.0
    }
    if ($Sim2RealValidationCam -and -not $PSBoundParameters.ContainsKey('SimLatencyJitterMs')) {
        $effLatencyJitterMs = 20.0
    }
    $effSimFrameTimeoutExitSec = $SimFrameTimeoutExitSec
    if ($Headless -and -not $PSBoundParameters.ContainsKey('SimFrameTimeoutExitSec')) {
        $effSimFrameTimeoutExitSec = 90.0
    }
    $visionArgs = @(
        "python3 sim/main.py",
        "--sim",
        "--follow",
        "--follow-backend $FollowBackend",
        "--cmd-host $CmdHost",
        "--cmd-port $CmdPort",
        "--frame-port $FramePort",
        "--trt-engine '$TrtEngine'",
        "--sim-frame-timeout-exit-sec $effSimFrameTimeoutExitSec",
        "--sim-latency-ms $effLatencyMs",
        "--sim-latency-jitter-ms $effLatencyJitterMs",
        # Exact follow/PID shape from recorded top-landing run run_sim_20260618_222324_391.
        # The repaired stair collision gate below retains a separate no-contact floor.
        "--target-distance 0.45",
        # Three-zone cruise/brake: cruise when far, hold within ±tolerance of target, brake when
        # too close (forward-only; the parkour policy floors at ~0.5 m/s and ignores small commands).
        "--trans-x-max 0.85",
        "--trans-x-tolerance 0.12",
        "--trans-x-alpha 0.65",
        "--kp 1.1",
        "--kd 0.15",
        "--follow-standoff-speed-gain 0.4",
        "--follow-pace-distance 2.0",
        "--follow-pace-floor-speed 0.5",
        "--follow-standoff-band-out 0.15",
        "--follow-standoff-band-in -0.15",
        "--follow-stop-ramp-sec 0.7",
        "--stair-speed-scale 0.45",
        # Engage the climb only at the first riser (stair-near-distance 0.45) but RESTORE a
        # sustained walk-up drive there. With forward-floor 0.0 the dog reached the riser on the
        # policy creep, reared up (pitch +13 deg) and BEACHED with cmd_vx=0 (run_sim_20260619_190725).
        # The riser-gated floor only fires once depth<=0.45 (AT the step), so it cannot re-introduce
        # the old far-engage surge; the patient climbs ahead and the 0.65 collision floor blocks contact.
        # The person leaves the camera the instant the climb starts, so the loss-floor governs most of
        # the ascent -- raised 0.16 -> 0.22 because 0.16 was too weak to mount the step.
        "--stair-forward-floor 0.16",
        "--stair-loss-forward-floor 0.22",
        "--stair-target-distance 1.2",
        "--stair-near-distance 0.45",
        "--stair-policy-prepare-distance 1.0",
        "--stair-depth-engage-distance 0.45",
        "--stair-climb-collision-floor 0.65",
        # Square up to the staircase during the APPROACH (stairs detected, climb not yet engaged):
        # steer the heading to centre the staircase bbox so the dog hits the first riser HEAD-ON.
        # A crooked entry (yaw ~30-48 deg, run_sim_20260619_193145) makes the left/right legs meet
        # different step heights -> roll -18 deg -> topple. Gentle + capped (gain 0.5, max 0.4 rad),
        # heading-only; it never touches vx and frozen once the climb engages (then depth self-steer).
        "--stair-square-up",
        "--stair-square-up-gain 0.5",
        "--stair-square-up-max 0.4",
        "--ecs-log-dir /workspace/run_logs/debug/ecs",
        "--debug-trace-dir /workspace/run_logs/debug/debug_trace",
        # Write the OpenCV preview straight into videos/ (no preview-save-dir, which
        # would rmtree its target -- that is why this used to be boxed in a subfolder).
        "--preview-video-path /workspace/run_logs/videos/opencv_preview.mp4",
        "--preview-save-fps 5",
        # scene_view.mp4 is recorded by Isaac from the external scene Left view;
        # disable the controller's raw writer so the robot-POV stream isn't duplicated.
        "--no-raw-video"
    )
    if (-not $VisionPreview) {
        $visionArgs += "--headless"
    }
    # Approach square-up remains opt-in; the known top-landing run kept person-bearing steering.
    if ($StairSquareUp -and -not $NoStairSquareUp) {
        $visionArgs += "--stair-square-up"
    }
    $visionCommand = $visionArgs -join " "
    $byteTrackNumpyAliasFix = "find /opt/bytetrack -type f -name '*.py' -exec sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} + 2>/dev/null"
    $containerCommand = $byteTrackNumpyAliasFix + "; cd /workspace && exec " + $visionCommand

    $dockerArgs = @(
        "-e",
        "docker",
        "run",
        "--rm",
        "--name",
        $DockerContainerName,
        "--label",
        "com.cable.run_sim=true",
        "--gpus",
        "all",
        "-p",
        "${FramePort}:${FramePort}/udp",
        "-v",
        "${WslRepoRoot}:/workspace",
        "-v",
        "${WslRepoRoot}/models:/models",
        "-v",
        "${WslRunLogDir}:/workspace/run_logs",
        "-e",
        "SIM_LOG_DIR=/workspace/run_logs/debug",
        "-w",
        "/workspace",
        $Image,
        "bash",
        "-lc",
        $containerCommand
    )

    $isaacMonitorJob = $null
    $isaacExitedDuringDocker = $false
    $dockerStartTime = Get-Date
    try {
        if (-not $NoIsaac) {
            $isaacMonitorJob = Start-IsaacExitMonitorJob -Process $proc -ContainerName $DockerContainerName -TimeoutSec $MaxRunTimeSec
        }
        $dockerExit = Invoke-LoggedCommand -Stage "docker" -FilePath "wsl.exe" -Arguments $dockerArgs -LogPath $dockerLog
    } finally {
        $dockerDuration = (New-TimeSpan -Start $dockerStartTime -End (Get-Date)).TotalSeconds
        $isTimeout = ($MaxRunTimeSec -gt 0 -and $dockerDuration -ge ($MaxRunTimeSec - 5))
        if ($isTimeout) {
            Write-Stage "docker" "warning" "Docker run timed out after exceeding $MaxRunTimeSec seconds limit" @{
                duration_sec = [math]::Round($dockerDuration, 1)
                limit_sec = $MaxRunTimeSec
            }
        }
        try {
            $isaacExitedDuringDocker = (-not $NoIsaac -and $proc -and $proc.HasExited)
        } catch {
            $isaacExitedDuringDocker = $false
        }
        if ($isaacMonitorJob) {
            Stop-Job -Job $isaacMonitorJob -ErrorAction SilentlyContinue
            Receive-Job -Job $isaacMonitorJob -ErrorAction SilentlyContinue | Out-Null
            Remove-Job -Job $isaacMonitorJob -Force -ErrorAction SilentlyContinue
        }
        $null = Stop-DockerContainer -ContainerName $DockerContainerName -Reason "launcher cleanup"
        # Warm mode leaves Isaac running for the next episode; only the one-shot path
        # tears it down when the Docker controller stops.
        if (-not $WarmIsaac) {
            $null = Stop-IsaacProcess -Process $proc -Reason "Docker controller stopped"
        }
    }
    if ($dockerExit -ne 0) {
        $null = Write-DockerFailureDiagnosis -LogPath $dockerLog
        if ($isaacExitedDuringDocker) {
            Write-Stage "summary" "failed" "Docker stopped because Isaac exited or was closed" @{ log = $dockerLog; container = $DockerContainerName }
        } else {
            Write-Stage "summary" "failed" "Docker run failed" @{ log = $dockerLog; container = $DockerContainerName }
        }
        Invoke-PerfTracker
        exit $dockerExit
    }
    if (Write-DockerFailureDiagnosis -LogPath $dockerLog) {
        Write-Stage "summary" "failed" "Docker run did not start main.py cleanly" @{ log = $dockerLog; container = $DockerContainerName }
        Invoke-PerfTracker
        exit 1
    }
    if (-not $DryRun -and (-not $NoIsaac) -and (Write-SimCompletionGateDiagnosis -EventLog $IsaacEventLog -DockerLog $dockerLog)) {
        Write-Stage "summary" "failed" "Simulation ended before the stair demo reached its required completion gate" @{
            event_log = $IsaacEventLog
            docker_log = $dockerLog
            container = $DockerContainerName
        }
        Invoke-PerfTracker
        exit 1
    }
}

# Append the detailed evaluation summary if it was generated
$evalSummaryFile = Join-Path $ReportsDir "evaluation_summary.txt"
if (Test-Path -LiteralPath $evalSummaryFile) {
    $evalContent = Get-Content -LiteralPath $evalSummaryFile -Raw -ErrorAction SilentlyContinue
    if ($evalContent) {
        Add-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value "`r`n========================================"
        Add-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value "DETAILED SIMULATION EVALUATION:"
        Add-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value "========================================"
        Add-Content -LiteralPath $SummaryLog -Encoding UTF8 -Value $evalContent
    }
}

# --- Update cross-run performance table + auto-charts ---
Invoke-PerfTracker

Write-Stage "summary" "complete" "run_sim completed" @{ run_log_dir = $RunLogDir }
Write-Host ""
Write-Host "Logs for this run:"
Write-Host "  $RunLogDir"
Write-Host "Read first:"
Write-Host "  $SummaryLog"
Write-Host "Status file:"
Write-Host "  $StatusLog"
Write-Host "Performance table:"
Write-Host "  $(Join-Path $RepoRoot 'perf_tracker\data\performance_table.csv')"
