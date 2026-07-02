#requires -Version 5.1
<#
.SYNOPSIS
    Fast multi-terrain locomotion benchmark for the Go2 (terrain_bench).

.DESCRIPTION
    Boots ONE headless Isaac/Kit (the ~100-120s RTX boot is paid once), then drives
    the frozen parkour policy across a battery of terrains (flat / ramps / stair
    presets) back-to-back inside that warm Kit -- NO Docker, NO vision container, NO
    TensorRT. Each terrain records the same headless videos run_sim does (topdown +
    scene_view) into its own run folder, and the batch is rolled into perf_tracker
    plus a benchmark_summary.

    Terrain catalogue + drive commands: sim/isaac/terrain_bench/terrain_registry.py
    Per-terrain wiring lives behind isaac_env.py's --bench flag (default one-shot path
    is unaffected).
#>
param(
    [string]$IsaacSimDir = $(if ($env:ISAACSIM_DIR) { $env:ISAACSIM_DIR } else { "C:\isaac_sim_600" }),
    [string]$Only = "",              # comma-separated terrain ids to subset/reorder the battery
    [int]$ReadyTimeoutSec = 420,     # warm Kit boot timeout
    [int]$EpisodeTimeoutSec = 220,   # per-terrain wall-clock budget
    [int]$KeepBatches = 3,           # prune old run_bench_* batches beyond this many
    [switch]$NoFastRender,           # default: lighter RaytracedLighting renderer ON
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$SrcRoot  = Join-Path $RepoRoot "src"
$SimDir   = Join-Path $SrcRoot "sim"
$LogRoot  = Join-Path $RepoRoot "log"
$Stamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$BatchName = "run_bench_$Stamp"
$BatchDir  = Join-Path $LogRoot $BatchName
$RunsDir   = Join-Path $BatchDir "runs"
$BootDir   = Join-Path $BatchDir "_boot"
$LauncherLog = Join-Path $BatchDir "launcher.log"

# Dedicated warm state dir so a bench Kit never collides with a run_sim warm Kit
# (run_sim uses log/warm_isaac; bench uses log/warm_bench and passes --bench).
$WarmStateDir    = Join-Path $LogRoot "warm_bench"
$WarmCommandFile = Join-Path $WarmStateDir "command.json"
$WarmStatusFile  = Join-Path $WarmStateDir "warm_status.json"

$IsaacWindowScript = Join-Path $SimDir "run_isaac_window.ps1"
$TerrainBenchDir   = Join-Path (Join-Path $SimDir "isaac") "terrain_bench"
$TerrainRegistry   = Join-Path $TerrainBenchDir "terrain_registry.py"
$BenchMetrics      = Join-Path $TerrainBenchDir "bench_metrics.py"

function ConvertTo-WslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') { throw "Cannot convert to WSL path: $resolved" }
    return "/mnt/$($matches[1].ToLowerInvariant())/$($matches[2].Replace('\','/'))"
}

function Write-Log {
    param([string]$Message)
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $Message
    Write-Host $line
    try { Add-Content -LiteralPath $LauncherLog -Encoding UTF8 -Value $line } catch {}
}

function Get-WarmStatus {
    if (-not (Test-Path -LiteralPath $WarmStatusFile)) { return $null }
    try { return (Get-Content -LiteralPath $WarmStatusFile -Raw -ErrorAction Stop | ConvertFrom-Json) }
    catch { return $null }
}

# Write UTF-8 WITHOUT a BOM. Windows PowerShell 5.1's `-Encoding UTF8` prepends a BOM,
# which Python's json.load (plain open()) / read_text(encoding="utf-8") would reject.
function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding $false))
}

function Write-WarmCommand {
    param([int]$Seq, [string]$Action, [string]$RunDir, $Terrain = $null, $Drive = $null)
    $obj = [ordered]@{
        seq = $Seq; action = $Action; run_dir = $RunDir; stamp = (Get-Date -Format o)
    }
    if ($Terrain) { $obj["terrain"] = $Terrain }
    if ($Drive)   { $obj["drive"]   = $Drive }
    $json = $obj | ConvertTo-Json -Depth 8 -Compress
    $tmp = "$WarmCommandFile.tmp"
    Write-Utf8NoBom -Path $tmp -Text $json
    Move-Item -LiteralPath $tmp -Destination $WarmCommandFile -Force
}

# --- Pre-flight ------------------------------------------------------------
if (-not (Test-Path -LiteralPath $TerrainRegistry)) { throw "Missing terrain registry: $TerrainRegistry" }
if (-not (Test-Path -LiteralPath $IsaacWindowScript)) { throw "Missing Isaac window script: $IsaacWindowScript" }

New-Item -ItemType Directory -Force -Path $BatchDir, $RunsDir, $BootDir, $WarmStateDir | Out-Null

$gitBranch = ""
try { $gitBranch = ((& git -C $RepoRoot rev-parse --abbrev-ref HEAD 2>&1) -join "").Trim() } catch {}

Write-Log "Terrain benchmark batch: $BatchName  (branch=$gitBranch)"

# Read the battery (host-side python; no Isaac needed).
$wslRegistry = ConvertTo-WslPath $TerrainRegistry
$emitArgs = @("-e", "python3", $wslRegistry, "--emit-json")
if ($Only) { $emitArgs += @("--only", $Only) }
$batteryJson = & wsl.exe @emitArgs
if ($LASTEXITCODE -ne 0 -or -not $batteryJson) { throw "Failed to read terrain battery from terrain_registry.py" }
# Force an array so a single-terrain --only subset still has .Count / indexing.
$battery = @(ConvertFrom-Json $batteryJson | ForEach-Object { $_ })
if ($battery.Count -eq 0) { throw "Terrain battery is empty." }
Write-Log ("Battery: " + (($battery | ForEach-Object { $_.terrain_id }) -join ", "))

if ($DryRun) {
    Write-Log "DryRun: would boot one warm Kit and run $($battery.Count) terrains, then aggregate."
    exit 0
}

# --- Clean stale bench Kit / sentinels -------------------------------------
try {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and $_.CommandLine.Contains("isaac_env.py") -and $_.CommandLine.Contains("--bench")
    } | ForEach-Object {
        Write-Log "Killing stale bench Isaac PID $($_.ProcessId)"
        & taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null
    }
} catch {}
Remove-Item -LiteralPath $WarmCommandFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $WarmStatusFile -Force -ErrorAction SilentlyContinue

# --- Boot the warm Kit ONCE -------------------------------------------------
# IMPORTANT: no -RawVideoPath, so scene_view.mp4 derives from each episode's --log-dir
# (isaac_env raw_video_path falls back to the per-episode folder). topdown/lidar/follow
# already derive from --log-dir, which the warm loop retargets per terrain.
$isaacArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $IsaacWindowScript,
    "-IsaacSimDir", $IsaacSimDir,
    "-RepoRoot", $RepoRoot,
    "-RunLogDir", $BootDir,
    "-FrameHost", "127.0.0.1",
    "-WarmIsaac",
    "-WarmCommandFile", $WarmCommandFile,
    "-WarmMaxRuns", [string]($battery.Count + 5),
    "-Bench",
    "-Headless"
)
if (-not $NoFastRender) { $isaacArgs += "-FastRender" }

$renderMode = if ($NoFastRender) { "full-render" } else { "fast-render" }
Write-Log "Booting warm Isaac (headless, $renderMode); boot is paid once..."
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $isaacArgs -PassThru -NoNewWindow

# Wait for boot: warm_status.json state == idle.
$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$booted = $false
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    $st = Get-WarmStatus
    if ($st -and $st.state -eq "idle") { $booted = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $booted) {
    Write-Log "ERROR: warm Isaac did not reach idle within $ReadyTimeoutSec s; aborting."
    if (-not $proc.HasExited) { & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null }
    exit 1
}
Write-Log "Warm Isaac booted (pid=$($st.pid)). Running battery..."

# --- Run the battery, one terrain per warm episode --------------------------
$seq = 0
$manifestTerrains = @()
foreach ($t in $battery) {
    $terrainId = [string]$t.terrain_id
    $runDir = Join-Path $RunsDir ("{0}_{1}" -f $Stamp, $terrainId)
    foreach ($b in @("videos", "reports", "logs", "debug")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $runDir $b) | Out-Null
    }
    $seq++
    $drive = [ordered]@{ vx = [double]$t.drive_vx; sec = [double]$t.max_time_sec }
    Write-WarmCommand -Seq $seq -Action "begin" -RunDir $runDir -Terrain $t -Drive $drive
    Write-Log ("[{0}/{1}] {2}: seq={3} vx={4} sec={5}" -f $seq, $battery.Count, $terrainId, $seq, $t.drive_vx, $t.max_time_sec)

    # Episode done == warm loop returned to idle AT this seq (main() fully returned,
    # videos flushed). Report file is the secondary signal.
    $epDeadline = (Get-Date).AddSeconds($EpisodeTimeoutSec)
    $reportPath = Join-Path $runDir "reports\stair_demo_report.json"
    $epDone = $false
    while ((Get-Date) -lt $epDeadline) {
        if ($proc.HasExited) { break }
        $st = Get-WarmStatus
        if ($st -and $st.state -eq "stopped") { break }
        if ($st -and ([int]$st.seq) -eq $seq -and $st.state -eq "idle") { $epDone = $true; break }
        Start-Sleep -Milliseconds 500
    }

    if ($epDone -and (Test-Path -LiteralPath $reportPath)) {
        Write-Log "    done: $terrainId (report written)"
        $manifestTerrains += [ordered]@{ terrain_id = $terrainId; run_dir = (ConvertTo-WslPath $runDir) }
    } else {
        Write-Log "    WARN: $terrainId did not complete cleanly (timeout/Kit exit); skipping in summary."
        if ($proc.HasExited -or (($st = Get-WarmStatus) -and $st.state -eq "stopped")) {
            Write-Log "    Warm Kit is no longer alive; ending battery early."
            break
        }
    }
}

# --- Shutdown warm Kit ------------------------------------------------------
if (-not $proc.HasExited) {
    $seq++
    Write-WarmCommand -Seq $seq -Action "shutdown" -RunDir ""
    $shutDeadline = (Get-Date).AddSeconds(40)
    while ((Get-Date) -lt $shutDeadline -and -not $proc.HasExited) { Start-Sleep -Seconds 1 }
    if (-not $proc.HasExited) {
        Write-Log "Warm Kit did not exit on shutdown; terminating."
        & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null
    }
}

# --- Manifest + aggregate ---------------------------------------------------
$manifest = [ordered]@{ stamp = $Stamp; git_branch = $gitBranch; terrains = @($manifestTerrains) }
Write-Utf8NoBom -Path (Join-Path $BatchDir "manifest.json") -Text ($manifest | ConvertTo-Json -Depth 5)

if ($manifestTerrains.Count -eq 0) {
    Write-Log "ERROR: no terrain episodes completed; nothing to aggregate."
    exit 1
}

Write-Log "Aggregating $($manifestTerrains.Count) terrain(s) into perf_tracker + benchmark_summary..."
$wslBatch   = ConvertTo-WslPath $BatchDir
$wslMetrics = ConvertTo-WslPath $BenchMetrics
& wsl.exe -e python3 $wslMetrics $wslBatch --git-branch $gitBranch 2>&1 | ForEach-Object { Write-Host $_ }

# --- Prune old batches ------------------------------------------------------
try {
    $keep = [Math]::Max(1, $KeepBatches)
    Get-ChildItem -LiteralPath $LogRoot -Directory -Filter "run_bench_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -Skip $keep | ForEach-Object {
            if ($_.FullName -ne $BatchDir) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
} catch {}

Write-Log "Done. Summary:  $(Join-Path $BatchDir 'benchmark_summary.md')"
Write-Log "Perf table:    $(Join-Path $SrcRoot 'perf_tracker\data\performance_table.csv')"
exit 0
