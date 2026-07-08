<#
.SYNOPSIS
  Person-follow step-height sweep using the full run_sim.bat pipeline.

.DESCRIPTION
  Runs the COMPLETE person-follow pipeline (Docker YOLO controller + Isaac Sim) across
  realistic riser heights, holding the commercial tread depth/count constant (0.305 m run x
  14 steps) so STEP HEIGHT is the only variable. Same mode as `run_sim.bat` with -WithO2Payload:
  Docker controller active, person spawned at default patrol position (-3.5, 0), robot starts
  at default -4.5 and follows via YOLO detection and the PGTT/blind-RL handoff policy.

  WARM (default): boots ONE Isaac Kit and reuses it for every height. Docker starts WITH the
  first Isaac boot and persists across warm-reuse episodes -- the controller keeps following
  whoever it detects regardless of stair height changes between episodes.
  Use -Cold to fully reboot Isaac (and Docker) for each height.

  After each height it reads that run's fall-diagnostic JSONL (debug/isaac_env.jsonl) and records:
    - ReachStatus: REACHED TOP / DID NOT REACH / FELL (flipped/collapsed)
    - the analyze_climb.py VERDICT (FELL / COLLIDED / CLEAN CLIMB / INCOMPLETE)
    - how many step-runs past the base the robot got before failing

  This sweep tests the FULL system path: person detection, follow, stair approach, climb.
  Compare with run_stair_sweep.ps1 (waypoint-only, no person/Docker) to isolate whether
  failures are in the climb policy itself or in the follow-and-approach pipeline.

.NOTES
  Verify via the kept logs (per the no-run-Isaac-by-eye rule), not by watching the window.
#>
[CmdletBinding()]
param(
    # Riser heights (m) to sweep. Default = realistic homes/hospitals/schools/universities envelope.
    [double[]]$Heights = @(0.100, 0.125, 0.150, 0.178, 0.198),
    [string]$ClimbBackend = "blind_rl",
    # Keep ALL run logs for the whole sweep (run_sim.ps1 defaults to KeepRunLogs=1, which
    # prunes every prior run when the next height starts -- only the LAST would survive).
    [int]$KeepRunLogs = 9999,
    # Cold mode: fully reboot Isaac (and Docker) for each height.
    # Default is WARM: boot once, change the riser per episode.
    [switch]$Cold,
    # Leave the warm Isaac alive after the sweep (default: shut it down).
    [switch]$KeepWarm,
    # Hard wall-clock cap PER SIMULATION (s), measured from world_ready (excludes the boot).
    # At the cap the episode is aborted (recordings finalized) and the sweep advances to the
    # next height. Default 1200 = 20 min. A key press during a sim also skips to the next height.
    # Isaac exits early only on a robot fall; the cap handles wedged/stalled runs.
    [int]$EpisodeTimeoutSec = 1200,
    # Show the Isaac window instead of headless (slower; for eyeballing one height).
    [switch]$Windowed,
    # Skip building the slide-ready presentation pack (montage + graphs + data section) at the end.
    [switch]$NoPresentation,
    # Target length (s) every montage clip is sped/slowed to so they all finish together.
    [double]$MontageSeconds = 20,
    # Number of lateral zigzag turns the person makes on the flat approach before the stairs.
    # 0 (default) = straight-line approach -- IDENTICAL to how run_sim.bat is set up, so the
    #   sweep reproduces the same follow conditions the user runs by hand.
    # A non-zero zigzag (e.g. 2 = right then left, re-centre) forces the robot to steer
    #   left/right. The geometry is now REALISTIC-PATIENT (gentle 0.5 m weave at a 1.0 m follow
    #   standoff -- see -PersonApproachAmplitude / -TargetDistance), which keeps the patient
    #   inside the 69 deg RGB/YOLO cone instead of the old aggressive 1.2 m / 0.6 m swing that
    #   threw the box off-axis for seconds at a time. Combined with the fixed lost-target
    #   recovery (yaw bridge + LiDAR bearing), the dog tracks the weave instead of freezing.
    [int]$PersonApproachTurns = 0,
    # Lateral amplitude (m) per zigzag turn. 0.5 = realistic gentle weave (was 1.2, which pushed
    # the patient ~40-47 deg off-axis at each apex, past the camera's +/-34.5 deg half-FOV).
    [double]$PersonApproachAmplitude = 0.5,
    # Follow standoff (m) the controller holds behind the patient. 1.0 = realistic O2-patient
    # spacing; wider than run_sim.bat's 0.6 so a weaving patient subtends a smaller bearing and
    # stays in frame. (run_sim.bat itself is unchanged -- this only affects the sweep.)
    [double]$TargetDistance = 1.0
)

$ErrorActionPreference = "Stop"
$SimDir   = $PSScriptRoot
$SrcRoot  = Split-Path -Parent $SimDir
$RepoRoot = Split-Path -Parent $SrcRoot
$Launcher = Join-Path $SimDir "run_sim.ps1"
$Analyzer = Join-Path $SimDir "analysis\analyze_climb.py"
$Presenter = Join-Path $SimDir "analysis\follow_present.py"
$LogDir   = Join-Path $RepoRoot "log"
$SweepStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WarmStatusFile  = Join-Path (Join-Path $LogDir "warm_isaac") "warm_status.json"
$WarmCommandFile = Join-Path (Join-Path $LogDir "warm_isaac") "command.json"

if (-not (Test-Path $Launcher)) { throw "Missing launcher: $Launcher" }

# Clear previous sweep data so each run starts fresh.
Write-Host "Clearing previous follow sweep logs..." -ForegroundColor DarkGray
Get-ChildItem -Path $LogDir -Directory -Filter "follow_sweep_*" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  removed $($_.Name)" -ForegroundColor DarkGray }
Get-ChildItem -Path $LogDir -Directory -Filter "run_sim_*" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  removed $($_.Name)" -ForegroundColor DarkGray }
Write-Host "Done clearing." -ForegroundColor DarkGray

$labels = @{
    "0.1"   = "accessible/gentle (~4in)"
    "0.125" = "hospital/accessible (~5in)"
    "0.15"  = "commercial: schools/hospitals/uni (~6in, ADA)"
    "0.178" = "residential homes / IBC max (~7in)"
    "0.198" = "steep code-max (~7.75in)"
}

# --- warm helpers ----------------------------------------------------------
function Get-WarmStatus {
    if (-not (Test-Path -LiteralPath $WarmStatusFile)) { return $null }
    try { return (Get-Content -LiteralPath $WarmStatusFile -Raw -ErrorAction Stop | ConvertFrom-Json) }
    catch { return $null }
}

# Preempt the running warm episode: post a newer-seq 'abort' command so isaac_env finalizes
# recordings and returns to idle. UTF-8 WITHOUT BOM (Python json.load rejects a BOM).
function Post-WarmAbort {
    $seq = 1
    if (Test-Path -LiteralPath $WarmCommandFile) {
        try { $seq = [int]((Get-Content -LiteralPath $WarmCommandFile -Raw | ConvertFrom-Json).seq) + 1 } catch { $seq = 1 }
    }
    $obj = [ordered]@{ seq = $seq; action = 'abort'; run_dir = ''; stair_step_height = 0; stamp = (Get-Date -Format o) }
    $json = $obj | ConvertTo-Json -Compress
    $tmp = "$WarmCommandFile.tmp"
    [System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding $false))
    Move-Item -LiteralPath $tmp -Destination $WarmCommandFile -Force
}

function Wait-WarmEpisode {
    param([int]$MinRunsServed, [int]$TimeoutSec)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $preempt  = $null
    # Drain any keystrokes already buffered (e.g. Enter used to launch the sweep).
    try { while ([Console]::KeyAvailable) { [void][Console]::ReadKey($true) } } catch {}
    Write-Host "  (press any key to skip this height; Esc/q to quit sweep; hard cap $TimeoutSec s)" -ForegroundColor DarkGray
    while ($true) {
        Start-Sleep -Milliseconds 1500
        $st = Get-WarmStatus
        if ($st -and ([int]$st.runs_served -ge $MinRunsServed) -and ([string]$st.state -eq 'idle')) {
            if ($preempt) { return $preempt } else { return 'completed' }
        }
        if ($st -and $st.pid -and -not (Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue)) {
            Write-Host "  ! warm Isaac process (pid $($st.pid)) is gone" -ForegroundColor Red
            return 'dead'
        }
        if (-not $preempt) {
            $key = $null
            try { if ([Console]::KeyAvailable) { $key = [Console]::ReadKey($true) } } catch {}
            $quit = $key -and (($key.Key -eq 'Escape') -or ($key.KeyChar -eq 'q') -or ($key.KeyChar -eq 'Q'))
            if ($key -or ((Get-Date) -ge $deadline)) {
                if     ($quit) { $preempt = 'quit' }
                elseif ($key)  { $preempt = 'skipped' }
                else           { $preempt = 'capped' }
                $why = switch ($preempt) { 'quit' { 'quit key' } 'skipped' { 'key press' } default { "$TimeoutSec s cap" } }
                Write-Host "  -> ending this sim early ($why); finalizing recordings + advancing" -ForegroundColor Yellow
                Post-WarmAbort
                $deadline = (Get-Date).AddSeconds(120)   # grace for finalize + return to idle
            }
        } elseif ((Get-Date) -ge $deadline) {
            Write-Host "  ! episode did not finalize after abort" -ForegroundColor Red
            return 'timeout'
        }
    }
}

# Read result from the JSONL for a follow-mode run (no waypoint event; judge by fall type + max_x).
function Read-HeightResult {
    param([double]$H, [string]$Label, [string]$RunDirPath, [int]$Exit)
    $reachStatus = "DID NOT REACH"; $verdict = "(no fall_diag)"; $steps = "-"
    $jsonl = Join-Path $RunDirPath "debug\isaac_env.jsonl"
    if (Test-Path $jsonl) {
        # Determine reach / fall status from the JSONL directly.
        if (Select-String -Path $jsonl -Pattern '"fall_type":"flipped"' -Quiet) {
            $reachStatus = "FELL (flipped)"
        } elseif (Select-String -Path $jsonl -Pattern '"fall_type":"collapsed_low"' -Quiet) {
            $reachStatus = "FELL (collapsed)"
        }
        try {
            $out = & python $Analyzer $RunDirPath 2>&1
            $vline = ($out | Select-String -Pattern "VERDICT:" | Select-Object -First 1)
            if ($vline) { $verdict = ($vline.ToString() -replace ".*VERDICT:\s*", "") }
            $sline = ($out | Select-String -Pattern "step-runs past base" | Select-Object -First 1)
            if ($sline -and ($sline.ToString() -match "~([\d\.]+)\s+step-runs")) { $steps = $Matches[1] }
            # Override reach status if the robot made it to the top area (max_x near 6.27 m top edge).
            $mline = ($out | Select-String -Pattern "max_x\s*=\s*[\d\.]+" | Select-Object -First 1)
            if ($mline -and ($mline.ToString() -match "max_x\s*=\s*([\d\.]+)")) {
                if ([double]$Matches[1] -ge 5.8) { $reachStatus = "REACHED TOP" }
            }
        } catch { $verdict = "(analyzer error: $_)" }
    } else {
        $reachStatus = "NO_JSONL (exit $Exit)"
    }
    return [pscustomobject]@{
        Riser_m = $H; Label = $Label; ReachStatus = $reachStatus; Verdict = $verdict
        StepsPastBase = $steps; RunDir = (Split-Path $RunDirPath -Leaf)
    }
}

$results = @()
$mode = if ($Cold) { "COLD (reboot per height)" } else { "WARM (boot once, reuse)" }
Write-Host "Person-follow stair sweep -- mode: $mode -- heights: $($Heights -join ', ') m" -ForegroundColor Magenta

# Capture Ctrl+C as a readable key so it SKIPS to the next height instead of killing the sweep.
$origCtrlC = $null
if (-not $Cold) { try { $origCtrlC = [Console]::TreatControlCAsInput; [Console]::TreatControlCAsInput = $true } catch {} }
$quitSweep = $false

# Kill any leftover warm Kit so the sweep boots CURRENT code. Docker will restart with the new Kit.
if (-not $Cold) {
    $st0 = Get-WarmStatus
    $oldPid = if ($st0 -and $st0.pid) { [int]$st0.pid } else { 0 }
    if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Leftover warm Isaac (pid $oldPid) -- shutting it down so the sweep boots fresh..." -ForegroundColor Yellow
        & $Launcher -WarmShutdown | Out-Null
        $deadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $deadline -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 500
        }
        if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
            Write-Host "  graceful shutdown timed out -- force-killing pid $oldPid" -ForegroundColor Red
            try { Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue } catch {}
        }
        Write-Host "  ...leftover Kit stopped." -ForegroundColor DarkGray
    }
    foreach ($p in @($WarmCommandFile, $WarmStatusFile)) {
        if ($p -and (Test-Path -LiteralPath $p)) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
    }
    Write-Host ("Warm sweep: booting a FRESH Isaac for {0} heights (Docker starts with first boot)." -f $Heights.Count) -ForegroundColor Magenta
}

try {
foreach ($h in $Heights) {
    $label = $labels["$h"]; if (-not $label) { $label = "custom" }
    $endStatus = "completed"
    $topZ = 14 * $h
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Magenta
    Write-Host (" RISER {0:N3} m  --  {1}" -f $h, $label) -ForegroundColor Magenta
    Write-Host ("   14 steps -> top landing {0:N2} m high  |  top edge x=6.27 m" -f $topZ) -ForegroundColor Magenta
    Write-Host ("   Person: default patrol (-3.5, 0) -> stairs -> top landing") -ForegroundColor Magenta
    Write-Host "=================================================================" -ForegroundColor Magenta

    # Full person-follow mode: no StairWaypointTest, no NoDockerRun, person at default spawn.
    # Docker starts with the first Isaac boot (warm) and persists across episodes.
    $simArgs = @{
        WithO2Payload            = $true
        HandoffClimbBackend      = $ClimbBackend
        StairStepHeight          = $h
        KeepRunLogs              = $KeepRunLogs
        PersonApproachTurns      = $PersonApproachTurns
        PersonApproachAmplitude  = $PersonApproachAmplitude
        TargetDistance           = $TargetDistance
    }
    if (-not $Windowed) { $simArgs.Headless = $true; $simArgs.FastRender = $true }

    if ($Cold) {
        & $Launcher @simArgs
        $exit = $LASTEXITCODE
    } else {
        $simArgs.WarmIsaac   = $true
        $simArgs.WarmDocker  = $true
        $simArgs.WarmMaxRuns = ($Heights.Count + 5)
        $rs0 = 0
        $st0 = Get-WarmStatus
        if ($st0 -and $st0.runs_served) { $rs0 = [int]$st0.runs_served }
        & $Launcher @simArgs
        $exit = $LASTEXITCODE
        Write-Host "  waiting for warm episode to finish (runs_served > $rs0)..." -ForegroundColor DarkGray
        $epStatus = Wait-WarmEpisode -MinRunsServed ($rs0 + 1) -TimeoutSec $EpisodeTimeoutSec
        if ($epStatus -in @('dead', 'timeout')) {
            $results += [pscustomobject]@{ Riser_m=$h; Label=$label; ReachStatus="WARM_$($epStatus.ToUpper())"; Verdict="-"; StepsPastBase="-"; End=$epStatus; RunDir="-"; RunDirFull="" }
            Write-Host "  ! warm episode failed ($epStatus); falling back to COLD for the rest." -ForegroundColor Red
            $Cold = $true
            continue
        }
        $endStatus = $epStatus
    }

    $runDir = Get-ChildItem -Path $LogDir -Directory -Filter "run_sim_*" |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $runDir) {
        $results += [pscustomobject]@{ Riser_m=$h; Label=$label; ReachStatus="NO_RUN_DIR"; Verdict="-"; StepsPastBase="-"; End=$endStatus; RunDir="-"; RunDirFull="" }
        continue
    }
    $row = Read-HeightResult -H $h -Label $label -RunDirPath $runDir.FullName -Exit $exit
    $row | Add-Member -NotePropertyName End -NotePropertyValue $endStatus
    $row | Add-Member -NotePropertyName RunDirFull -NotePropertyValue $runDir.FullName
    $results += $row
    Write-Host (" -> {0}  |  {1}  |  ~{2} steps  |  end={3}" -f $row.ReachStatus, $row.Verdict, $row.StepsPastBase, $endStatus) -ForegroundColor Yellow
    if ($endStatus -eq 'quit') { $quitSweep = $true; Write-Host "Quit requested -- stopping sweep." -ForegroundColor Yellow; break }
}
} finally {
    if (-not $Cold -and $origCtrlC -ne $null) { try { [Console]::TreatControlCAsInput = $origCtrlC } catch {} }
    if (-not $KeepWarm) {
        $stEnd = Get-WarmStatus
        if ($stEnd -and $stEnd.pid -and (Get-Process -Id ([int]$stEnd.pid) -ErrorAction SilentlyContinue)) {
            Write-Host ""
            Write-Host "Shutting down warm Isaac (pid $($stEnd.pid)) so the next sweep boots fresh..." -ForegroundColor DarkGray
            & $Launcher -WarmShutdown | Out-Null
        }
        Write-Host "Shutting down warm Docker container so the next sweep boots fresh..." -ForegroundColor DarkGray
        & wsl.exe -e bash -c "docker rm -f go2-warm-sim >/dev/null 2>&1"
    }
}

Write-Host ""
Write-Host "================ PERSON-FOLLOW STEP-HEIGHT SWEEP SUMMARY ================" -ForegroundColor Magenta
$results | Format-Table -AutoSize Riser_m, Label, ReachStatus, StepsPastBase, End, Verdict
Write-Host "Run dirs under: $LogDir" -ForegroundColor DarkGray
$results | ForEach-Object { Write-Host ("  {0:N3} m -> {1}" -f $_.Riser_m, $_.RunDir) -ForegroundColor DarkGray }

# --- presentation pack -----------------------------------------------------
if (-not $NoPresentation) {
    try {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python not found on PATH" }
        if (-not (Test-Path $Presenter)) { throw "missing presenter: $Presenter" }
        $SweepDir = Join-Path $LogDir "follow_sweep_$SweepStamp"
        New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null
        $episodes = @()
        foreach ($r in $results) {
            if ($r.RunDirFull -and (Test-Path -LiteralPath $r.RunDirFull)) {
                $episodes += [ordered]@{ height = $r.Riser_m; label = "$($r.Label)"; run_dir = $r.RunDirFull; follow_reach = "$($r.ReachStatus)" }
            }
        }
        if ($episodes.Count -eq 0) { throw "no completed episodes with run dirs to present" }
        $manifest = [ordered]@{ stamp = $SweepStamp; mode = "follow"; heights = @($Heights); climb_backend = $ClimbBackend; episodes = @($episodes) }
        $manifestPath = Join-Path $SweepDir "manifest.json"
        $json = $manifest | ConvertTo-Json -Depth 6
        # UTF-8 WITHOUT BOM (Python json.load rejects a BOM -- see CLAUDE.md incident ledger).
        [System.IO.File]::WriteAllText($manifestPath, $json, (New-Object System.Text.UTF8Encoding $false))
        $pres = Join-Path $SweepDir "presentation"
        Write-Host ""
        Write-Host ("Building presentation pack ({0} episodes) -> {1}" -f $episodes.Count, $pres) -ForegroundColor Magenta
        & python $Presenter --manifest $manifestPath --out $pres --montage-seconds $MontageSeconds
        Write-Host ""
        Write-Host "Presentation pack (drop these into your slides):" -ForegroundColor Magenta
        Write-Host ("  viewer  : {0}" -f (Join-Path $pres 'viewer.html')) -ForegroundColor Green
        Write-Host ("  montage : {0}" -f (Join-Path $pres 'follow_sweep_montage.mp4')) -ForegroundColor Cyan
        Write-Host ("  graphs  : {0}" -f (Join-Path $pres 'graphs')) -ForegroundColor Cyan
        Write-Host ("  card    : {0}" -f (Join-Path $pres 'stats_card.png')) -ForegroundColor Cyan
        Write-Host ("  data    : {0}" -f (Join-Path $pres 'sweep_summary.csv')) -ForegroundColor Cyan
        Write-Host ("  clips   : {0}" -f (Join-Path $pres 'clips')) -ForegroundColor Cyan
        Write-Host ("  (regen anytime: python src\sim\analysis\follow_present.py --manifest `"{0}`")" -f $manifestPath) -ForegroundColor DarkGray
    } catch {
        Write-Host "Presentation pack step skipped/failed (sweep results above are unaffected): $_" -ForegroundColor Yellow
    }
}
