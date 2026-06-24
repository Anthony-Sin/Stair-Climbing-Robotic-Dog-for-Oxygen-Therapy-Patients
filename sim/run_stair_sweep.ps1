<#
.SYNOPSIS
  Baseline step-height sweep for the CURRENT blind-RL climb policy (no retraining).

.DESCRIPTION
  Runs the isolated stair-waypoint test (--stair-waypoint-test) WITH the O2 payload,
  once per realistic riser height, holding the commercial tread depth/count constant
  (0.305 m run x 14 steps) so STEP HEIGHT is the only variable. Same mode as
  `run_sim.bat --stair-waypoint-test --with-o2-payload`: NO person-follow, NO YOLO/Docker
  (the test forces -NoDockerRun), person parked off-lane and ignored -- just the raw climb
  policy driven straight at the waypoint, across multiple risers.

  WARM (default): boots ONE Isaac Kit and reuses it for every height (the riser is changed
  per-episode via the warm command.json -> _warm_run_loop -> configure_stairs), so the
  ~2-min RTX boot is paid ONCE instead of per height. Use -Cold to fully reboot each height.

  After each height it reads that run's fall-diagnostic JSONL (debug/isaac_env.jsonl) -- the
  physics ground-truth, NOT the synthetic demo report -- and records:
    - waypoint PASS/FAIL (robot_reached_stair_waypoint upright + held 2 s)
    - the analyze_climb.py VERDICT (FELL / COLLIDED / CLEAN CLIMB / INCOMPLETE)
    - how many step-runs past the base it actually got before failing

  Use this to decide whether retraining is the right fix: if the policy already
  clean-climbs up to ~0.150 m and only fails at 0.178-0.198 m, retraining on taller
  risers is justified. If it COLLIDES/wedges at every height, the bottleneck is more
  likely PhysX contact dynamics than the policy, and retraining may not help.

.NOTES
  Verify via the kept logs (per the no-run-Isaac-by-eye rule), not by watching the window.
#>
[CmdletBinding()]
param(
    # Riser heights (m) to sweep. Default = realistic homes/hospitals/schools/universities envelope.
    [double[]]$Heights = @(0.100, 0.125, 0.150, 0.178, 0.198),
    # Planar target X (m). 0 = INHERIT run_sim.ps1's geometry-derived default (commercial
    # top landing, 6.5). Only set this to override; the top X does not change with riser height.
    [double]$WaypointX = 0,
    [string]$ClimbBackend = "blind_rl",
    # Keep ALL run logs for the whole sweep (run_sim.ps1 defaults to KeepRunLogs=1, which
    # prunes every prior run when the next height starts -- only the LAST would survive).
    [int]$KeepRunLogs = 9999,
    # Cold mode: fully reboot Isaac for each height (the original ~2-min-boot-per-height path).
    # Default is WARM: boot once, change the riser per episode.
    [switch]$Cold,
    # Leave the warm Isaac alive after the sweep (default: shut it down).
    [switch]$KeepWarm,
    # Hard wall-clock cap PER SIMULATION (s), measured from world_ready (excludes the boot).
    # At the cap the episode is aborted (recordings finalized) and the sweep advances to the
    # next height. Default 300 = 5 min. A key press during a sim also skips to the next height.
    [int]$EpisodeTimeoutSec = 300,
    # Show the Isaac window instead of headless (slower; for eyeballing one height).
    [switch]$Windowed,
    # Skip building the slide-ready presentation pack (montage + graphs + data section) at the end.
    [switch]$NoPresentation,
    # Target length (s) every montage clip is sped/slowed to so they all finish together.
    [double]$MontageSeconds = 20
)

$ErrorActionPreference = "Stop"
$SimDir   = $PSScriptRoot
$RepoRoot = Split-Path -Parent $SimDir
$Launcher = Join-Path $SimDir "run_sim.ps1"
$Analyzer = Join-Path $SimDir "analysis\analyze_climb.py"
$Presenter = Join-Path $SimDir "analysis\sweep_present.py"
$LogDir   = Join-Path $RepoRoot "log"
$SweepStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WarmStatusFile  = Join-Path (Join-Path $LogDir "warm_isaac") "warm_status.json"
$WarmCommandFile = Join-Path (Join-Path $LogDir "warm_isaac") "command.json"

if (-not (Test-Path $Launcher)) { throw "Missing launcher: $Launcher" }

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

# Preempt the running warm episode: post a newer-seq 'abort' command. isaac_env's main
# loop polls _warm_should_abort_episode(), breaks, FINALIZES recordings, and returns to
# idle -- so the videos for this height are kept. The action is not 'begin', so the warm
# loop skips it; the next height posts its own begin (Get-NextWarmSeq carries on from here).
# UTF-8 WITHOUT BOM (Python json.load rejects a BOM -- see CLAUDE.md incident ledger).
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

# Wait for the warm episode to finish: runs_served advanced past the pre-launch count AND
# the loop is back to 'idle'. Enforces the per-sim wall cap and the key-press skip -- on
# either, it preempts the episode (abort) and waits for it to finalize. Returns a status
# string: 'completed' | 'capped' | 'skipped' | 'dead' | 'timeout'. Liveness is judged by
# the pid (the heartbeat can lag during a busy episode).
function Wait-WarmEpisode {
    param([int]$MinRunsServed, [int]$TimeoutSec)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $preempt  = $null   # $null until aborted, then 'capped' or 'skipped'
    # Drain any keystrokes ALREADY buffered before this episode's wait begins -- the Enter used
    # to launch the sweep, or console noise from booting Isaac. Without this, the first poll reads
    # that stale key and instantly "skips" the FIRST height (run ..095709: 0.100 m was preempted
    # at 16 s with no key pressed during the wait). Only keys pressed DURING the wait below skip.
    try { while ([Console]::KeyAvailable) { [void][Console]::ReadKey($true) } } catch {}
    Write-Host "  (press any key to skip this height; hard cap $TimeoutSec s)" -ForegroundColor DarkGray
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
            # A key press (incl. Ctrl+C, which TreatControlCAsInput delivers as a key) skips
            # to the next height. Esc / 'q' quits the whole sweep. Either way the current
            # episode is aborted cleanly so its recordings are finalized first.
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

# --- per-height result parsing (shared by warm + cold) ---------------------
function Read-HeightResult {
    param([double]$H, [string]$Label, [string]$RunDirPath, [int]$Exit)
    $waypoint = "FAIL"; $verdict = "(no fall_diag)"; $steps = "-"
    $jsonl = Join-Path $RunDirPath "debug\isaac_env.jsonl"
    if (Test-Path $jsonl) {
        if (Select-String -Path $jsonl -Pattern "robot_reached_stair_waypoint" -Quiet) {
            $waypoint = "PASS (upright, held 2s)"
        } elseif (Select-String -Path $jsonl -Pattern "stair_waypoint_collision" -Quiet) {
            $waypoint = "FAIL (collided)"
        } elseif (Select-String -Path $jsonl -Pattern '"fall_type":"flipped"' -Quiet) {
            $waypoint = "FAIL (flipped)"
        }
        try {
            $out = & python $Analyzer $RunDirPath 2>&1
            $vline = ($out | Select-String -Pattern "VERDICT:" | Select-Object -First 1)
            if ($vline) { $verdict = ($vline.ToString() -replace ".*VERDICT:\s*", "") }
            $sline = ($out | Select-String -Pattern "step-runs past base" | Select-Object -First 1)
            if ($sline -and ($sline.ToString() -match "~([\d\.]+)\s+step-runs")) { $steps = $Matches[1] }
        } catch { $verdict = "(analyzer error: $_)" }
    } else {
        $waypoint = "NO_JSONL (exit $Exit)"
    }
    return [pscustomobject]@{
        Riser_m = $H; Label = $Label; Waypoint = $waypoint; Verdict = $verdict
        StepsPastBase = $steps; RunDir = (Split-Path $RunDirPath -Leaf)
    }
}

$results = @()
$mode = if ($Cold) { "COLD (reboot per height)" } else { "WARM (boot once, reuse)" }
Write-Host "Stair height sweep -- mode: $mode -- heights: $($Heights -join ', ') m" -ForegroundColor Green

# Capture Ctrl+C as a readable key (not a pipeline-terminating signal) so a key press --
# including Ctrl+C -- SKIPS to the next height instead of killing the whole sweep. Restored
# in the finally below no matter how the loop ends. (Warm mode only; cold blocks in run_sim.)
$origCtrlC = $null
if (-not $Cold) { try { $origCtrlC = [Console]::TreatControlCAsInput; [Console]::TreatControlCAsInput = $true } catch {} }
$quitSweep = $false

# Every sweep invocation starts from a FRESH warm Kit: boot ONCE for THESE heights, reuse across
# them, shut it down at the end. A warm Kit reuses whatever code+args it FIRST booted with (on reuse
# run_sim.ps1 only re-sends the per-episode stair_step_height, NOT -StairWaypointX/-StairPreset or the
# Python), so a Kit left alive by a prior run/stop would silently run STALE code -- the "sweep
# waypoints look off but run_sim.bat is right" symptom. Kill any leftover Kit + clear its sentinels
# here so the first height boots clean. (Cold mode reboots per height already; this is warm-only.)
if (-not $Cold) {
    $st0 = Get-WarmStatus
    $oldPid = if ($st0 -and $st0.pid) { [int]$st0.pid } else { 0 }
    if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Leftover warm Isaac (pid $oldPid) -- shutting it down so the sweep boots CURRENT code/waypoint..." -ForegroundColor Yellow
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
    # Clear stale warm sentinels (AFTER the old Kit is gone) so the fresh Kit never reads an old
    # begin/abort/shutdown command or a stale runs_served baseline.
    foreach ($p in @($WarmCommandFile, $WarmStatusFile)) {
        if ($p -and (Test-Path -LiteralPath $p)) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
    }
    Write-Host ("Warm sweep: booting a FRESH Isaac for {0} heights, then shutting it down at the end." -f $Heights.Count) -ForegroundColor Green
}

try {
foreach ($h in $Heights) {
    $label = $labels["$h"]; if (-not $label) { $label = "custom" }
    $endStatus = "completed"   # 'completed' | 'capped' (5-min) | 'skipped' (key) for this height
    # Commercial footprint is fixed (14 steps x 0.305 m run); only the RISER changes, so
    # the top edge X is constant (6.27 m) but the total climb HEIGHT scales with the riser.
    $topZ = 14 * $h
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host (" RISER {0:N3} m  --  {1}" -f $h, $label) -ForegroundColor Cyan
    Write-Host ("   14 steps -> top landing {0:N2} m high, top edge x=6.27 m" -f $topZ) -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan

    $simArgs = @{
        StairWaypointTest   = $true
        WithO2Payload       = $true
        HandoffClimbBackend = $ClimbBackend
        StairStepHeight     = $h
        KeepRunLogs         = $KeepRunLogs
    }
    if ($WaypointX -gt 0) { $simArgs.StairWaypointX = $WaypointX }
    if (-not $Windowed)   { $simArgs.Headless = $true; $simArgs.FastRender = $true }

    if ($Cold) {
        # Cold: run_sim.ps1 boots, runs, and (NoDockerRun) waits for the process itself.
        & $Launcher @simArgs
        $exit = $LASTEXITCODE
    } else {
        # Warm: one Kit reused across heights. run_sim.ps1 posts the begin + returns after
        # world_ready (it does NOT block on episode end in reuse), so we poll for completion.
        $simArgs.WarmIsaac   = $true
        $simArgs.WarmMaxRuns = ($Heights.Count + 5)   # avoid a mid-sweep self-reboot
        $rs0 = 0
        $st0 = Get-WarmStatus
        if ($st0 -and $st0.runs_served) { $rs0 = [int]$st0.runs_served }
        & $Launcher @simArgs
        $exit = $LASTEXITCODE
        Write-Host "  waiting for warm episode to finish (runs_served > $rs0)..." -ForegroundColor DarkGray
        $epStatus = Wait-WarmEpisode -MinRunsServed ($rs0 + 1) -TimeoutSec $EpisodeTimeoutSec
        if ($epStatus -in @('dead','timeout')) {
            $results += [pscustomobject]@{ Riser_m=$h; Label=$label; Waypoint="WARM_$($epStatus.ToUpper())"; Verdict="-"; StepsPastBase="-"; End=$epStatus; RunDir="-"; RunDirFull="" }
            Write-Host "  ! warm episode failed ($epStatus); falling back to COLD for the rest." -ForegroundColor Red
            $Cold = $true   # degrade gracefully so remaining heights still run
            continue
        }
        # 'completed' | 'capped' (hit 5-min cap) | 'skipped' (key press) all have a finalized run dir.
        $endStatus = $epStatus
    }

    $runDir = Get-ChildItem -Path $LogDir -Directory -Filter "run_sim_*" |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $runDir) {
        $results += [pscustomobject]@{ Riser_m=$h; Label=$label; Waypoint="NO_RUN_DIR"; Verdict="-"; StepsPastBase="-"; End=$endStatus; RunDir="-"; RunDirFull="" }
        continue
    }
    $row = Read-HeightResult -H $h -Label $label -RunDirPath $runDir.FullName -Exit $exit
    $row | Add-Member -NotePropertyName End -NotePropertyValue $endStatus
    $row | Add-Member -NotePropertyName RunDirFull -NotePropertyValue $runDir.FullName
    $results += $row
    Write-Host (" -> {0}  |  {1}  |  ~{2} steps  |  end={3}" -f $row.Waypoint, $row.Verdict, $row.StepsPastBase, $endStatus) -ForegroundColor Yellow
    if ($endStatus -eq 'quit') { $quitSweep = $true; Write-Host "Quit requested -- stopping sweep." -ForegroundColor Yellow; break }
}
} finally {
    if (-not $Cold -and $origCtrlC -ne $null) { try { [Console]::TreatControlCAsInput = $origCtrlC } catch {} }
    # Shut the warm Kit down at the END (unless -KeepWarm) so it never lingers stale into the next
    # sweep. In the finally so it ALSO runs on an error / quit / Ctrl+C. Liveness-checked directly so
    # a mid-sweep cold-degrade is still cleaned up.
    if (-not $KeepWarm) {
        $stEnd = Get-WarmStatus
        if ($stEnd -and $stEnd.pid -and (Get-Process -Id ([int]$stEnd.pid) -ErrorAction SilentlyContinue)) {
            Write-Host ""
            Write-Host "Shutting down warm Isaac (pid $($stEnd.pid)) so the next sweep boots fresh..." -ForegroundColor DarkGray
            & $Launcher -WarmShutdown | Out-Null
        }
    }
}

Write-Host ""
Write-Host "================ BASELINE STEP-HEIGHT SWEEP SUMMARY ================" -ForegroundColor Green
$results | Format-Table -AutoSize Riser_m, Label, Waypoint, StepsPastBase, End, Verdict
Write-Host "Run dirs under: $LogDir" -ForegroundColor DarkGray
$results | ForEach-Object { Write-Host ("  {0:N3} m -> {1}" -f $_.Riser_m, $_.RunDir) -ForegroundColor DarkGray }

# --- presentation pack -----------------------------------------------------
# Bundle every episode's physics log + recorded video into ONE slide-ready folder under
# log\stair_sweep_<stamp>\presentation\: a 2x3 montage (all clips speed-normalized to finish
# together), matplotlib graphs, a stats card, and a CSV/JSON data section. Wrapped in try/catch
# so a presentation hiccup never masks the sweep results printed above. Re-runnable standalone:
#   python sim\analysis\sweep_present.py --manifest log\stair_sweep_<stamp>\manifest.json
if (-not $NoPresentation) {
    try {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python not found on PATH" }
        if (-not (Test-Path $Presenter)) { throw "missing presenter: $Presenter" }
        $SweepDir = Join-Path $LogDir "stair_sweep_$SweepStamp"
        New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null
        $episodes = @()
        foreach ($r in $results) {
            if ($r.RunDirFull -and (Test-Path -LiteralPath $r.RunDirFull)) {
                $episodes += [ordered]@{ height = $r.Riser_m; label = "$($r.Label)"; run_dir = $r.RunDirFull; waypoint = "$($r.Waypoint)" }
            }
        }
        if ($episodes.Count -eq 0) { throw "no completed episodes with run dirs to present" }
        $manifest = [ordered]@{ stamp = $SweepStamp; heights = @($Heights); backend = $ClimbBackend; episodes = @($episodes) }
        $manifestPath = Join-Path $SweepDir "manifest.json"
        $json = $manifest | ConvertTo-Json -Depth 6
        # UTF-8 WITHOUT BOM (Python json.load rejects a BOM -- see CLAUDE.md incident ledger).
        [System.IO.File]::WriteAllText($manifestPath, $json, (New-Object System.Text.UTF8Encoding $false))
        $pres = Join-Path $SweepDir "presentation"
        Write-Host ""
        Write-Host ("Building presentation pack ({0} episodes) -> {1}" -f $episodes.Count, $pres) -ForegroundColor Green
        & python $Presenter --manifest $manifestPath --out $pres --montage-seconds $MontageSeconds
        Write-Host ""
        Write-Host "Presentation pack (drop these into your slides):" -ForegroundColor Green
        Write-Host ("  montage : {0}" -f (Join-Path $pres 'stair_sweep_montage.mp4')) -ForegroundColor Cyan
        Write-Host ("  graphs  : {0}" -f (Join-Path $pres 'graphs')) -ForegroundColor Cyan
        Write-Host ("  card    : {0}" -f (Join-Path $pres 'stats_card.png')) -ForegroundColor Cyan
        Write-Host ("  data    : {0}" -f (Join-Path $pres 'sweep_summary.csv')) -ForegroundColor Cyan
        Write-Host ("  clips   : {0}" -f (Join-Path $pres 'clips')) -ForegroundColor Cyan
        Write-Host ("  (regen anytime: python sim\analysis\sweep_present.py --manifest `"{0}`")" -f $manifestPath) -ForegroundColor DarkGray
    } catch {
        Write-Host "Presentation pack step skipped/failed (sweep results above are unaffected): $_" -ForegroundColor Yellow
    }
}
