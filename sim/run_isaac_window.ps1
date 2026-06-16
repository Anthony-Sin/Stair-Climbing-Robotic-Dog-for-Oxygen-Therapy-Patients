param(
    [Parameter(Mandatory = $true)][string]$IsaacSimDir,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$RunLogDir,
    [string]$RawVideoPath = "",
    [Parameter(Mandatory = $true)][string]$FrameHost,
    [int]$FramePort = 55002,
    [int]$CmdPort = 55001,
    [string]$LocomotionMode = "rl",
    [string]$RlPolicyPath = "",
    [string]$RlPolicyFormat = "auto",
    [double]$RlControlHz = 50.0,
    [double]$RlActionScale = 0.25,
    [string]$RlStairsStrategy = "policy"
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
Write-ConsoleLog "  Locomotion mode:     $LocomotionMode"
if ($RlPolicyPath) {
    Write-ConsoleLog "  RL policy path:      $RlPolicyPath"
}
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

$rlArgs = "--locomotion-mode $LocomotionMode --rl-policy-format $RlPolicyFormat --rl-control-hz $RlControlHz --rl-action-scale $RlActionScale --rl-stairs-strategy $RlStairsStrategy"
if ($RlPolicyPath) {
    $rlArgs = "$rlArgs --rl-policy-path `"$RlPolicyPath`""
}

# Isaac records the external scene Left view to raw_camera.mp4 (beside opencv_preview.mp4).
$rawArg = ""
if ($RawVideoPath) {
    $rawArg = "--raw-video-path `"$RawVideoPath`""
}

$cmdArgs = "/c `"`"$IsaacBat`" `"$IsaacEnv`" --person-move --frame-host $FrameHost --frame-port $FramePort --cmd-port $CmdPort --log-dir `"$RunLogDir`" --no-view-follow-camera $rawArg $rlArgs > `"$RawLog`" 2>&1`""

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
