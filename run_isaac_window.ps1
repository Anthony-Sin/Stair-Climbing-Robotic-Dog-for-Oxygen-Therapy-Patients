param(
    [Parameter(Mandatory = $true)][string]$IsaacSimDir,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$RunLogDir,
    [Parameter(Mandatory = $true)][string]$FrameHost,
    [int]$FramePort = 55002,
    [int]$CmdPort = 55001
)

$ErrorActionPreference = "Stop"
$RawLog = Join-Path $RunLogDir "isaac_raw.log"
$ConsoleLog = Join-Path $RunLogDir "isaac_console.log"
$IsaacEnv = Join-Path $RepoRoot "isaac\isaac_env.py"

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
Write-ConsoleLog "  Isaac JSONL events:  $(Join-Path $RunLogDir 'isaac_env.jsonl')"
Write-ConsoleLog "  Frame target:        ${FrameHost}:${FramePort}"
Write-ConsoleLog "  Command receiver:    0.0.0.0:${CmdPort}"
Write-ConsoleLog ""

if (-not (Test-Path -LiteralPath $IsaacSimDir)) {
    Write-ConsoleLog "ERROR: Isaac Sim directory does not exist: $IsaacSimDir"
    exit 1
}
if (-not (Test-Path -LiteralPath $IsaacEnv)) {
    Write-ConsoleLog "ERROR: Missing Isaac environment script: $IsaacEnv"
    exit 1
}

Set-Location -LiteralPath $IsaacSimDir
$env:PYTHONUNBUFFERED = "1"

& ".\python.bat" $IsaacEnv `
    --person-move `
    --frame-host $FrameHost `
    --frame-port $FramePort `
    --cmd-port $CmdPort `
    --log-dir $RunLogDir `
    2>&1 | ForEach-Object {
        $line = [string]$_
        Add-Content -LiteralPath $RawLog -Encoding UTF8 -Value $line
        if (Should-ShowIsaacLine -Line $line) {
            Write-ConsoleLog $line
        }
    }

$exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
Write-ConsoleLog ""
Write-ConsoleLog "Isaac process exited with code $exitCode"
exit $exitCode
