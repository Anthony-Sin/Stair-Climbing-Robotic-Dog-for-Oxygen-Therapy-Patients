@echo off
setlocal EnableExtensions

rem Fast multi-terrain locomotion benchmark (terrain_bench): boots ONE headless Isaac,
rem drives the frozen policy across the terrain battery (no Docker), and rolls the
rem results into perf_tracker + a benchmark_summary. See sim\run_bench.ps1.

set "REPO_ROOT=%~dp0"
set "LAUNCHER=%REPO_ROOT%run_bench.ps1"
set "PS_ARGS="

if not exist "%LAUNCHER%" (
    echo ERROR: Missing launcher script: "%LAUNCHER%"
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

:parse_args
if "%~1"=="" goto run_launcher
set "ARG=%~1"
if /I "%ARG%"=="--only" set "ARG=-Only"
if /I "%ARG%"=="--isaacsim-dir" set "ARG=-IsaacSimDir"
if /I "%ARG%"=="--ready-timeout-sec" set "ARG=-ReadyTimeoutSec"
if /I "%ARG%"=="--episode-timeout-sec" set "ARG=-EpisodeTimeoutSec"
if /I "%ARG%"=="--keep-batches" set "ARG=-KeepBatches"
if /I "%ARG%"=="--no-fast-render" set "ARG=-NoFastRender"
if /I "%ARG%"=="--dry-run" set "ARG=-DryRun"
set "PS_ARGS=%PS_ARGS% "%ARG%""
shift
goto parse_args

:run_launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" %PS_ARGS%
set "RUN_BENCH_EXIT=%ERRORLEVEL%"

echo.
if not "%RUN_BENCH_EXIT%"=="0" (
    echo run_bench failed with exit code %RUN_BENCH_EXIT%.
) else (
    echo run_bench finished.
)
if not "%NO_PAUSE%"=="1" pause
exit /b %RUN_BENCH_EXIT%
