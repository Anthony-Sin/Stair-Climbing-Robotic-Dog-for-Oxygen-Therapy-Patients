@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0"
set "LAUNCHER=%REPO_ROOT%run_sim.ps1"
set "PS_ARGS="

if not exist "%LAUNCHER%" (
    echo ERROR: Missing launcher script: "%LAUNCHER%"
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

:parse_args
if "%~1"=="" goto run_launcher
set "ARG=%~1"
if /I "%ARG%"=="--skip-build" set "ARG=-SkipBuild"
if /I "%ARG%"=="--force-build" set "ARG=-ForceBuild"
if /I "%ARG%"=="--dry-run" set "ARG=-DryRun"
if /I "%ARG%"=="--no-pause-after-isaac" set "ARG=-NoPauseAfterIsaac"
if /I "%ARG%"=="--pause-after-isaac" set "ARG=-PauseAfterIsaac"
if /I "%ARG%"=="--no-isaac" set "ARG=-NoIsaac"
if /I "%ARG%"=="--no-docker-run" set "ARG=-NoDockerRun"
if /I "%ARG%"=="--no-isaac-ready-wait" set "ARG=-NoIsaacReadyWait"
if /I "%ARG%"=="--isaac-ready-timeout-sec" set "ARG=-IsaacReadyTimeoutSec"
if /I "%ARG%"=="--keep-run-logs" set "ARG=-KeepRunLogs"
if /I "%ARG%"=="--vision-preview" set "ARG=-VisionPreview"
if /I "%ARG%"=="--isaacsim-dir" set "ARG=-IsaacSimDir"
if /I "%ARG%"=="--image" set "ARG=-Image"
if /I "%ARG%"=="--frame-host" set "ARG=-FrameHost"
if /I "%ARG%"=="--cmd-host" set "ARG=-CmdHost"
if /I "%ARG%"=="--cmd-port" set "ARG=-CmdPort"
if /I "%ARG%"=="--frame-port" set "ARG=-FramePort"
if /I "%ARG%"=="--follow-backend" set "ARG=-FollowBackend"
if /I "%ARG%"=="--trt-engine" set "ARG=-TrtEngine"
if /I "%ARG%"=="--sim-frame-timeout-exit-sec" set "ARG=-SimFrameTimeoutExitSec"
if /I "%ARG%"=="--no-model-preflight" set "ARG=-NoModelPreflight"
if /I "%ARG%"=="--max-run-time-sec" set "ARG=-MaxRunTimeSec"
if /I "%ARG%"=="--locomotion-policy" set "ARG=-LocomotionPolicy"
if /I "%ARG%"=="--pgtt-level" set "ARG=-PgttLevel"
if /I "%ARG%"=="--pgtt-action-scale" set "ARG=-PgttActionScale"
if /I "%ARG%"=="--pgtt-heightscan-scale" set "ARG=-PgttHeightscanScale"
if /I "%ARG%"=="--go2-x" set "ARG=-Go2X"
if /I "%ARG%"=="--target-distance" set "ARG=-TargetDistance"
if /I "%ARG%"=="--parkour-heading-mode" set "ARG=-ParkourHeadingMode"
if /I "%ARG%"=="--sim2real-validation-cam" set "ARG=-Sim2RealValidationCam"
if /I "%ARG%"=="--sim-latency-ms" set "ARG=-SimLatencyMs"
if /I "%ARG%"=="--sim-latency-jitter-ms" set "ARG=-SimLatencyJitterMs"
if /I "%ARG%"=="--self-test-walk" set "ARG=-SelfTestWalk"
if /I "%ARG%"=="--self-test-vx" set "ARG=-SelfTestVx"
if /I "%ARG%"=="--self-test-sec" set "ARG=-SelfTestSec"
if /I "%ARG%"=="--self-test-no-policy" set "ARG=-SelfTestNoPolicy"
if /I "%ARG%"=="--self-test-heading-hold" set "ARG=-SelfTestHeadingHold"
if /I "%ARG%"=="--self-test-stairs" set "ARG=-SelfTestStairs"
if /I "%ARG%"=="--handoff-climb-backend" set "ARG=-HandoffClimbBackend"
if /I "%ARG%"=="--stair-waypoint-test" set "ARG=-StairWaypointTest"
if /I "%ARG%"=="--stair-waypoint-x" set "ARG=-StairWaypointX"
if /I "%ARG%"=="--stair-waypoint-y" set "ARG=-StairWaypointY"
if /I "%ARG%"=="--stair-step-height" set "ARG=-StairStepHeight"
if /I "%ARG%"=="--no-parkour-person-mask" set "ARG=-NoParkourPersonMask"
if /I "%ARG%"=="--parkour-mask-fill" set "ARG=-ParkourMaskFill"
if /I "%ARG%"=="--stair-square-up" set "ARG=-StairSquareUp"
if /I "%ARG%"=="--no-stair-square-up" set "ARG=-NoStairSquareUp"
if /I "%ARG%"=="--with-o2-payload" set "ARG=-WithO2Payload"
if /I "%ARG%"=="--no-o2-payload" set "ARG=-NoO2Payload"
if /I "%ARG%"=="--no-parkour-walk-mode" set "ARG=-NoParkourWalkMode"
if /I "%ARG%"=="--no-speed-governor" set "ARG=-NoSpeedGovernor"
if /I "%ARG%"=="--no-stand-up-from-ground" set "ARG=-NoStandUpFromGround"
if /I "%ARG%"=="--patient-character-usd" set "ARG=-PatientCharacterUsd"
if /I "%ARG%"=="--no-hold-motion" set "ARG=-NoHoldMotion"
if /I "%ARG%"=="--headless" set "ARG=-Headless"
if /I "%ARG%"=="--fast-render" set "ARG=-FastRender"
if /I "%ARG%"=="--warm" set "ARG=-WarmIsaac"
if /I "%ARG%"=="--warm-isaac" set "ARG=-WarmIsaac"
if /I "%ARG%"=="--warm-shutdown" set "ARG=-WarmShutdown"
if /I "%ARG%"=="--warm-max-runs" set "ARG=-WarmMaxRuns"
set "PS_ARGS=%PS_ARGS% "%ARG%""
shift
goto parse_args

:run_launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" %PS_ARGS%
set "RUN_SIM_EXIT=%ERRORLEVEL%"

echo.
if not "%RUN_SIM_EXIT%"=="0" (
    echo run_sim failed with exit code %RUN_SIM_EXIT%.
) else (
    echo run_sim finished.
)
if not "%NO_PAUSE%"=="1" pause
exit /b %RUN_SIM_EXIT%
