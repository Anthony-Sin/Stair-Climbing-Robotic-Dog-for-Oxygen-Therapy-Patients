# train_windows_rl.ps1 - Native Windows blind-RL stair training runner.
# Activates the local venv, configures environment, stages the checkpoint, and runs train_rl.py.

$ErrorActionPreference = "Stop"

# Paths
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$VenvDir = if ($env:FT_RL_VENV_DIR) { $env:FT_RL_VENV_DIR } else { Join-Path $env:USERPROFILE ".venv_rl" }

# Tunable defaults (can be overridden via environment variables)
$RobotLabDir = if ($env:FT_RL_REPO_DIR) { $env:FT_RL_REPO_DIR } else { Join-Path $env:USERPROFILE "robot_lab" }
$IsaacLabDir = if ($env:FT_RL_ISAACLAB_DIR) { $env:FT_RL_ISAACLAB_DIR } else { Join-Path $env:USERPROFILE "IsaacLab" }
$ExptId = if ($env:FT_RL_EXPTID) { $env:FT_RL_EXPTID } else { "unitree_go2_rough" }
$LoadRun = if ($env:FT_RL_LOAD_RUN) { $env:FT_RL_LOAD_RUN } else { "o2stair_local" }
$NumEnvs = if ($env:FT_RL_NUM_ENVS) { $env:FT_RL_NUM_ENVS } else { 1024 }
$AddIters = if ($env:FT_RL_MAX_ITERS) { $env:FT_RL_MAX_ITERS } else { 4000 }
$ResumeCkpt = if ($env:FT_RL_RESUME_CKPT) { $env:FT_RL_RESUME_CKPT } else { "model_3000.pt" }
$DL = if ($env:FT_RL_DOWNLOADS) { $env:FT_RL_DOWNLOADS } else { Join-Path $env:USERPROFILE "Downloads" }
$ResumeSrc = if ($env:FT_RL_RESUME_SRC) { $env:FT_RL_RESUME_SRC } else { Join-Path $DL "o2stair_run\trained_o2stair" }
$ResumeTar = if ($env:FT_RL_RESUME_TAR) { $env:FT_RL_RESUME_TAR } else { Join-Path $DL "_o2stair_tmp\trained_o2stair_FULL.tar" }

# Reward tuning defaults (overridable via env)
$Orient = if ($env:FT_RL_ORIENT_REWARD) { $env:FT_RL_ORIENT_REWARD } else { "-0.5" }
$Ascent = if ($env:FT_RL_ASCENT_REWARD) { $env:FT_RL_ASCENT_REWARD } else { "3.0" }
$TrackLinVel = if ($env:FT_RL_TRACK_LINVEL_W) { $env:FT_RL_TRACK_LINVEL_W } else { "1.5" }
$MaxLevel = if ($env:FT_RL_MAX_INIT_TERRAIN_LEVEL) { $env:FT_RL_MAX_INIT_TERRAIN_LEVEL } else { 8 }
$LinVelX = if ($env:FT_RL_LINVELX_MAX) { $env:FT_RL_LINVELX_MAX } else { "0.4" }
$ExploreExtra = if ($env:FT_RL_EXPLORE_EXTRA) { $env:FT_RL_EXPLORE_EXTRA } else { "agent.algorithm.entropy_coef=0.02 agent.policy.init_noise_std=1.2" }

Write-Host "============================================="
Write-Host "  NATIVE WINDOWS ISAAC TRAINING RUNNER"
Write-Host "============================================="
Write-Host "repo=$RobotLabDir  exptid=$ExptId  load_run=$LoadRun  num_envs=$NumEnvs  +iters=$AddIters"
Write-Host "resume_ckpt=$ResumeCkpt  orient=$Orient  ascent=$Ascent  track_linvel=$TrackLinVel  max_level=$MaxLevel  linvel_x=$LinVelX"
Write-Host "explore=$ExploreExtra"
Write-Host "============================================="

# Verify setup
if (-not (Test-Path $VenvDir)) {
    Write-Error "ERROR: Virtual environment not found at $VenvDir. Run setup_windows_rl.ps1 first."
}

# Activate virtual environment
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host "Activating virtual environment..."
. $ActivateScript

# Stage the resume checkpoint
Write-Host "`n== Staging Checkpoints =="
$DestDir = Join-Path $RobotLabDir "logs\rsl_rl\$ExptId\$LoadRun"
$ResumePath = Join-Path $DestDir $ResumeCkpt
if (-not (Test-Path $ResumePath)) {
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    $ResumeSrcPath = Join-Path $ResumeSrc $ResumeCkpt
    if (Test-Path $ResumeSrcPath) {
        Write-Host "Staging $ResumeSrcPath to $DestDir"
        Copy-Item -Path $ResumeSrcPath -Destination $DestDir -Force
        $ParamsSrc = Join-Path $ResumeSrc "params"
        if (Test-Path $ParamsSrc) {
            Copy-Item -Path $ParamsSrc -Destination $DestDir -Recurse -Force
        }
    } elseif (Test-Path $ResumeTar) {
        Write-Host "Extracting $ResumeCkpt from $ResumeTar"
        $StageDir = Join-Path $env:TEMP "o2stair_ckpts"
        if (Test-Path $StageDir) { Remove-Item -Path $StageDir -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
        
        # Run tar.exe to extract
        tar.exe -xf $ResumeTar -C $StageDir "trained_o2stair/$ResumeCkpt" "trained_o2stair/params" 2>$null
        if ($LASTEXITCODE -ne 0) {
            tar.exe -xf $ResumeTar -C $StageDir "trained_o2stair/$ResumeCkpt"
        }
        
        Copy-Item -Path (Join-Path $StageDir "trained_o2stair\$ResumeCkpt") -Destination $DestDir -Force
        $ParamsStage = Join-Path $StageDir "trained_o2stair\params"
        if (Test-Path $ParamsStage) {
            Copy-Item -Path $ParamsStage -Destination $DestDir -Recurse -Force
        }
    } else {
        Write-Error "ERROR: Could not find checkpoint $ResumeCkpt. Looked in:`n  $ResumeSrcPath`n  $ResumeTar"
    }
} else {
    Write-Host "Checkpoint $ResumeCkpt is already staged at $DestDir"
}

# Auto-open monitor in a new window unless NO_MONITOR is set
if ($env:NO_MONITOR -ne "1") {
    $MonitorBat = Join-Path $PSScriptRoot "watch_training_windows.bat"
    Write-Host "`nAuto-launching live training monitor..."
    Start-Process cmd.exe -ArgumentList "/k", "`"$MonitorBat`""
}

# Configure environment variables for training python invocation
$env:FT_RL_REPO_DIR = $RobotLabDir
$env:FT_RL_ISAACLAB_DIR = $IsaacLabDir
$env:FT_WANDB = "0"  # Local training logs to tensorboard, not W&B

# Parse explorer arguments
$ExploreArgs = $ExploreExtra.Split(" ")

# Execute training script
Write-Host "`n== Executing train_rl.py natively =="
Push-Location $RepoRoot

try {
    # Build list of default arguments for python train_rl.py
    $PythonArgs = @(
        "src/fine_tuning/rl/train_rl.py",
        "--resume",
        "--load-run", $LoadRun,
        "--exptid", $ExptId,
        "--num-envs", $NumEnvs.ToString(),
        "--max-iters", $AddIters.ToString(),
        "--lin-vel-x-max", $LinVelX,
        "--orientation-reward", $Orient,
        "--ascent-reward", $Ascent,
        "--track-lin-vel-weight", $TrackLinVel,
        "--max-init-terrain-level", $MaxLevel.ToString()
    )
    
    # Safely combine default runner args and user-supplied scripts arguments ($args)
    # Important: Place user flags ($args) BEFORE --train-extra, so argparse.REMAINDER does not swallow them!
    $AllArgs = $PythonArgs + $args
    
    if ($ExploreArgs) {
        $AllArgs += "--train-extra"
        $AllArgs += $ExploreArgs
    }
    
    python $AllArgs
} finally {
    Pop-Location
}

Write-Host "`nLocal fine-tune completed."
