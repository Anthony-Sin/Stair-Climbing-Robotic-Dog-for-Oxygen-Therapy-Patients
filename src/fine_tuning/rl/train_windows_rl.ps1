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
# NOTE (2026-07-11): falls now TERMINATE the episode via train_rl.py's default
# --fall-limit-angle-deg 60 -> terminations.fell_over = bad_orientation(60 deg) (see
# CLAUDE.md 8.11). Every run before 2026-07-11 had NO fall termination (robot_lab nulls
# illegal_contact), so their "100% time_out" telemetry included invisible fallen-and-
# flailing episode time, not just cautious/successful ones.
$Orient = if ($env:FT_RL_ORIENT_REWARD) { $env:FT_RL_ORIENT_REWARD } else { "-0.5" }
$Ascent = if ($env:FT_RL_ASCENT_REWARD) { $env:FT_RL_ASCENT_REWARD } else { "3.0" }
# track_lin_vel 1.5 -> 2.5: run 2026-07-10_21-39-53 still let "stand still" earn ~85% of the
# exp-kernel tracking reward at 1.5; raising it alone would make that worse, so it only moves
# together with the new lin-vel-x-min floor below (which removes the zero-speed freebie).
$TrackLinVel = if ($env:FT_RL_TRACK_LINVEL_W) { $env:FT_RL_TRACK_LINVEL_W } else { "2.5" }
# max_init_terrain_level 8 -> 6: level 8 dropped the resumed level-3.7 policy onto mean
# level ~5, past what the XY-distance-only curriculum could hold -> mass demotion. 6 is a
# gentler cliff (paired with the ascent-aware curriculum fix in config_patch.py).
$MaxLevel = if ($env:FT_RL_MAX_INIT_TERRAIN_LEVEL) { $env:FT_RL_MAX_INIT_TERRAIN_LEVEL } else { 6 }
# lin_vel_x ceiling 0.4 -> 0.8: paired with the new floor (LinVelXMin) so the command range
# no longer straddles zero -- "stand still" stops being a legal, near-optimal command.
$LinVelX = if ($env:FT_RL_LINVELX_MAX) { $env:FT_RL_LINVELX_MAX } else { "0.8" }
# lin_vel_x floor: was implicitly 0.0 (commands.base_velocity.ranges.lin_vel_x = (0.0, max)),
# which let a motionless policy earn ~85% of track_lin_vel_xy_exp (exp kernel, std=0.5) --
# root cause (3) of the run 2026-07-10_21-39-53 collapse. Force it to actually move.
$LinVelXMin = if ($env:FT_RL_LINVELX_MIN) { $env:FT_RL_LINVELX_MIN } else { "0.2" }
# upward weight 1.0 -> 0.25: stock `upward` = square(1 - proj_grav_z) pays ~4/step for just
# standing upright at level 0 -- 53% of the positive reward budget in run 2026-07-10_21-39-53,
# earnable without ever moving, and it out-earned ascent_rate ~32:1 (root cause (1)).
$UpwardWeight = if ($env:FT_RL_UPWARD_WEIGHT) { $env:FT_RL_UPWARD_WEIGHT } else { "0.25" }
# lin_vel_z weight -2.0 -> -1.0: stock punishes ANY vertical velocity, including the vz a
# genuinely climbing policy needs to produce; ease it so climbing isn't fighting its own
# penalty term as hard (paired with rebalancing ascent vs. upward/tracking above).
$LinVelZWeight = if ($env:FT_RL_LINVELZ_WEIGHT) { $env:FT_RL_LINVELZ_WEIGHT } else { "-1.0" }
# tall_step_min 0.10: the "pyramid_stairs_tall" sub-terrain ships a FIXED (0.2, 0.2) step
# height even at curriculum difficulty 0 -- an unwinnable pit for a policy that can't yet
# climb 0.15 m, so 20% of envs spawn permanently trapped at the pit bottom, dragging
# tracking averages and gradient quality at level 0 (observed run 2026-07-11_01-53: stuck
# at terrain level 0 with flat tracking reward). Lowering the tile's OWN easy edge gives it
# a difficulty ramp like the other three sub-terrains instead of a fixed hard floor.
$TallStepMin = if ($env:FT_RL_TALL_STEP_MIN) { $env:FT_RL_TALL_STEP_MIN } else { "0.10" }
# tall_start_prop 0.2 -> 0.10: halve the hard-tile share of the terrain mix while the
# ramped tall tile (above) is re-learned from scratch at the easy end; full exposure to the
# hard edge returns via the difficulty ramp/curriculum rather than a large fixed-hard slice.
$TallProp = if ($env:FT_RL_TALL_START_PROP) { $env:FT_RL_TALL_START_PROP } else { "0.10" }
# payload_mass_scale: stage-1 payload curriculum (see config_patch.py StairPatchParams.payload_mass_scale
# docstring) -- scales ONLY the added-mass DR band, not the CoM offset, so a fresh/staged policy can
# learn to climb before facing the full ~1.6-3.5 kg tank load. 1.0 = full mass (default, no staging).
$PayloadMassScale = if ($env:FT_RL_PAYLOAD_MASS_SCALE) { $env:FT_RL_PAYLOAD_MASS_SCALE } else { "1.0" }
# rel_standing_envs 0.02 -> 0.12: stage-4 halt fix (see config_patch.py StairPatchParams.rel_standing_envs
# docstring) -- stage-3 (model_3200) lean-creeps 0.146 m/s mean / 0.367 m/s p95 at commanded vx=0
# mid-stairs and toppled after a ~3 min near-crest hold (sim run 2026-07-11_161710_451); robot_lab/
# IsaacLab only sample a TRUE zero command on 2% of envs, so raise 6x to give halting real signal.
$RelStandingEnvs = if ($env:FT_RL_REL_STANDING_ENVS) { $env:FT_RL_REL_STANDING_ENVS } else { "0.12" }
# pitch_dip_hinge_rad / pitch_dip_weight / trunk_thigh_contact_weight: stage-5 mount-softening fix
# (see config_patch.py StairPatchParams docstrings) -- the deployed stage-4 policy strikes each riser
# nose-first (press-stall-push mounts measured ~22-34 deg pitch dips vs a normal ~8-12 deg climb lean).
# pitch_dip_hinge (0.26 rad ~15 deg) is compared against THIS training env's OWN verified pitch sign
# (positive == nose-down; see _reward_pitch_dip_hinge in config_patch.py -- NOT the deployed-sim sign),
# so the normal lean costs zero and only a genuine deep mount is penalised. trunk_thigh_contact reuses
# robot_lab's own undesired_contacts function at a smaller weight, scoped to just trunk+thigh bodies.
# Both are shaping only (no termination added, CLAUDE.md 8.11).
$PitchDipHingeRad = if ($env:FT_RL_PITCH_DIP_HINGE_RAD) { $env:FT_RL_PITCH_DIP_HINGE_RAD } else { "0.26" }
$PitchDipWeight = if ($env:FT_RL_PITCH_DIP_WEIGHT) { $env:FT_RL_PITCH_DIP_WEIGHT } else { "-1.0" }
$TrunkThighContactWeight = if ($env:FT_RL_TRUNK_THIGH_CONTACT_WEIGHT) { $env:FT_RL_TRUNK_THIGH_CONTACT_WEIGHT } else { "-0.25" }
# entropy_coef 0.008 -> 0.005: 0.008 stabilized noise_std around ~0.7, but sigma decay was
# too slow for tracking precision to recover -- run 2026-07-11_01-53 held track reward flat
# (+0.05/500 iters) with terrain level pinned at 0 while sigma~0.7 execution noise itself
# capped precision. 0.005 is safe to try now ONLY because the reward landscape that made
# 0.002 dangerous is gone: 0.002 froze exploration under the OLD landscape where "stand
# still" was a legal, near-optimal command (no lin_vel_x floor) and `upward` paid ~4/step
# for merely standing upright (53% of the positive reward budget) -- both are now
# structurally removed (LinVelXMin floor above; UpwardWeight trimmed to 0.25), so a
# lower-noise policy is no longer rewarded for freezing in place.
# 0.002 collapsed exploration (run 2026-07-10_21-39-53: noise_std 1.01 -> 0.33, policy froze
# into "stand still"); 0.02 EXPLODED it (run 2026-07-10_23-41-18: noise_std 1.0 -> 1.93,
# entropy bonus out-paid the rebalanced/trimmed task rewards, policy became a noise-ball --
# error_vel_xy 1.04, reward -18.6 from action_rate/joint_acc thrash, stuck at level 0).
# Healthy band to watch on the monitor: noise_std settling ~0.5-1.2, neither trending to
# extremes. init_noise_std only matters on fresh (non-resume) runs; resume loads std from
# the checkpoint.
$ExploreExtra = if ($env:FT_RL_EXPLORE_EXTRA) { $env:FT_RL_EXPLORE_EXTRA } else { "agent.algorithm.entropy_coef=0.005 agent.policy.init_noise_std=1.2" }

Write-Host "============================================="
Write-Host "  NATIVE WINDOWS ISAAC TRAINING RUNNER"
Write-Host "============================================="
Write-Host "repo=$RobotLabDir  exptid=$ExptId  load_run=$LoadRun  num_envs=$NumEnvs  +iters=$AddIters"
Write-Host "resume_ckpt=$ResumeCkpt  orient=$Orient  ascent=$Ascent  track_linvel=$TrackLinVel  max_level=$MaxLevel  linvel_x=$LinVelXMin..$LinVelX"
Write-Host "upward=$UpwardWeight  linvel_z=$LinVelZWeight  tall_step_min=$TallStepMin  tall_prop=$TallProp  explore=$ExploreExtra"
Write-Host "payload_mass_scale=$PayloadMassScale  rel_standing_envs=$RelStandingEnvs"
Write-Host "pitch_dip_hinge_rad=$PitchDipHingeRad  pitch_dip_weight=$PitchDipWeight  trunk_thigh_contact_weight=$TrunkThighContactWeight"
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
        "--lin-vel-x-min", $LinVelXMin,
        "--orientation-reward", $Orient,
        "--ascent-reward", $Ascent,
        "--track-lin-vel-weight", $TrackLinVel,
        "--max-init-terrain-level", $MaxLevel.ToString(),
        "--upward-weight", $UpwardWeight,
        "--lin-vel-z-weight", $LinVelZWeight,
        "--tall-step-min", $TallStepMin,
        "--tall-start-prop", $TallProp,
        "--payload-mass-scale", $PayloadMassScale,
        "--rel-standing-envs", $RelStandingEnvs,
        "--pitch-dip-hinge-rad", $PitchDipHingeRad,
        "--pitch-dip-weight", $PitchDipWeight,
        "--trunk-thigh-contact-weight", $TrunkThighContactWeight
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
