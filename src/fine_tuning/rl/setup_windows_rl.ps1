# Set up Native Windows Isaac Sim & IsaacLab local training environment.
# Creates Python 3.11 virtual environment, installs pip dependencies, clones and installs IsaacLab + robot_lab.

$ErrorActionPreference = "Stop"

# Prevent pip from using the cache to avoid permission errors on temp folders
$env:PIP_NO_CACHE_DIR = 1

# Paths
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$VenvDir = if ($env:FT_RL_VENV_DIR) { $env:FT_RL_VENV_DIR } else { Join-Path $env:USERPROFILE ".venv_rl" }

# Set default clone paths to user profiles, but allow environment overrides
$IsaacLabDir = if ($env:FT_RL_ISAACLAB_DIR) { $env:FT_RL_ISAACLAB_DIR } else { Join-Path $env:USERPROFILE "IsaacLab" }
$RobotLabDir = if ($env:FT_RL_REPO_DIR) { $env:FT_RL_REPO_DIR } else { Join-Path $env:USERPROFILE "robot_lab" }

Write-Host "============================================="
Write-Host "  NATIVE WINDOWS ISAAC TRAINING SETUP"
Write-Host "============================================="
Write-Host "Repo Root:   $RepoRoot"
Write-Host "Virtual Env: $VenvDir"
Write-Host "IsaacLab:    $IsaacLabDir"
Write-Host "robot_lab:   $RobotLabDir"
Write-Host "============================================="

# 1. Verify py -3.11 or python 3.11 is present
Write-Host "`n== 1. Verifying Python 3.11 =="
$pyCheck = & py -3.11 -c "import sys; print(sys.version)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $pyCheck) {
    # Check if python.exe is 3.11
    $pyCheck = & python -c "import sys; print(sys.version)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $pyCheck -match "^3\.11") {
        $PyCmd = "python"
    } else {
        Write-Error "ERROR: Python 3.11 is required. Please install official Python 3.11 for Windows (e.g. from python.org)."
    }
} else {
    $PyCmd = "py -3.11"
}
Write-Host "Using Python launcher command: $PyCmd ($($pyCheck.Split([char]10)[0].Trim()))"

# 2. Create Python 3.11 virtual environment if it does not exist
if (-not (Test-Path $VenvDir)) {
    Write-Host "`n== 2. Creating Python 3.11 virtual environment == "
    if ($PyCmd -eq "py -3.11") {
        & py -3.11 -m venv $VenvDir
    } else {
        & python -m venv $VenvDir
    }
} else {
    Write-Host "`n== 2. Virtual environment already exists at $VenvDir (skipping creation) =="
}

# 3. Activate venv and upgrade/install basic packages
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host "`nActivating virtual environment: $ActivateScript"
. $ActivateScript

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# We install wheel first, then setuptools < 81, so flatdict 4.0.1 compiles without pkg_resources errors
Write-Host "Installing wheel, setuptools<81, and flatdict..."
python -m pip install wheel "setuptools<81"
python -m pip install --no-build-isolation flatdict==4.0.1

# 4. Install Isaac Sim pip package
Write-Host "`n== 3. Installing Isaac Sim 5.1.0 (Nvidia Index) =="
Write-Host "Note: This is ~15-20 GB and installs PyTorch with CUDA support automatically. Please stand by..."
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 5. Clone and install IsaacLab
Write-Host "`n== 4. Cloning and Installing IsaacLab v2.3.2 =="
if (-not (Test-Path (Join-Path $IsaacLabDir ".git"))) {
    if ((Test-Path $IsaacLabDir) -and (Get-ChildItem -Path $IsaacLabDir)) {
        Write-Host "IsaacLab directory exists but is not a git repo. Cloning to temp and merging..."
        $TempDir = Join-Path $env:TEMP "isaaclab_clone"
        if (Test-Path $TempDir) { Remove-Item -Path $TempDir -Recurse -Force }
        git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git $TempDir
        Copy-Item -Path "$TempDir\*" -Destination $IsaacLabDir -Recurse -Force
        Remove-Item -Path $TempDir -Recurse -Force
    } else {
        Write-Host "Cloning IsaacLab v2.3.2 to $IsaacLabDir..."
        git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git $IsaacLabDir
    }
} else {
    Write-Host "IsaacLab repository already exists at $IsaacLabDir. Skipping clone."
}

Write-Host "Installing IsaacLab dependencies (including rsl_rl)..."
Push-Location $IsaacLabDir
try {
    # Run the official isaaclab.bat install script inside our virtual env
    # Using CMD since it is a batch file and we want to ensure execution env matches
    cmd.exe /c "isaaclab.bat --install rsl_rl"
} finally {
    Pop-Location
}

Write-Host "Installing IsaacLab core package in editable mode..."
python -m pip install -e "$IsaacLabDir\source\isaaclab"

# 6. Clone and install robot_lab
Write-Host "`n== 5. Cloning and Installing robot_lab v2.3.2 =="
if (-not (Test-Path (Join-Path $RobotLabDir ".git"))) {
    if ((Test-Path $RobotLabDir) -and (Get-ChildItem -Path $RobotLabDir)) {
        Write-Host "robot_lab directory exists but is not a git repo. Cloning to temp and merging..."
        $TempDir = Join-Path $env:TEMP "robot_lab_clone"
        if (Test-Path $TempDir) { Remove-Item -Path $TempDir -Recurse -Force }
        git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git $TempDir
        Copy-Item -Path "$TempDir\*" -Destination $RobotLabDir -Recurse -Force
        Remove-Item -Path $TempDir -Recurse -Force
    } else {
        Write-Host "Cloning robot_lab v2.3.2 to $RobotLabDir..."
        git clone --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git $RobotLabDir
    }
} else {
    Write-Host "robot_lab repository already exists at $RobotLabDir. Skipping clone."
}

Write-Host "Installing robot_lab package in editable mode..."
python -m pip install -e "$RobotLabDir\source\robot_lab"

# 7. Install extra python requirements
Write-Host "`n== 6. Installing requirements_rl.txt == "
python -m pip install -r "$PSScriptRoot\requirements_rl.txt"

# 8. Run Preflight verification
Write-Host "`n== 7. Running Preflight Verification == "
$env:FT_RL_REPO_DIR = $RobotLabDir
python "$PSScriptRoot\preflight_rl.py"

Write-Host "`n============================================="
Write-Host "  SETUP COMPLETE!"
Write-Host "============================================="
Write-Host "Trained checkpoints and logs will be saved to: $RobotLabDir\logs"
Write-Host "You can now run train_windows.bat to launch native local training."
Write-Host "============================================="
