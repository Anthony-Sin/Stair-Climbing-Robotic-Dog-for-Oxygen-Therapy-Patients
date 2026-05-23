# 🐕 Stair-Climbing Robotic Dog for Oxygen Therapy Patients
### Complete Simulation Dev Environment: Isaac Sim · WSL2 · Docker · VS Code · Vision Pipeline

> **What this guide does:** Gets Isaac Sim running on **Windows**, a GPU-enabled Docker container running in **WSL2 Ubuntu**, VS Code wired up to both, and the full person-following simulation working end-to-end.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Host Machine Requirements](#2-host-machine-requirements)
3. [Part 1 — VS Code Setup](#3-part-1--vs-code-setup)
4. [Part 2 — Python Setup on Windows](#4-part-2--python-setup-on-windows)
5. [Part 3 — WSL2 + Ubuntu Setup](#5-part-3--wsl2--ubuntu-setup)
6. [Part 4 — Docker Desktop Setup](#6-part-4--docker-desktop-setup)
7. [Part 5 — NVIDIA Isaac Sim (Open Source) on Windows](#7-part-5--nvidia-isaac-sim-open-source-on-windows)
8. [Part 6 — Getting the Project Files](#8-part-6--getting-the-project-files)
9. [Part 7 — Building the Docker Images](#9-part-7--building-the-docker-images)
10. [Part 8 — Preparing AI Models (TensorRT Engines)](#10-part-8--preparing-ai-models-tensorrt-engines)
11. [Part 9 — Network Configuration (The Hard Part)](#11-part-9--network-configuration-the-hard-part)
12. [Part 10 — Running the Full Simulation](#12-part-10--running-the-full-simulation)
13. [Part 11 — ROS2 Sidecar (MPPI + Nav2)](#13-part-11--ros2-sidecar-mppi--nav2)
14. [Part 12 — LiDAR Support](#14-part-12--lidar-support)
15. [Troubleshooting](#15-troubleshooting)
16. [Project Structure Reference](#16-project-structure-reference)
17. [Quick Reference Card](#17-quick-reference-card)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WINDOWS HOST                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  NVIDIA Isaac Sim 5.1  (isaac_env.py)                      │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐  │   │
│  │  │  Go2 Robot   │   │  Person      │   │  RTX Camera   │  │   │
│  │  │  (PhysX sim) │   │  (patrol)    │   │  (RGB+Depth)  │  │   │
│  │  └──────────────┘   └──────────────┘   └───────────────┘  │   │
│  │                                               │             │   │
│  │  Sends JPEG+zlib compressed frames ──→ UDP :55002          │   │
│  │  Receives velocity commands       ←── UDP :55001           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↕  UDP over virtual NIC                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Docker Desktop  (WSL2 backend)                            │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  go2-pose-x86 container  (Ubuntu 22.04 + CUDA 12)  │   │   │
│  │  │                                                     │   │   │
│  │  │  SimCameraCapture ← UDP :55002                      │   │   │
│  │  │       ↓                                             │   │   │
│  │  │  YoloPoseInference (TensorRT FP16)                  │   │   │
│  │  │       ↓                                             │   │   │
│  │  │  ByteTrack multi-object tracker                     │   │   │
│  │  │       ↓                                             │   │   │
│  │  │  ReIDManager (OSNet-AIN TensorRT)                   │   │   │
│  │  │       ↓                                             │   │   │
│  │  │  PersonFollower (PID or MPPI target export)         │   │   │
│  │  │       ↓                                             │   │   │
│  │  │  SimRobotController → UDP :55001                    │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Frame rate budget (what to expect):**
- Isaac Sim → Docker: ~30 frames/sec (limited by UDP + JPEG compression)
- TRT inference: ~60–120 fps (RTX 4070 Laptop)
- Effective follow loop: ~25–40 fps

---

## 2. Host Machine Requirements

| Component | Minimum | Recommended (what was tested) |
|-----------|---------|-------------------------------|
| OS | Windows 10 21H2 (build 19044) | Windows 11 23H2+ |
| CPU | Intel 10th gen / AMD Ryzen 4000 | Intel 13th Gen i7-13620H ✓ |
| RAM | 32 GB | 16 GB works but tight — 32 GB preferred |
| GPU | NVIDIA RTX 3070 (8 GB VRAM) | RTX 4070 Laptop GPU (8 GB) ✓ |
| NVIDIA Driver | 525.85+ | 561.17 ✓ |
| CUDA Toolkit | 12.x (for Docker) | 12.6 ✓ |
| Storage | 150 GB free SSD | 200 GB+ (Isaac Sim alone is ~50 GB) |
| Internet | Required | Needed for Nucleus asset downloads |

> ⚠️ Isaac Sim **does not work** with AMD or Intel GPUs for rendering/physics acceleration. You must have an NVIDIA GPU on the Windows host.
>
> ⚠️ The CUDA Toolkit version inside Docker (12.6) must be **≤** your Windows NVIDIA driver's max supported CUDA version. Driver 561.17 supports up to CUDA 12.6. Check yours at: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

---

## 3. Part 1 — VS Code Setup

VS Code is the recommended editor for this project. You'll use it to edit Python on both Windows (for Isaac Sim scripts) and inside WSL/Docker (for the vision pipeline).

### 3.1 Install VS Code on Windows

Download from: https://code.visualstudio.com/download

During installation check:
- ✅ **Add "Open with Code" action to Windows Explorer file context menu**
- ✅ **Add "Open with Code" action to Windows Explorer directory context menu**
- ✅ **Add to PATH** (important — lets you type `code .` in terminals)
- ✅ **Register Code as an editor for supported file types**

### 3.2 Essential Extensions to Install

Open VS Code, press `Ctrl+Shift+X` to open the Extensions panel, and install all of these:

**Core extensions:**

| Extension | Publisher | Why you need it |
|-----------|-----------|-----------------|
| **Python** | Microsoft | Python language support, IntelliSense, linting |
| **Pylance** | Microsoft | Fast Python type checking and autocomplete |
| **Remote - WSL** | Microsoft | Edit files inside WSL from VS Code on Windows |
| **Remote - Containers** | Microsoft | Attach VS Code to a running Docker container |
| **Docker** | Microsoft | Docker file syntax, container management sidebar |
| **Remote - SSH** | Microsoft | Connect to remote machines if needed |

**Quality of life extensions:**

| Extension | Publisher | Why you need it |
|-----------|-----------|-----------------|
| **GitLens** | GitKraken | Better git blame, history, diff |
| **YAML** | Red Hat | YAML syntax for ROS2 launch files and nav2 configs |
| **XML** | Red Hat | package.xml syntax highlighting |
| **C/C++** | Microsoft | For the C++ LiDAR filter node |
| **CMake** | twxs | CMake syntax highlighting |
| **CMake Tools** | Microsoft | Build C++ packages from VS Code |
| **Rainbow CSV** | mechatroner | Makes CSV debug trace files readable |
| **Better TOML** | bungcip | If any config files use TOML |
| **Error Lens** | Alexander | Shows errors inline as you type |
| **Path Intellisense** | Christian Kohler | Autocomplete file paths in code |

Install all at once by pasting into the terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-vscode-remote.remote-wsl
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension eamodio.gitlens
code --install-extension redhat.vscode-yaml
code --install-extension redhat.vscode-xml
code --install-extension ms-vscode.cpptools
code --install-extension twxs.cmake
code --install-extension ms-vscode.cmake-tools
code --install-extension mechatroner.rainbow-csv
code --install-extension usernamehw.errorlens
code --install-extension christian-kohler.path-intellisense
```

### 3.3 Open the Project in VS Code

```cmd
:: In Windows Command Prompt or PowerShell:
code "C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients"
```

### 3.4 Connect VS Code to WSL

This lets you edit files inside WSL with full IntelliSense:

1. Press `Ctrl+Shift+P` → type **"WSL: Connect to WSL"** → Enter
2. VS Code reopens connected to Ubuntu
3. Open the project folder: **File → Open Folder** → `/home/yourname/robot`
4. The bottom-left corner shows `WSL: Ubuntu`

Now when you open a `.py` file, VS Code uses the Python inside WSL, not Windows.

### 3.5 Connect VS Code to a Running Docker Container

When the vision container is running, you can edit and debug code inside it:

1. Start the container (see Section 12)
2. In VS Code, click the **Docker icon** in the left sidebar
3. Under **Containers**, right-click your running container
4. Click **Attach Visual Studio Code**
5. A new VS Code window opens **inside the container**

This is extremely useful for debugging TensorRT issues because you have full access to the container's Python environment and can set breakpoints.

### 3.6 VS Code Settings for This Project

Create `.vscode/settings.json` in the project root:

```json
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": false,
    "files.associations": {
        "*.launch.py": "python",
        "*.yaml": "yaml",
        "package.xml": "xml"
    },
    "python.analysis.extraPaths": [
        "./src"
    ],
    "terminal.integrated.defaultProfile.windows": "Command Prompt",
    "remote.WSL.fileWatcher.polling": true
}
```

Create `.vscode/launch.json` to run Isaac Sim scripts with F5:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Isaac Sim: isaac_env.py (with person)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/isaac/isaac_env.py",
            "args": [
                "--person-move",
                "--frame-host", "192.168.1.91"
            ],
            "console": "integratedTerminal",
            "pythonPath": "C:\\isaacsim\\IsaacSim-main\\_build\\windows-x86_64\\release\\python.bat"
        },
        {
            "name": "Vision Pipeline: main.py (sim mode)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/main.py",
            "args": [
                "--sim",
                "--follow",
                "--follow-backend", "pid",
                "--debug",
                "--trt-engine", "/models/yolo11n-pose-fp16.trt",
                "--osnet-trt-engine", "/models/osnet_ain_x1_0.trt"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

---

## 4. Part 2 — Python Setup on Windows

Isaac Sim ships with its **own Python interpreter** bundled inside the installation. You do **not** use system Python or a conda env to run Isaac Sim scripts.

### 4.1 Understanding Which Python to Use

| Script | Where it runs | Python to use |
|--------|--------------|---------------|
| `isaac/isaac_env.py` | Windows, inside Isaac Sim | **Isaac Sim's bundled python.bat** |
| `isaac/go2_usd_setup.py` | Windows, inside Isaac Sim | **Isaac Sim's bundled python.bat** |
| `src/main.py` | Ubuntu WSL / Docker | **Python 3.10 inside Docker container** |
| `ros2_ws/...` | Ubuntu WSL / Docker | **Python 3.10 inside ROS2 container** |

**Never** use system Python (`python3` on Windows, or a conda env) to run Isaac Sim scripts. The Isaac Sim Python environment has all the `omni.*`, `isaacsim.*`, `pxr.*` packages pre-installed and they won't be available elsewhere.

### 4.2 Isaac Sim's Python Interpreter Location

```
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat
```

This `.bat` file sets up all the environment variables Isaac Sim needs and then calls the real Python binary at:
```
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\kit\python\python.exe
```

### 4.3 Register Isaac Sim Python in VS Code

To get IntelliSense for Isaac Sim packages in VS Code, add the Isaac Sim Python as an interpreter:

1. Press `Ctrl+Shift+P` → **"Python: Select Interpreter"**
2. Click **"Enter interpreter path..."**
3. Paste: `C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\kit\python\python.exe`

You'll now get autocomplete for `from omni.isaac.core import World` etc.

### 4.4 Python Version Requirements

| Environment | Python Version | Notes |
|-------------|---------------|-------|
| Isaac Sim 5.1 bundled | **3.10.x** | Don't change this |
| Docker container (vision) | **3.10.x** | Set in Dockerfile |
| ROS2 Humble sidecar | **3.10.x** | ROS2 Humble ships with 3.10 |

> ⚠️ Do **not** try to run Isaac Sim scripts with Python 3.11 or 3.12. The USD/Omniverse bindings are compiled for a specific Python version and will segfault or throw ImportErrors.

### 4.5 Install Git on Windows (if not already installed)

```powershell
# Option A: winget
winget install Git.Git

# Option B: Download from https://git-scm.com/download/win
# During install:
# ✅ Git from the command line and also from 3rd-party software
# ✅ Use Visual Studio Code as Git's default editor
# ✅ Override the default branch name for new repos: main
# ✅ Git Credential Manager
```

Verify:
```cmd
git --version
# git version 2.45.x.windows.1
```

---

## 5. Part 3 — WSL2 + Ubuntu Setup

WSL2 (Windows Subsystem for Linux version 2) runs a real Linux kernel inside a lightweight VM on Windows. Docker Desktop uses it as its backend, meaning your Docker containers get native Linux performance.

### 5.1 Check if WSL2 is Already Installed

```powershell
# Open PowerShell as Administrator
wsl --status
```

If you see `Default Version: 2` you're good. If WSL isn't installed at all, continue below.

### 5.2 Enable Required Windows Features

```powershell
# Open PowerShell as Administrator (right-click PowerShell → "Run as administrator")

# Enable WSL subsystem
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform (required for WSL2)
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**Restart your PC.** Both features need a reboot to activate.

After reboot, back in Administrator PowerShell:

```powershell
# Set WSL2 as the default version
wsl --set-default-version 2

# Update the WSL kernel to the latest
wsl --update
```

### 5.3 Install Ubuntu

```powershell
# List available distros
wsl --list --online

# Install Ubuntu 22.04 (recommended — matches ROS2 Humble)
wsl --install -d Ubuntu-22.04

# OR install latest Ubuntu (24.04 as of 2025)
wsl --install -d Ubuntu
```

After installation, Ubuntu launches and asks you to create a username and password. **This password is important** — you'll use it with `sudo` constantly. Pick something you'll remember.

```powershell
# Verify it's WSL version 2 (not 1)
wsl -l -v

# Expected output:
#   NAME            STATE           VERSION
# * Ubuntu-22.04    Running         2
```

If it shows `VERSION 1`:
```powershell
wsl --set-version Ubuntu-22.04 2
# This conversion takes a few minutes
```

### 5.4 First-Time Ubuntu Setup

Open the Ubuntu app (search "Ubuntu" in Start menu) and run:

```bash
# Update all packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    curl \
    git \
    wget \
    build-essential \
    python3-pip \
    python3-dev \
    net-tools \
    iputils-ping \
    dnsutils \
    nano \
    vim \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Verify git is installed
git --version
```

### 5.5 Configure Git in WSL

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.editor "code --wait"
git config --global init.defaultBranch main
```

### 5.6 Understand WSL2's File Systems

WSL2 has two separate file systems and mixing them has big performance implications:

| Location | Description | Speed |
|----------|-------------|-------|
| `/home/yourname/` | WSL2's native Linux filesystem | ⚡ Fast |
| `/mnt/c/Users/...` | Your Windows C: drive mounted in WSL | 🐢 Slow (10x slower for small files) |

**Keep Docker build contexts and model files under `/home/`** (the WSL native filesystem), not under `/mnt/c/`. Docker builds over `/mnt/c/` paths can be 10× slower.

```bash
# Create a fast working directory in WSL native filesystem
mkdir -p ~/robot
mkdir -p ~/robot/models

# Create a symlink to the project on Windows drive for easy access
ln -s "/mnt/c/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients" ~/robot/project
```

### 5.7 WSL2 Resource Limits

By default WSL2 can use up to **50% of your RAM and all CPU cores**. With Isaac Sim also running, you might run out of memory. Cap it:

On Windows, create/edit `C:\Users\antho\.wslconfig`:

```ini
[wsl2]
# Max memory WSL2 can use (leaves rest for Windows + Isaac Sim)
memory=8GB

# Max CPU cores WSL2 can use
processors=8

# Swap space
swap=4GB

# Disable page reporting (reduces stuttering when memory is reclaimed)
pageReporting=false

# Enable localhost forwarding (important for port access)
localhostForwarding=true
```

After editing, restart WSL:
```powershell
# In PowerShell:
wsl --shutdown
# Wait 5 seconds, then reopen Ubuntu
```

### 5.8 Find and Note Your WSL2 IP

```bash
# Method 1
hostname -I | awk '{print $1}'

# Method 2
ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1

# Method 3 — also shows Windows host IP
cat /etc/resolv.conf
# The "nameserver" line is your Windows host's IP from WSL's perspective
```

**Add this to `~/.bashrc`** so you always see it on terminal open:

```bash
echo '' >> ~/.bashrc
echo '# Show WSL2 IP on terminal open (needed for Isaac Sim --frame-host)' >> ~/.bashrc
echo 'echo "🤖 WSL2 IP for Isaac Sim --frame-host: $(hostname -I | awk '"'"'{print $1}'"'"')"' >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Part 4 — Docker Desktop Setup

### 6.1 Install Docker Desktop

Download from: https://www.docker.com/products/docker-desktop/

**During installation:**
- ✅ **Use WSL 2 instead of Hyper-V** ← this is the critical checkbox. If you install with Hyper-V instead, Docker won't see your GPU and you'll waste hours debugging.
- ✅ **Add shortcut to desktop**
- ✅ **Start Docker Desktop when you log in** (optional but convenient)

After installation, restart your machine when prompted.

### 6.2 Configure Docker Desktop WSL2 Integration

Open Docker Desktop after it starts, then:

1. Click the **gear icon** (Settings) in the top right
2. **General** tab:
   - ✅ **Use the WSL 2 based engine** must be ON
   - ✅ **Send usage statistics** (optional, up to you)
3. **Resources → WSL Integration**:
   - ✅ **Enable integration with my default WSL distro**
   - Toggle ON for **Ubuntu-22.04** (or whichever you installed)
4. Click **Apply & Restart**

### 6.3 Increase Docker's Resource Limits

Still in Docker Desktop Settings:

1. **Resources → Advanced**:
   - **CPUs:** 6–8 (leave 2–4 for Windows + Isaac Sim)
   - **Memory:** 6–8 GB (leave the rest for Isaac Sim which needs ~8 GB)
   - **Disk image size:** 100+ GB (Docker images for this project are large)
2. Click **Apply & Restart**

> 💡 If you set `.wslconfig` memory to `8GB` and Docker to `8GB`, that's 16 GB total — more than your machine has. WSL2 and Docker share the WSL2 memory budget, so Docker's limit in the UI here is a soft hint. The hard limit is set in `.wslconfig`.

### 6.4 Install NVIDIA Container Toolkit

This is what allows Docker containers to access your GPU. Without it, `--gpus all` does nothing.

Open **Ubuntu WSL** and run:

```bash
# Step 1: Add NVIDIA's package signing key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Step 2: Add NVIDIA's apt repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Step 3: Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Step 4: Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Step 5: Verify the config was written
cat /etc/docker/daemon.json
# Should show something like:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       ...
#     }
#   }
# }
```

After this, **restart Docker Desktop** from the Windows system tray (right-click the whale icon → Restart).

### 6.5 Add Your User to the Docker Group

```bash
# Without this, every docker command needs sudo
sudo usermod -aG docker $USER

# Apply the group change immediately (or restart WSL)
newgrp docker

# Verify you can run docker without sudo
docker run hello-world
```

Expected output ends with: `Hello from Docker! This message shows that your installation appears to be working correctly.`

### 6.6 Verify GPU Access in Docker

```bash
# This is the definitive test — if this works, everything is wired up correctly
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 561.17      Driver Version: 561.17      CUDA Version: 12.6   |
|-------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M| Bus-Id       Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4070 ...    On |  00000000:01:00.0 Off |                  N/A |
```

If you get `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]`:
```bash
# Try re-running the configure step
sudo nvidia-ctk runtime configure --runtime=docker
# Then restart Docker Desktop from Windows system tray
```

### 6.7 Configure Docker Logging (Prevents Disk Bloat)

Without log rotation, Docker container logs grow forever. Add this to Docker's daemon config:

```bash
# Edit Docker daemon config
sudo nano /etc/docker/daemon.json
```

Make it look like:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "50m",
        "max-file": "3"
    },
    "default-runtime": "runc"
}
```

Save (`Ctrl+O`, `Enter`, `Ctrl+X`) then restart Docker Desktop.

### 6.8 Understanding Docker Networking in WSL2

This is important for understanding the port setup:

```
Windows Host (192.168.1.100 on your LAN)
│
├── Docker Desktop manages a WSL2 VM
│   └── Ubuntu WSL (gets an IP like 172.19.x.x or 192.168.1.91)
│       └── Docker daemon
│           └── go2-pose-x86 container
│               ├── eth0 (container internal: 172.17.0.x)
│               └── Port -p 55002:55002/udp maps container:55002 → host:55002
│
└── vEthernet (WSL) adapter on Windows
    └── Windows can reach WSL at the WSL IP (e.g., 192.168.1.91)
```

When you run with `-p 55002:55002/udp`:
- Isaac Sim (Windows) sends UDP to WSL IP `192.168.1.91:55002`
- Docker Desktop's port mapping forwards that to the container's port 55002
- `SimCameraCapture` inside the container receives it

### 6.9 Common Docker Port Flags Explained

```bash
# -p HOST_PORT:CONTAINER_PORT/PROTOCOL
-p 55002:55002/udp    # Map UDP port 55002 on WSL host → container

# --network host
# Skips Docker's network isolation entirely — container shares Windows host networking
# Useful for real hardware but CAN cause issues with sim (don't use it for sim)

# --privileged
# Gives container access to all host devices (/dev/*)
# Required for real hardware, not needed for sim

# -v SOURCE:DEST
-v ~/robot/models:/models         # Mount models directory into container
-v ~/robot/src:/workspace/src     # Mount source code (live editing!)
```

### 6.10 Useful Docker Commands for Daily Use

```bash
# List running containers
docker ps

# List all containers including stopped
docker ps -a

# Stop a specific container
docker stop <container-name-or-id>

# Stop ALL running containers
docker stop $(docker ps -q)

# Remove all stopped containers
docker container prune -f

# View container logs (live tail)
docker logs -f <container-name>

# Open a bash shell in a running container
docker exec -it <container-name> bash

# See resource usage (CPU, RAM, GPU is via nvidia-smi inside container)
docker stats

# List images
docker images

# Remove an image
docker rmi go2-pose-x86:latest

# Clean up everything not in use (reclaim disk space)
docker system prune -af --volumes

# Check how much disk Docker is using
docker system df
```

---

## 7. Part 5 — NVIDIA Isaac Sim (Open Source) on Windows

### 7.1 About Isaac Sim Open Source

NVIDIA open-sourced Isaac Sim in 2024. The full source code, build system, and prebuilt binaries are available at:

- **GitHub (source):** https://github.com/isaac-sim/IsaacSim
- **Releases (prebuilt):** https://github.com/isaac-sim/IsaacSim/releases
- **Documentation:** https://docs.isaacsim.omniverse.nvidia.com/latest/
- **Python API reference:** https://docs.isaacsim.omniverse.nvidia.com/latest/python_api.html

Isaac Sim is built on top of **NVIDIA Omniverse Kit** — a modular platform for building simulation apps. It uses:
- **USD** (Universal Scene Description) for 3D scene files
- **PhysX 5** for rigid body, articulation, and contact physics
- **RTX rendering** via NVIDIA's real-time path tracer
- **Python 3.10** for scripting and automation

This project uses **Isaac Sim 5.1** (released early 2025). The logs confirm:
```
[isaacsim.exp.base-5.1.0] startup
[isaacsim.simulation_app-2.12.2] startup
```

### 7.2 Installation — Option A: Omniverse Launcher (Easiest)

1. Download Omniverse Launcher: https://www.nvidia.com/en-us/omniverse/download/
2. Create a free NVIDIA account if you don't have one
3. Install and launch the Omniverse Launcher
4. Go to the **Exchange** tab
5. Search for **"Isaac Sim"**
6. Click **Install** and choose your install directory
   - Recommended: `C:\isaacsim\` (no spaces, short path)
   - **Avoid** `C:\Program Files\...` — permission issues
7. Wait for the ~50 GB download

After install, launch once from the Launcher to verify it works, then close it.

### 7.3 Installation — Option B: Direct GitHub Release (What the Project Uses)

```powershell
# In PowerShell:

# Create install directory
mkdir C:\isaacsim

# Go to: https://github.com/isaac-sim/IsaacSim/releases
# Download the Windows zip for 5.1 (file will be ~50 GB)
# Extract to C:\isaacsim\IsaacSim-main\
```

After extraction, your path should be:
```
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat
```

### 7.4 Installation — Option C: Build from Source

Only recommended if you need to modify Isaac Sim internals:

```powershell
# Prerequisites: Visual Studio 2022 Community, CMake 3.24+, Git LFS
git clone https://github.com/isaac-sim/IsaacSim.git C:\isaacsim\IsaacSim-main
cd C:\isaacsim\IsaacSim-main

# This pulls prebuilt binaries (~40 GB) — uses Git LFS
.\pull_binaries.bat

# Build the full app
.\build.bat
```

### 7.5 First Launch Verification

```cmd
:: Open Command Prompt (not PowerShell)
:: Run a minimal test — just print the Python version
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat -c "import isaacsim; print('Isaac Sim Python OK')"
```

If you get a crash or missing DLL error, install:
- **Visual C++ Redistributable 2022:** https://aka.ms/vs/17/release/vc_redist.x64.exe
- **DirectX Runtime:** https://www.microsoft.com/en-us/download/details.aspx?id=35

### 7.6 Nucleus Content Server

Isaac Sim downloads robot/environment 3D assets from **Nucleus** — NVIDIA's cloud asset server. For this project, assets are fetched directly from AWS S3:

```
Go2 robot USD:
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/Go2/go2.usd

Human character USD:
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd
```

**On first run, Isaac Sim will download these assets.** This can take 2–5 minutes depending on your internet. They're cached locally after the first download.

If you're on a slow/restricted network, you can also set up a local Nucleus server — see: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_nucleus.html

### 7.7 How to Run Isaac Sim Scripts

Isaac Sim scripts **must** be run through the bundled `python.bat`, not system Python:

```cmd
:: Syntax:
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat <script.py> [args]

:: Example: Run isaac_env.py
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat ^
    C:\Users\antho\Downloads\...\isaac\isaac_env.py ^
    --person-move ^
    --frame-host 192.168.1.91

:: Note: ^ is the line continuation character in Windows CMD (like \ in bash)
```

> ⚠️ **Do NOT run Isaac Sim scripts from PowerShell if you have execution policy issues.** Use **Command Prompt** (cmd.exe). The `.bat` file works more reliably in cmd.

### 7.8 Isaac Sim on First Run Behavior

Expect Isaac Sim to:
1. Take **30–90 seconds** to start (shader compilation on first run is slow)
2. Show a **full 3D viewport** with the Go2 robot and a person target
3. Print verbose startup logs to the console (mostly deprecation warnings — these are harmless)
4. Download the human character USD from S3 (~50 MB, one-time)
5. Begin streaming frames to your Docker container

The meshes on the Go2 model will show **UV/texture warnings** in the logs — these are harmless and are bugs in the upstream Unitree USD asset, not your code.

---

## 8. Part 6 — Getting the Project Files

### 8.1 Clone on Windows

```powershell
cd C:\Users\antho\Downloads
git clone <your-repo-url> Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients
cd Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients
```

### 8.2 Access from WSL

The Windows filesystem is mounted in WSL at `/mnt/c/`. Your project is at:

```bash
# Slow path (Windows drive via WSL):
/mnt/c/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/

# Create convenient shortcuts
ln -s "/mnt/c/Users/antho/Downloads/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients" ~/robot/project

# Verify
ls ~/robot/project/
```

### 8.3 Clone into WSL Native Filesystem (Faster for Docker)

For Docker builds, it's significantly faster to clone directly into the WSL filesystem:

```bash
# In Ubuntu WSL terminal:
cd ~
git clone <your-repo-url> robot/project
cd ~/robot/project
ls
```

When building Docker images, use this path (`~/robot/project`) as your build context instead of the `/mnt/c/...` path. Build times drop from minutes to seconds.

---

## 9. Part 7 — Building the Docker Images

### 9.1 Which Image to Build

| Image | Dockerfile | Use case |
|-------|-----------|----------|
| `go2-pose-x86:latest` | `Dockerfile_x86_sim` | **This one — for simulation on x86 PC** |
| `go2-pose:latest` | `Dockerfile_x86` | Real hardware on Linux workstation |
| `go2-nav2-sidecar:latest` | `Dockerfile_ros2_sidecar` | MPPI backend with ROS2/Nav2 |
| `go2-nav2-sidecar-lidar:latest` | `Dockerfile_ros2_sidecar_lidar` | MPPI + Hesai LiDAR |
| `go2-rviz:latest` | `Dockerfile_rviz` | RViz2 visualization window |

For the simulation workflow, you need at minimum: `go2-pose-x86:latest`.

### 9.2 Build the Simulation Image

```bash
# In Ubuntu WSL terminal:
cd ~/robot/project

# Quick build (uses the helper script)
bash docker/docker_build_x86_sim.sh
```

If you need to build manually (script fails):

```bash
# Check what Dockerfile the script uses:
cat docker/docker_build_x86_sim.sh
# It calls: docker build -f docker/Dockerfile_x86_sim -t go2-pose-x86:latest .

# Manual build with progress output:
docker build \
    --progress=plain \
    -f docker/Dockerfile_x86_sim \
    -t go2-pose-x86:latest \
    ~/robot/project/
```

> ⏱️ **First build takes 20–60 minutes.** It downloads a CUDA base image (~6 GB), installs PyTorch, builds OpenCV with CUDA support, installs TensorRT Python bindings, ByteTrack, and all other dependencies. Subsequent builds use the Docker layer cache and take seconds.

### 9.3 What's Inside the Simulation Image

The `Dockerfile_x86_sim` installs roughly:
- Ubuntu 22.04 base
- CUDA 12.x + cuDNN
- Python 3.10
- PyTorch 2.x (CUDA build)
- TensorRT 8.5+ Python bindings
- OpenCV 4.x with CUDA
- pycuda
- torchvision (for GPU NMS)
- ByteTrack multi-object tracker
- pyrealsense2 (RealSense SDK — still imported even in sim mode)
- ecs-logging (structured JSON logging)
- numpy, scipy, etc.

### 9.4 Build the ROS2 Sidecar (Optional for Sim)

Only needed if using `--follow-backend mppi`:

```bash
bash docker/docker_build_ros2_sidecar.sh
```

### 9.5 Verify All Images

```bash
docker images | grep go2

# Expected output (sizes are approximate):
# go2-pose-x86          latest    abc123def456   10 minutes ago   9.5GB
# go2-nav2-sidecar      latest    def456abc123    5 minutes ago   4.2GB
```

### 9.6 Save Docker Images for Teammates

If a teammate doesn't want to build from scratch (takes 40+ minutes):

```bash
# Save image to a tar file (~4 GB compressed)
docker save go2-pose-x86:latest | gzip > go2-pose-x86.tar.gz

# They load it with:
docker load < go2-pose-x86.tar.gz
```

Or push to Docker Hub:
```bash
docker tag go2-pose-x86:latest yourteam/go2-pose-x86:latest
docker push yourteam/go2-pose-x86:latest

# Teammate pulls with:
docker pull yourteam/go2-pose-x86:latest
```

---

## 10. Part 8 — Preparing AI Models (TensorRT Engines)

TensorRT `.trt` engine files are **compiled for a specific GPU architecture**. An engine built on an RTX 4070 will not load on an RTX 3080 — you get a runtime error. **Every team member must build their own engines on their own machine.**

### 10.1 Create the Models Directory

```bash
mkdir -p ~/robot/models
```

### 10.2 Get the YOLO11 Pose ONNX Model

```bash
# Build ONNX inside the Docker container (so it has the right dependencies)
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    go2-pose-x86:latest \
    bash -c "
        pip install ultralytics --quiet && \
        python3 -c \"
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
model.export(
    format='onnx',
    opset=13,
    half=False,
    imgsz=640,
    dynamic=False
)
import shutil
shutil.move('yolo11n-pose.onnx', '/models/yolo11n-pose.onnx')
print('Exported yolo11n-pose.onnx to /models/')
\"
    "
```

### 10.3 Convert YOLO11 ONNX → TensorRT FP16 Engine

```bash
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    go2-pose-x86:latest \
    bash -c "
        trtexec \
            --onnx=/models/yolo11n-pose.onnx \
            --saveEngine=/models/yolo11n-pose-fp16.trt \
            --fp16 \
            --workspace=4096 \
            --minShapes=images:1x3x640x640 \
            --optShapes=images:1x3x640x640 \
            --maxShapes=images:1x3x640x640 \
            --verbose
    "
```

This takes 2–10 minutes. The output ends with:
```
[I] Engine built in X.X sec.
[I] Saving engine to path /models/yolo11n-pose-fp16.trt
```

### 10.4 Get the OSNet-AIN ReID ONNX Model

```bash
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    go2-pose-x86:latest \
    bash -c "
        pip install torchreid --quiet && \
        python3 -c \"
import torchreid
import torch

model = torchreid.models.build_model(
    name='osnet_ain_x1_0',
    num_classes=1,
    pretrained=True
)
model.eval()
dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(
    model, dummy, '/models/osnet_ain_x1_0.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}},
    opset_version=11
)
print('Exported osnet_ain_x1_0.onnx')
\"
    "
```

### 10.5 Convert OSNet ONNX → TensorRT Engine

```bash
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    -v ~/robot/project/isaac:/workspace/isaac \
    go2-pose-x86:latest \
    python3 /workspace/isaac/build_reid_engine.py
```

### 10.6 Verify Both Engines Exist

```bash
ls -lh ~/robot/models/

# You need both of these:
# -rw-r--r-- 1 ...  12M  yolo11n-pose-fp16.trt
# -rw-r--r-- 1 ...   5M  osnet_ain_x1_0.trt
```

### 10.7 Quick Test That Engines Load

```bash
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    -v ~/robot/project/src:/workspace/src \
    go2-pose-x86:latest \
    bash -c "
        cd /workspace/src && python3 -c \"
from trt_inference import TRTInference
import numpy as np
trt = TRTInference('/models/yolo11n-pose-fp16.trt', verbose=True)
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = trt.infer(dummy)
print('TRT output shape:', out.shape)
print('Engine loads OK')
\"
    "
```

---

## 11. Part 9 — Network Configuration (The Hard Part)

This section covers the most common point of failure. Take your time here.

### 11.1 The Full Network Diagram

```
Your LAN (e.g., 192.168.1.0/24)
│
├── Windows Host
│   ├── LAN IP: 192.168.1.100  (your router-assigned IP)
│   ├── WSL virtual adapter (vEthernet WSL)
│   │   └── Windows side: 172.x.x.1
│   │       └── WSL side: 172.x.x.2  ← THIS is --frame-host value
│   │           └── OR: WSL might get a 192.168.x.x IP
│   └── Docker Desktop manages port forwarding into WSL
│
└── Isaac Sim (runs on Windows, sends to WSL IP:55002)
    └── SimRobotController in Docker (sends commands to host IP:55001)
```

### 11.2 Step-by-Step: Finding All the Right IPs

**Step 1 — Find WSL2 IP** (this is `--frame-host` for Isaac Sim):
```bash
# In Ubuntu WSL:
hostname -I | awk '{print $1}'
# Example: 192.168.1.91  or  172.19.168.233
```

**Step 2 — Find Windows host IP accessible from WSL** (this is where Docker sends commands):
```bash
# In Ubuntu WSL (nameserver in resolv.conf is usually the Windows host):
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Example: 172.19.160.1  or  192.168.1.100
```

**Step 3 — Verify connectivity from WSL to Windows:**
```bash
# Can WSL reach Windows? (Replace with Windows IP)
ping -c 3 172.19.160.1
```

**Step 4 — Verify from Windows to WSL:**
```powershell
# In Windows PowerShell: can Windows reach WSL?
ping 192.168.1.91
```

### 11.3 Configure Windows Firewall

**Option A — Add specific rules (recommended):**

```powershell
# Open PowerShell as Administrator

# Allow Isaac Sim to receive velocity commands from WSL (UDP 55001 inbound)
New-NetFirewallRule `
    -DisplayName "Isaac Sim: Receive velocity commands UDP 55001" `
    -Direction Inbound `
    -Protocol UDP `
    -LocalPort 55001 `
    -Action Allow `
    -Profile Any

# Allow Isaac Sim to send frames to WSL (UDP 55002 outbound)
New-NetFirewallRule `
    -DisplayName "Isaac Sim: Send camera frames UDP 55002" `
    -Direction Outbound `
    -Protocol UDP `
    -RemotePort 55002 `
    -Action Allow `
    -Profile Any

# Allow inbound on 55002 (WSL-bound traffic passes through Windows)
New-NetFirewallRule `
    -DisplayName "Isaac Sim: Camera frame receive WSL UDP 55002" `
    -Direction Inbound `
    -Protocol UDP `
    -LocalPort 55002 `
    -Action Allow `
    -Profile Any
```

**Option B — Temporarily disable firewall for testing:**

```powershell
# ONLY for debugging — re-enable after testing
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Re-enable after you confirm the setup works:
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

### 11.4 WSL2 Port Forwarding (If Needed)

Sometimes WSL2's networking doesn't automatically forward ports from Windows to WSL. If Isaac Sim can't reach the Docker container, add a port proxy:

```powershell
# In PowerShell as Administrator:

# Get your current WSL IP
$wslIp = (wsl hostname -I).Trim().Split(' ')[0]
Write-Host "WSL IP: $wslIp"

# Forward port 55002 from Windows to WSL
netsh interface portproxy add v4tov4 `
    listenaddress=0.0.0.0 `
    listenport=55002 `
    connectaddress=$wslIp `
    connectport=55002

# Verify the rule was added
netsh interface portproxy show all
```

To remove it later:
```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=55002
```

> ⚠️ Port proxy rules **don't survive reboots** by default. If you need them permanently, add the `netsh` commands to a startup task in Task Scheduler.

### 11.5 Test UDP Connectivity End-to-End

Before running Isaac Sim, verify UDP actually gets through:

**In WSL Terminal 1 (listen for frames):**
```bash
# Listen on port 55002 using netcat
nc -u -l -p 55002 -v
# Stays open waiting for data
```

**In Windows Command Prompt (send a test packet):**
```cmd
# Replace 192.168.1.91 with your WSL IP
echo test | powershell -c "$socket = New-Object System.Net.Sockets.UdpClient; $bytes = [System.Text.Encoding]::ASCII.GetBytes('test'); $socket.Send($bytes, $bytes.Length, '192.168.1.91', 55002)"
```

If WSL Terminal 1 prints `test`, UDP is flowing correctly. If nothing appears, the issue is in the firewall or port proxy.

### 11.6 The "WSL IP Changes on Reboot" Problem

Every time Windows restarts, WSL2 gets a new IP address from its virtual DHCP. This breaks your `--frame-host` argument.

**Solution A — Script to get current IP and run Isaac Sim:**

Create `run_isaac.bat` in your project folder:
```batch
@echo off
for /f "tokens=*" %%i in ('wsl hostname -I') do set WSL_IP=%%i
for /f "tokens=1" %%a in ("%WSL_IP%") do set WSL_IP=%%a
echo WSL2 IP is: %WSL_IP%
echo Starting Isaac Sim with --frame-host %WSL_IP%

C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat ^
    C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\isaac_env.py ^
    --person-move ^
    --frame-host %WSL_IP%
```

Double-click this `.bat` file and it automatically uses the correct current IP.

**Solution B — Static WSL2 IP via hosts file:**

```powershell
# In PowerShell as Administrator, set a static /etc/hosts entry
Add-Content -Path "C:\Windows\System32\drivers\etc\hosts" -Value "192.168.1.91 wsl-ubuntu"
```

Then in WSL, configure a static IP (advanced — requires editing Hyper-V settings or using a startup script).

---

## 12. Part 10 — Running the Full Simulation

### 12.1 Pre-Flight Checklist

Before every session:

```bash
# 1. Get current WSL IP (changes every reboot)
hostname -I | awk '{print $1}'

# 2. Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# 3. Verify models exist
ls -lh ~/robot/models/*.trt

# 4. Verify Docker image exists
docker images | grep go2-pose-x86
```

### 12.2 Step 1 — Launch Isaac Sim (Windows)

Open **Command Prompt** (`cmd.exe`, not PowerShell — use Start menu → type `cmd`):

```cmd
:: Replace 192.168.1.91 with your actual WSL IP from the checklist above
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat ^
    C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\isaac_env.py ^
    --person-move ^
    --frame-host 192.168.1.91
```

Wait until you see **both** of these lines in the Isaac Sim output:
```
[isaac_env] World ready. Physics @ 60 Hz, frames every 2 steps.
[isaac_env] Frame publisher sending to UDP 192.168.1.91:55002
[isaac_env] CMD receiver listening on UDP 0.0.0.0:55001
```

At this point, the sim is running and sending camera frames.

### 12.3 All Isaac Sim Flags

```cmd
:: Full command with all available flags and their defaults:
C:\isaacsim\...\python.bat C:\...\isaac\isaac_env.py ^
    --frame-host 192.168.1.91 ^    [IP to send frames to — YOUR WSL IP]
    --frame-port 55002 ^           [UDP port for frames]
    --cmd-port 55001 ^             [UDP port for velocity commands]
    --physics-hz 60 ^              [Physics rate — don't exceed 120]
    --render-every 2 ^             [Publish frame every N physics steps]
    --person-x 2.5 ^               [Person starting X position (meters)]
    --person-y 0.0 ^               [Person starting Y position (meters)]
    --person-move ^                [Enable person patrol walk]
    --headless                     [Disable GUI — faster, for CI/headless servers]
```

Performance guide for `--render-every`:
- `--render-every 1` → 60 fps, heavier GPU load
- `--render-every 2` → 30 fps, balanced (default)
- `--render-every 3` → 20 fps, lighter, use if GPU is struggling

### 12.4 Step 2 — Launch Vision Container (WSL)

Open **Ubuntu WSL** terminal:

```bash
# Set your WSL IP (for sim robot controller target)
WSL_IP=$(hostname -I | awk '{print $1}')
echo "WSL IP: $WSL_IP"

# Run the container
docker run --rm -it --gpus all \
    -p 55002:55002/udp \
    -v ~/robot/models:/models \
    -v ~/robot/project/src:/workspace/src \
    go2-pose-x86:latest \
    bash -c "
        # Fix numpy deprecation in ByteTrack (safe to run every time)
        find /opt/bytetrack -type f -name '*.py' -exec \
            sed -i \
            's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' \
            {} + 2>/dev/null || true

        cd /workspace/src && python3 main.py \
            --sim \
            --follow \
            --follow-backend pid \
            --trt-engine /models/yolo11n-pose-fp16.trt \
            --osnet-trt-engine /models/osnet_ain_x1_0.trt \
            --target-distance 0.8 \
            --preview-fps 10 \
            --rotate 0
    "
```

### 12.5 What to Expect on Successful Connection

```
[main] Sim mode: using SimCameraCapture
[SimCameraCapture] Listening on UDP 0.0.0.0:55002
[SimCameraCapture] Socket bound to 0.0.0.0:55002 – waiting for data...
[YoloPoseInference] GPU NMS enabled (torchvision)
[main] Sim mode: using SimRobotController
[SimRobotController] Sending commands to 192.168.1.91:55001

# These appear when frames start arriving:
[SimCameraCapture] Received XXXX bytes
[SimCameraCapture] Decoded frame, queue size 1
[DEBUG] TRT output:
  shape: (1, 56, 8400)
  dtype: float32
  min/max: -28.8125 688.625
  any NaN: False
```

Timeouts while waiting are normal — they just mean Isaac Sim hasn't sent a frame yet. Once Isaac Sim's world loads (~30 sec), frames start flowing.

### 12.6 All Vision Pipeline Flags

```bash
python3 main.py \
    # ── Simulation mode ──────────────────────────────────────────
    --sim                          # Use Isaac Sim (not RealSense camera)
    --frame-port 55002             # UDP port to receive sim frames
    --cmd-port 55001               # UDP port to send velocity commands

    # ── Models ───────────────────────────────────────────────────
    --trt-engine /models/yolo11n-pose-fp16.trt
    --osnet-trt-engine /models/osnet_ain_x1_0.trt

    # ── Following behavior ───────────────────────────────────────
    --follow                       # Enable person following
    --follow-backend pid           # 'pid' or 'mppi'
    --target-distance 0.8          # Stop this far from person (meters)
    --motion-lock-frames 10        # Detections needed before moving

    # ── Camera ───────────────────────────────────────────────────
    --rotate 0                     # Rotate frame 0/90/180/270 degrees
    --camera-mode single           # Only 'single' supported now

    # ── Preview ──────────────────────────────────────────────────
    --preview-fps 10               # Max refresh rate of preview window
    --headless                     # No preview window (faster)
    --rotation-debug               # Show rotation error debug window

    # ── Preprocessing ────────────────────────────────────────────
    --preprocess-backend gpu       # 'gpu' or 'cpu'

    # ── PID tuning (--follow-backend pid only) ───────────────────
    --kp 0.9 --kd 0.3 --ki 0.0    # Forward/back PID gains
    --trans-x-max 0.6              # Max forward speed
    --trans-x-tolerance 0.3        # Dead zone (meters)
    --rot-kp 0.0                   # Rotation PID (0 = disabled)
    --edge-penalty-k 10.0          # Penalty when person near frame edge
    --size-penalty-k 8.0           # Penalty when person bbox is small

    # ── ReID tuning ──────────────────────────────────────────────
    --reid-match-thresh 0.85       # Cosine similarity to re-identify person
    --reid-gallery-size 50         # Max embeddings kept in gallery
    --reid-reacquire-timeout-sec 5.0  # Give up reacquire after this long

    # ── MPPI export (--follow-backend mppi only) ─────────────────
    --target-export-host 0.0.0.0   # Where to send target UDP packets
    --target-export-port 41234     # Port for target packets

    # ── Logging ──────────────────────────────────────────────────
    --debug                        # Verbose TRT output + extra logging
    --log-components all           # ECS JSON logs: none/all/vision.main/vision.exporter
    --ecs-log-dir ./logs           # Where to write ECS logs
    --debug-trace-dir ./debug_logs # JSONL timing trace logs
    --debug-trace-every-n-frames 1 # How often to emit timing traces
```

### 12.7 MPPI Backend (More Sophisticated Following)

If you want to use Nav2's MPPI controller instead of the simple PID:

```bash
# Terminal 1 — Start Isaac Sim (same as before)

# Terminal 2 — Start vision with MPPI backend
docker run --rm -it --gpus all \
    -p 55002:55002/udp \
    -v ~/robot/models:/models \
    -v ~/robot/project/src:/workspace/src \
    go2-pose-x86:latest \
    bash -c "
        find /opt/bytetrack -type f -name '*.py' -exec \
            sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} + 2>/dev/null || true
        cd /workspace/src && python3 main.py \
            --sim \
            --follow \
            --follow-backend mppi \
            --target-export-host 0.0.0.0 \
            --target-export-port 41234 \
            --trt-engine /models/yolo11n-pose-fp16.trt \
            --osnet-trt-engine /models/osnet_ain_x1_0.trt
    "

# Terminal 3 — Start ROS2 sidecar
bash ~/robot/project/docker/docker_run_ros2_sidecar.sh
```

---

## 13. Part 11 — ROS2 Sidecar (MPPI + Nav2)

### 13.1 What the Sidecar Does

```
[Vision Container] ──UDP 41234──→ [person_follow_nav node]
                                        ↓  Path generation
                                  [Nav2 MPPI Controller]
                                        ↓  cmd_vel
                                  [velocity_smoother]
                                        ↓  cmd_vel_smoothed
                                  [go2_nav_bridge node]
                                        ↓  UDP commands
                                  [SimRobotController] ──→ Isaac Sim
```

The `person_follow_nav` node receives target x,y positions in the robot's base frame, generates a local Nav2 path, and feeds it to the MPPI controller which produces smooth velocity commands.

### 13.2 Run the Sidecar for Simulation

```bash
bash ~/robot/project/docker/docker_run_ros2_sidecar.sh
```

Or with the all-in-one orchestrator:

```bash
cd ~/robot/project

# Start vision + sidecar together
bash docker/start_follow_system.sh up

# With debug logging enabled
bash docker/start_follow_system.sh up --log sidecar.follow --log vision.main

# Check status
bash docker/start_follow_system.sh status

# Stop everything
bash docker/start_follow_system.sh down

# View logs
docker logs -f go2-follow-vision
docker logs -f go2-follow-sidecar
```

### 13.3 MPPI Tuning (nav2_controller.yaml)

Key parameters in `ros2_ws/src/person_follow_nav/config/nav2_controller.yaml`:

```yaml
FollowPath:
  time_steps: 48        # Planning horizon length (more = smoother but slower)
  batch_size: 1000      # Random trajectory samples (more = better quality)
  temperature: 0.3      # Randomness (lower = more deterministic)
  vx_max: 1.0           # Max forward speed (m/s)
  vy_max: 0.56          # Max lateral speed (m/s)
  wz_max: 2.59          # Max rotation rate (rad/s)

# Target following parameters (in follow_sidecar.launch.py defaults):
desired_distance: 0.35    # Target distance to keep from person (meters)
follow_tolerance_m: 0.40  # Dead zone radius (don't move if within this)
target_timeout_sec: 0.8   # Stop if no target for this long
target_hold_sec: 0.7      # Keep last valid target this long after signal lost
```

---

## 14. Part 12 — LiDAR Support

LiDAR adds obstacle avoidance via Nav2 costmaps. This is only functional on real hardware with a Hesai XT16 sensor; in simulation it's not yet integrated.

### 14.1 Build the LiDAR Image

```bash
bash ~/robot/project/docker/docker_build_ros2_sidecar_lidar.sh
```

### 14.2 Run with LiDAR Enabled

```bash
SIDECAR_ENABLE_LIDAR_MAPPING=1 \
    bash ~/robot/project/docker/start_follow_system.sh up
```

### 14.3 LiDAR Filter Stages

The `hesai_lidar_filter` C++ node applies three sequential filters to `/xt16/lidar_points`:

1. **CropBox** — Removes points inside ±0.2m×±0.3m box (the robot's own body)
2. **PassThrough** — Keeps only points within ±2m×±2m×(−0.2m to 3m) volume
3. **DROR (Dynamic Radius Outlier Removal)** — Distance-aware noise filtering; uses a search radius that scales with `2 × range × sin(azimuth_angle)` so it filters more aggressively at range

Filtered cloud publishes to `/lidar_points_filter`, which Nav2's voxel costmap subscribes to for obstacle marking.

---

## 15. Troubleshooting

### ❌ `[SimCameraCapture] Timeout -- is isaac_env.py running?`

Frames aren't reaching the Docker container.

**Checklist:**
```bash
# 1. Is Isaac Sim running and showing "Frame publisher sending to..."?
#    Check the Isaac Sim CMD window

# 2. Is --frame-host set to the CURRENT WSL IP?
hostname -I | awk '{print $1}'
# Compare to what you passed to Isaac Sim

# 3. Is port 55002 mapped?
docker ps --format "table {{.Names}}\t{{.Ports}}"
# Should show: 0.0.0.0:55002->55002/udp

# 4. Test UDP directly
# Terminal 1 (WSL):
nc -u -l -p 55002 &
# Terminal 2 (WSL), send test packet FROM WINDOWS to see if it arrives:
```

```powershell
# In Windows PowerShell — send a test UDP packet to WSL
$udpClient = New-Object System.Net.Sockets.UdpClient
$bytes = [System.Text.Encoding]::ASCII.GetBytes("test")
$udpClient.Send($bytes, $bytes.Length, "192.168.1.91", 55002)
$udpClient.Close()
```

If `nc` receives "test" — UDP works, the issue is in Isaac Sim. If not — it's firewall or port proxy.

---

### ❌ `Error response from daemon: could not select device driver "nvidia"`

The NVIDIA Container Toolkit isn't configured properly.

```bash
# Step 1: Reinstall
sudo apt-get install -y nvidia-container-toolkit

# Step 2: Reconfigure
sudo nvidia-ctk runtime configure --runtime=docker

# Step 3: Check daemon.json looks right
cat /etc/docker/daemon.json

# Step 4: Restart Docker Desktop from Windows tray
# Right-click whale icon → Restart Docker Desktop

# Step 5: Test again
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

### ❌ Isaac Sim crashes immediately on launch

**Symptom:** Window appears briefly then disappears, or never appears at all.

**Fixes:**
```cmd
:: Fix 1: Update GPU driver
:: Download from https://www.nvidia.com/drivers — must be 525+ for Isaac Sim 5.x

:: Fix 2: Install Visual C++ Redistributable
:: https://aka.ms/vs/17/release/vc_redist.x64.exe

:: Fix 3: Check the Kit log for the actual error
:: Log location:
:: C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\kit\logs\Kit\Isaac-Sim Python\5.1\kit_YYYYMMDD_HHMMSS.log

:: Fix 4: Run headless to bypass rendering bugs
C:\isaacsim\...\python.bat C:\...\isaac_env.py --headless --frame-host 192.168.1.91
```

---

### ❌ Isaac Sim runs but the Go2 robot immediately falls through the floor

This is a physics initialization issue. The robot spawns 40cm in the air (`position=np.array([0.0, 0.0, 0.4])`) but sometimes PhysX doesn't initialize fast enough.

**Fix:** Add a longer sleep after `world.reset()` in `isaac_env.py`:
```python
world.reset()
# Add this:
import time
time.sleep(2.0)  # Give PhysX time to settle
```

---

### ❌ `np.float` / `np.int` / `np.bool` AttributeError in ByteTrack

NumPy 1.24+ removed these aliases. The Docker run command patches them:

```bash
find /opt/bytetrack -type f -name '*.py' -exec \
    sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} +
```

If this runs but you still get errors, the file might be a `.pyc` cached bytecode. Fix:

```bash
find /opt/bytetrack -name '*.pyc' -delete
find /opt/bytetrack -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
# Then re-run the sed patch
```

---

### ❌ TensorRT engine fails to load or gives wrong output shape

```
RuntimeError: Failed to deserialize TensorRT engine
# or
shape: (1, X, Y) where X or Y are wrong
```

The engine was built for a different GPU or TensorRT version.

```bash
# Rebuild the engine on THIS machine inside the container:
docker run --rm --gpus all \
    -v ~/robot/models:/models \
    go2-pose-x86:latest \
    bash -c "
        trtexec \
            --onnx=/models/yolo11n-pose.onnx \
            --saveEngine=/models/yolo11n-pose-fp16.trt \
            --fp16
    "
```

---

### ❌ `[SimRobotController] Send error: [Errno 111] Connection refused`

The sim velocity commands can't reach Isaac Sim's UDP receiver.

```bash
# The controller sends to Windows host IP on port 55001
# Check what IP it's sending to:
grep "SimRobotController" ~/robot/project/src/sim_robot_controller.py

# The IP is hardcoded to 192.168.1.91 in the docker run CMD above
# Find actual Windows host IP from WSL:
cat /etc/resolv.conf | grep nameserver

# Update the docker run command with the correct host:
# Replace 192.168.1.91 with your Windows host IP in SimRobotController initialization
# OR: use --network host (but this removes Docker's network isolation)
```

---

### ❌ Very low FPS (< 5 fps)

Multiple possible causes:

```bash
# Check GPU utilization
docker exec -it <container-name> nvidia-smi

# Check if both GPU processes are competing
nvidia-smi  # From Windows too — Isaac Sim uses GPU for rendering

# Fixes:
# 1. Run Isaac Sim headless to free GPU for inference:
#    Add --headless to Isaac Sim command

# 2. Reduce Isaac Sim render rate:
#    Add --render-every 3 to Isaac Sim command

# 3. Reduce preview FPS in vision pipeline:
#    Add --preview-fps 5 or --headless

# 4. Check WSL memory isn't swapping:
free -h  # Should have plenty of free RAM
```

---

### ❌ Docker build fails with network errors

```bash
# Error: unable to resolve host / Could not resolve 'archive.ubuntu.com'
# Fix: Configure DNS in WSL
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Or add to /etc/wsl.conf:
sudo tee /etc/wsl.conf << EOF
[network]
generateResolvConf = false
EOF

# Then:
sudo rm /etc/resolv.conf
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
```

---

### ❌ Port 55002 already in use

```bash
# Find what's using it
sudo lsof -i udp:55002
# or
sudo ss -ulpn | grep 55002

# Kill the old container
docker ps | grep 55002
docker stop <container-id>

# Or kill the process
sudo kill <PID>
```

---

### ❌ WSL2 runs out of disk space

```bash
# Check WSL disk usage
df -h /

# Clean Docker
docker system prune -af --volumes  # ⚠️ This removes ALL unused images!

# Check what's using space
du -sh ~/robot/models/*
du -sh ~/robot/debug_logs/ 2>/dev/null || true
```

---

### ❌ Isaac Sim shows "Nucleus not found" or asset download fails

Assets are loaded from NVIDIA's S3 bucket. Check:

```cmd
:: Test internet access from Windows CMD
curl -I https://omniverse-content-production.s3-us-west-2.amazonaws.com/

:: If behind a corporate proxy, set HTTP_PROXY / HTTPS_PROXY environment variables
:: before launching Isaac Sim
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
C:\isaacsim\...\python.bat C:\...\isaac_env.py ...
```

---

### 🔍 Debug Tools Reference

```bash
# Watch UDP traffic (frames Isaac Sim → Docker)
sudo tcpdump -i eth0 udp port 55002 -n -c 20

# Watch UDP traffic (commands Docker → Isaac Sim)
sudo tcpdump -i eth0 udp port 55001 -n -c 20

# GPU utilization in real-time
watch -n 1 nvidia-smi

# Container resource usage
docker stats --no-stream

# Container live logs
docker logs -f <container-name> 2>&1 | head -100

# Inspect container networking
docker inspect <container-name> | python3 -m json.tool | grep -A5 "Ports"

# Read ECS structured logs prettily
tail -f ~/robot/debug_logs/*/vision/ecs/*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try: print(json.dumps(json.loads(line), indent=2))
    except: print(line, end='')
"
```

---

## 16. Project Structure Reference

```
Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/
│
├── src/                              # Vision pipeline (runs in Docker)
│   ├── main.py                       # ★ Entry point — main inference loop
│   ├── args_parser.py                # All CLI arguments
│   ├── sim_camera_capture.py         # ★ Isaac Sim UDP frame receiver
│   ├── sim_robot_controller.py       # ★ Isaac Sim UDP velocity sender
│   ├── camera_capture.py             # RealSense camera wrapper
│   ├── robot_controller.py           # Unitree Go2 SportClient wrapper
│   ├── yolo_pose_inference.py        # YOLO11 decode + NMS + draw
│   ├── trt_inference.py              # TensorRT engine runner
│   ├── reid_trt_inference.py         # ReID TensorRT engine runner
│   ├── reid_manager.py               # ReID gallery + state machine
│   ├── reid_augment.py               # LGPR (patch grayscale) augmentation
│   ├── nfc_gpu.py                    # GPU Neighborhood Feature Convolution
│   ├── single_person_tracker.py      # ByteTrack wrapper + main person select
│   ├── person_follower.py            # PID following + depth estimation
│   ├── pid_controller.py             # PID with anti-windup + smoothing
│   ├── depth_processor.py            # Bimodal depth histogram + centroid
│   ├── vision_target_export.py       # UDP target exporter for MPPI sidecar
│   ├── pixel_to_3d_api.py            # RealSense distortion-aware deprojection
│   ├── dual_camera_system.py         # Stereo charuco calibration + 3D
│   ├── dual_camera_centerline_viewer.py  # Calibration quality tool
│   ├── visualization.py              # OpenCV overlay drawing
│   ├── structured_logging.py         # ECS JSON logging
│   ├── debug_trace_logger.py         # JSONL trace logger
│   └── utils.py                      # rotate_image, draw_fps
│
├── isaac/                            # Isaac Sim scripts (Windows only)
│   ├── isaac_env.py                  # ★ Main sim loop — run on Windows
│   ├── go2_usd_setup.py              # Download Go2 URDF/USD helper
│   └── build_reid_engine.py          # Convert OSNet ONNX → TRT
│
├── docker/
│   ├── Dockerfile                    # ARM64 (Jetson) image
│   ├── Dockerfile_x86                # x86 production image
│   ├── Dockerfile_x86_sim            # ★ x86 simulation image (use this)
│   ├── Dockerfile_ros2_sidecar       # ROS2 Humble + Nav2 sidecar
│   ├── Dockerfile_ros2_sidecar_lidar # ROS2 + Nav2 + Hesai LiDAR driver
│   ├── Dockerfile_rviz               # RViz2 visualization
│   ├── docker_build_x86_sim.sh       # ★ Build simulation image
│   ├── docker_run.sh                 # Run production image
│   ├── docker_run_ros2_sidecar.sh    # Run sidecar
│   ├── docker_run_rviz.sh            # Run RViz2
│   └── start_follow_system.sh        # ★ All-in-one up/down/status/logs
│
├── ros2_ws/src/
│   ├── person_follow_nav/            # UDP target → Nav2 path generation
│   │   ├── person_follow_nav/
│   │   │   ├── follow_controller_node.py  # Main ROS2 follow node
│   │   │   ├── camera_client.py           # RealSense for ROS2 sidecar
│   │   │   ├── debug_trace_logger.py      # JSONL + CSV tracers
│   │   │   └── ecs_logging_utils.py       # ECS structured logging
│   │   ├── launch/
│   │   │   ├── follow_sidecar.launch.py        # Without LiDAR
│   │   │   └── follow_sidecar_lidar.launch.py  # With LiDAR
│   │   └── config/nav2_controller.yaml   # MPPI + costmap + smoother config
│   │
│   ├── go2_nav_bridge/               # Nav2 cmd_vel → Unitree DDS
│   │   └── go2_nav_bridge/
│   │       ├── bridge_node.py        # Translates cmd_vel to SportClient.Move()
│   │       ├── debug_trace_logger.py
│   │       └── ecs_logging_utils.py
│   │
│   ├── hesai_lidar_filter/           # Hesai XT16 point cloud filter (C++)
│   │   ├── src/
│   │   │   ├── lidar_filter_node.cpp # CropBox + PassThrough + DROR
│   │   │   └── DROH.h                # Dynamic Radius Outlier Removal impl
│   │   └── launch/filter.launch.py
│   │
│   └── pointcloud_to_grid_ros2/      # PointCloud2 → OccupancyGrid for Nav2
│       └── pointcloud_to_grid/
│           ├── pointcloud_to_grid_node.py
│           ├── interpolated_grid_node.py
│           ├── pointcloud_to_grid_core.py
│           └── point_cloud2.py       # PointCloud2 deserialization helper
│
├── models/                           # TensorRT engines — NOT in git
│   ├── yolo11n-pose-fp16.trt         # ★ Required — build from ONNX
│   └── osnet_ain_x1_0.trt           # ★ Required — build from ONNX
│
├── debug_logs/                       # Auto-created at runtime
│   └── follow_run_YYYYMMDD_HHMMSS/
│       ├── vision/ecs/*.jsonl        # ECS structured vision logs
│       ├── vision/debug_trace/*.jsonl # Timing traces
│       ├── sidecar/ecs/*.jsonl       # ECS structured sidecar logs
│       └── sidecar/debug_trace/*.jsonl
│
└── sync.sh                           # rsync to remote machine helper
```

★ = Files you'll interact with most during simulation development

---

## 17. Quick Reference Card

### Every Session — 3 Steps

**Step 1 (WSL): Get current IP**
```bash
hostname -I | awk '{print $1}'
# Note this IP — use it in Step 2
```

**Step 2 (Windows CMD): Start Isaac Sim**
```cmd
C:\isaacsim\IsaacSim-main\_build\windows-x86_64\release\python.bat ^
    C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\isaac_env.py ^
    --person-move ^
    --frame-host YOUR_WSL_IP_HERE
```

**Step 3 (WSL): Start Vision Container**
```bash
docker run --rm -it --gpus all \
    -p 55002:55002/udp \
    -v ~/robot/models:/models \
    -v ~/robot/project/src:/workspace/src \
    go2-pose-x86:latest \
    bash -c "find /opt/bytetrack -type f -name '*.py' -exec sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} + 2>/dev/null; cd /workspace/src && python3 main.py --sim --follow --follow-backend pid --trt-engine /models/yolo11n-pose-fp16.trt --osnet-trt-engine /models/osnet_ain_x1_0.trt"
```

---

### Key Ports

| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| `55001` | UDP | Docker → Windows | Robot velocity commands |
| `55002` | UDP | Windows → Docker | Camera frames (JPEG+zlib) |
| `41234` | UDP | Vision → Sidecar | MPPI target positions (x,y in base_link) |

### Python Environments at a Glance

| Where | Python | How to run |
|-------|--------|------------|
| `isaac/*.py` | Isaac Sim bundled 3.10 | `python.bat script.py` |
| `src/*.py` | Docker container 3.10 | `docker run ... python3 main.py` |
| `ros2_ws/**` | ROS2 container 3.10 | `docker run ... ros2 launch ...` |

### Build Commands

```bash
bash docker/docker_build_x86_sim.sh          # Main vision image
bash docker/docker_build_ros2_sidecar.sh     # ROS2 Nav2 sidecar
bash docker/docker_build_ros2_sidecar_lidar.sh  # ROS2 + LiDAR
bash docker/docker_build_rviz.sh             # RViz2 viewer
```

### Orchestration Commands

```bash
bash docker/start_follow_system.sh up        # Start vision + sidecar
bash docker/start_follow_system.sh down      # Stop everything
bash docker/start_follow_system.sh status    # Check containers
bash docker/start_follow_system.sh logs      # Show log targets
docker logs -f go2-follow-vision             # Tail vision logs
docker logs -f go2-follow-sidecar            # Tail sidecar logs
```

---

*Last updated for: Isaac Sim 5.1 · Docker Desktop 4.x · WSL2 Ubuntu 22.04 · CUDA 12.6 · ROS2 Humble · YOLO11n-pose*
