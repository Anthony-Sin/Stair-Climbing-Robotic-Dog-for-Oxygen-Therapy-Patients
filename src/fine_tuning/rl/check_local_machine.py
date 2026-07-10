"""Check whether THIS computer can run the blind-RL stair fine-tune locally.

Unlike ``preflight_rl.py`` (which assumes the training stack is already installed and
checks the *deploy contract*), this script inspects the raw HARDWARE + host tooling and
tells you, in plain language, what you can run locally, what you cannot, and the settings
to use (e.g. how many parallel envs your VRAM can hold). It is safe to run on a bare
Windows laptop with nothing installed -- every probe is wrapped so a missing piece is
reported, never crashed on.

    py -3.11 src/fine_tuning/rl/check_local_machine.py
    # or just:  src\fine_tuning\rl\check_my_computer.bat

It answers three questions:
  1) Do I have the hardware (CUDA GPU, VRAM, disk, RAM) to train at all?
  2) Do I have the plumbing (Docker + WSL2 GPU, or a native Isaac Sim) to host IsaacLab?
  3) Given my VRAM, how many envs should I train with, and how long will it take?

Nothing here is RunPod-specific -- it is the local-machine counterpart of the pod's
`runpod_setup_rl.sh` + `preflight_rl.py`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

# ---- tiny result model (no dependency on the fine_tuning package so this runs anywhere) --
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
_MARK = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[ -- ]"}

# Sizing constants (see verdict()): Isaac Sim's renderer/PhysX base cost, per-env VRAM for
# the Go2 rough task, and the disk the whole stack needs.
ISAAC_BASE_VRAM_GB = 3.0      # Isaac Sim app + PhysX context floor before a single env
PER_ENV_VRAM_MB = 3.3        # ~measured for Go2 rough (4096 envs ~ 13-15 GB used on the pod)
MIN_VRAM_GB = 6.0            # below this, headless training is not realistic
STACK_DISK_GB = 35.0        # Isaac Sim pip (~15-20 GB) + IsaacLab + robot_lab + caches


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self, title: str) -> str:
        w = max((len(n) for _, n, _ in self.rows), default=10)
        lines = [f"  {title}", "  " + "-" * (len(title))]
        for status, name, detail in self.rows:
            lines.append(f"  {_MARK[status]}  {name.ljust(w)}  {detail}")
        return "\n".join(lines)

    @property
    def worst(self) -> str:
        order = {FAIL: 3, WARN: 2, PASS: 1, INFO: 0}
        return max((s for s, _, _ in self.rows), key=lambda s: order[s], default=INFO)


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    """Run a command, return stdout stripped, or None on any failure (never raises)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return None
        return p.stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- host / cpu / ram
def check_host(rep: Report) -> None:
    rep.add(INFO, "os", f"{platform.system()} {platform.release()} ({platform.machine()})")
    rep.add(INFO, "cpu cores", str(os.cpu_count() or "?"))
    # RAM (best-effort, no psutil dep)
    ram_gb = None
    try:
        if platform.system() == "Windows":
            # wmic is removed on newer Windows -> prefer a PowerShell CIM query, fall back to wmic.
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]) \
                or _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]) or ""
            digits = [int(s) for s in out.split() if s.isdigit()]
            if digits:
                ram_gb = max(digits) / (1024**3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) / (1024**2)
                        break
    except Exception:
        pass
    if ram_gb:
        rep.add(PASS if ram_gb >= 15 else WARN, "system RAM",
                f"{ram_gb:.1f} GB" + ("" if ram_gb >= 15 else "  (16 GB+ recommended)"))
    else:
        rep.add(INFO, "system RAM", "could not read")


def check_python(rep: Report) -> None:
    pv = platform.python_version()
    rep.add(PASS if pv.startswith("3.11") else INFO, "this python",
            f"{pv}  ({sys.executable})")
    # enumerate installed pythons on Windows (py launcher)
    if platform.system() == "Windows":
        out = _run(["py", "-0p"])
        if out:
            has311 = "3.11" in out
            rep.add(PASS if has311 else WARN, "python 3.11 present",
                    "yes (py -3.11)" if has311 else "NOT found -- Isaac Sim 5.1 needs Python 3.11")


# --------------------------------------------------------------------------- gpu
def _nvidia_smi_query(fields: str) -> Optional[List[str]]:
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    return [c.strip() for c in out.splitlines()[0].split(",")]


def check_gpu(rep: Report) -> Optional[float]:
    """Return total VRAM in GB (or None). Adds GPU rows to the report."""
    vals = _nvidia_smi_query("name,memory.total,memory.free,driver_version")
    if not vals:
        rep.add(FAIL, "cuda gpu", "nvidia-smi not found / no NVIDIA GPU -- cannot train IsaacLab locally.")
        return None
    name = vals[0]
    total_gb = float(vals[1]) / 1024 if vals[1].replace(".", "").isdigit() else 0.0
    free_gb = float(vals[2]) / 1024 if len(vals) > 2 and vals[2].replace(".", "").isdigit() else 0.0
    driver = vals[3] if len(vals) > 3 else "?"
    used_gb = max(0.0, total_gb - free_gb)

    status = PASS if total_gb >= MIN_VRAM_GB else FAIL
    rep.add(status, "cuda gpu", f"{name}  |  driver {driver}")
    rep.add(status if total_gb >= 12 else WARN, "gpu VRAM total",
            f"{total_gb:.1f} GB" + ("" if total_gb >= 12 else "  (8 GB is tight -- see recommended num_envs below)"))
    rep.add(WARN if used_gb > 1.5 else PASS, "gpu VRAM free now",
            f"{free_gb:.1f} GB free, {used_gb:.1f} GB in use"
            + ("  <-- CLOSE browsers/apps before training to free VRAM" if used_gb > 1.5 else ""))
    # laptop GPUs are power/throughput limited -> flag expected slowness
    if "laptop" in name.lower() or "mobile" in name.lower():
        rep.add(WARN, "gpu class", "LAPTOP GPU -- expect ~2-4x slower than a desktop pod; fine-tune, don't train from scratch.")
    return total_gb


# --------------------------------------------------------------------------- disk
def check_disk(rep: Report) -> None:
    drive = os.path.splitdrive(os.path.abspath(__file__))[0] or "/"
    try:
        total, used, free = shutil.disk_usage(drive + "\\" if platform.system() == "Windows" else "/")
        free_gb = free / (1024**3)
        # WSL2/Docker stores images on the same physical disk on a default Windows install.
        rep.add(PASS if free_gb >= STACK_DISK_GB else WARN, f"disk free ({drive or '/'})",
                f"{free_gb:.0f} GB free"
                + ("" if free_gb >= STACK_DISK_GB else f"  (need ~{STACK_DISK_GB:.0f} GB for the Isaac stack -- free some space)"))
    except Exception as exc:
        rep.add(INFO, "disk free", f"could not read ({type(exc).__name__})")


# --------------------------------------------------------------------------- docker + wsl (the linux GPU host)
def check_docker_wsl(rep: Report) -> Tuple[bool, bool]:
    """Returns (docker_ready, wsl2_gpu_ready)."""
    ver = _run(["docker", "--version"])
    if not ver:
        rep.add(WARN, "docker", "not found -- needed for the Linux/WSL2 GPU training path (or install Isaac Sim natively).")
        docker_ok = False
    else:
        rep.add(PASS, "docker", ver)
        docker_ok = _run(["docker", "info", "--format", "{{.ServerVersion}}"]) is not None
        rep.add(PASS if docker_ok else WARN, "docker daemon",
                "running" if docker_ok else "installed but not running -- start Docker Desktop.")

    # WSL2 (Docker Desktop's Linux backend on Windows; also where GPU passthrough lives)
    wsl_gpu = False
    if platform.system() == "Windows":
        out = _run(["wsl", "-l", "-v"])
        # wsl output is UTF-16-ish; _run decodes best-effort. Look for a v2 distro.
        if out and "2" in out:
            rep.add(PASS, "wsl2", "present (Docker Desktop backend + CUDA-in-WSL host)")
            wsl_gpu = True
        elif out:
            rep.add(WARN, "wsl2", "WSL present but no v2 distro detected")
        else:
            rep.add(WARN, "wsl2", "not detected -- required for Docker GPU on Windows")
    else:
        wsl_gpu = docker_ok

    # Prove GPU passthrough actually works: does any running container see the GPU, or can
    # docker enumerate the nvidia runtime? Cheap check first (no image pull).
    runtimes = _run(["docker", "info", "--format", "{{json .Runtimes}}"]) if docker_ok else None
    if runtimes and "nvidia" in runtimes:
        rep.add(PASS, "docker gpu runtime", "nvidia runtime available")
    elif docker_ok:
        # On WSL2, --gpus works via the WSL CUDA driver even without a named 'nvidia' runtime.
        rep.add(INFO, "docker gpu runtime",
                "no named nvidia runtime; on WSL2 use `--gpus all` (works via the WSL CUDA driver).")

    # THE make-or-break check for local Isaac Sim: WSL needs GPU *Vulkan*, not just CUDA.
    # WSL commonly has libcuda but NOT the NVIDIA Vulkan lib -> Vulkan falls back to llvmpipe
    # (CPU) and Isaac Sim's RTX renderer refuses/crawls. Probe for the lib directly.
    vulkan_ok = False
    if platform.system() == "Windows" and wsl_gpu:
        distro = os.environ.get("FT_RL_WSL_DISTRO", "Ubuntu")
        probe = _run(["wsl", "-d", distro, "-e", "bash", "-lc",
                      "ls /usr/lib/wsl/lib/libGLX_nvidia.so.0 >/dev/null 2>&1 && echo HAVE || echo MISSING"],
                     timeout=40)
        if probe and "HAVE" in probe:
            rep.add(PASS, "wsl gpu vulkan", "NVIDIA Vulkan lib present in WSL -- Isaac Sim can render.")
            vulkan_ok = True
        elif probe and "MISSING" in probe:
            rep.add(FAIL, "wsl gpu vulkan",
                    "NVIDIA Vulkan lib ABSENT (WSL has CUDA only) -- Isaac Sim CANNOT run. Fix: clean-"
                    "reinstall the latest NVIDIA driver on Windows, then `wsl --shutdown`. Verify with "
                    "`wsl vulkaninfo --summary` (must show the RTX GPU, not llvmpipe).")
        else:
            rep.add(WARN, "wsl gpu vulkan", "could not probe (WSL slow / not Ubuntu?) -- verify with "
                    "`wsl vulkaninfo --summary` (needs the RTX GPU, not llvmpipe).")
    return docker_ok, wsl_gpu, vulkan_ok


# --------------------------------------------------------------------------- native training stack (optional)
def check_native_stack(rep: Report) -> None:
    """Is the IsaacLab training stack installed in THIS python? (Usually no on a laptop.)"""
    for mod, label in [("torch", "torch"), ("isaacsim", "isaac sim (pip)"),
                       ("isaaclab", "isaaclab"), ("rsl_rl", "rsl_rl"), ("robot_lab", "robot_lab")]:
        try:
            m = __import__(mod)
            extra = ""
            if mod == "torch":
                try:
                    cu = getattr(m, "version", None)
                    cuda_ok = bool(getattr(m, "cuda", None) and m.cuda.is_available())
                    extra = f"{m.__version__}  cuda={'YES' if cuda_ok else 'NO (cpu-only build)'}"
                except Exception:
                    extra = getattr(m, "__version__", "installed")
            rep.add(PASS if (mod != "torch") else (PASS if "YES" in extra else WARN),
                    f"import {label}", extra or "installed")
        except Exception:
            rep.add(INFO, f"import {label}",
                    "not installed here (expected on a laptop -- it lives in the Docker/WSL training env)")


# --------------------------------------------------------------------------- verdict
def verdict(total_vram_gb: Optional[float], docker_ok: bool, wsl_gpu: bool, vulkan_ok: bool) -> str:
    lines = ["", "  " + "=" * 74, "  VERDICT", "  " + "=" * 74]

    if not total_vram_gb:
        lines.append("  X  No CUDA GPU detected -> local IsaacLab training is NOT possible.")
        lines.append("     Use RunPod (fine_tuning/rl/runpod_setup_rl.sh) instead.")
        return "\n".join(lines)

    if wsl_gpu and not vulkan_ok:
        lines += [
            "  X  BLOCKED: WSL has CUDA but NO GPU Vulkan -> Isaac Sim can't run here yet.",
            "",
            "     Fix (Windows side):",
            "       1) Clean-reinstall the latest NVIDIA driver (Custom install -> tick 'clean install').",
            "       2) Run  wsl --shutdown  in PowerShell, reopen your terminal.",
            "       3) Verify:  wsl vulkaninfo --summary   (must list the RTX GPU, not 'llvmpipe').",
            "     Then re-run this checker; everything else below is already good to go.",
        ]
        return "\n".join(lines)

    # recommended env count from usable VRAM (leave the Isaac base + a safety margin free)
    usable = max(0.0, total_vram_gb - ISAAC_BASE_VRAM_GB - 0.7)
    rec_envs = int(usable * 1024 / PER_ENV_VRAM_MB)
    # clamp to a sane power-of-two-ish ladder
    ladder = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    rec_envs = max([e for e in ladder if e <= rec_envs] or [256])
    safe_envs = max(256, rec_envs // 2)

    can_local = (total_vram_gb >= MIN_VRAM_GB) and (docker_ok or wsl_gpu)
    if can_local:
        lines.append("  V  You CAN run the fine-tune locally (Linux/WSL2 + Docker GPU path).")
    else:
        lines.append("  !  Local training is marginal -- see warnings above.")

    lines += [
        "",
        f"     Recommended  --num-envs {rec_envs}   (fallback {safe_envs} if you hit CUDA OOM)",
        f"     RunPod used 4096 envs on 24 GB+; your {total_vram_gb:.0f} GB caps you well below that.",
        "",
        "  What this machine is good for:",
        "     * RESUMING / fine-tuning an existing checkpoint for a few thousand iters   (feasible)",
        "     * Short A/B experiments on reward / curriculum changes                     (feasible)",
        "  What to still do on RunPod:",
        "     * A from-scratch 15-25k-iter run at 4096 envs  (a laptop 8 GB GPU is too slow/small)",
        "",
        "  Before you launch:  close Chrome/other GPU apps (they eat VRAM), and keep the",
        "  laptop plugged in (battery throttles the GPU hard).",
    ]
    return "\n".join(lines)


def main() -> int:
    print()
    print("  " + "#" * 74)
    print("  #  LOCAL MACHINE READINESS -- blind-RL stair fine-tune")
    print("  " + "#" * 74)

    host = Report(); check_host(host); check_python(host)
    print("\n" + host.render("HOST"))

    gpu = Report(); total_vram = check_gpu(gpu); check_disk(gpu)
    print("\n" + gpu.render("GPU + DISK"))

    plumb = Report(); docker_ok, wsl_gpu, vulkan_ok = check_docker_wsl(plumb)
    print("\n" + plumb.render("LINUX GPU HOST (Docker / WSL2)"))

    stack = Report(); check_native_stack(stack)
    print("\n" + stack.render("TRAINING STACK (native -- usually absent on a laptop)"))

    print(verdict(total_vram, docker_ok, wsl_gpu, vulkan_ok))
    print()

    # exit non-zero only if there is NO usable GPU at all (hard blocker)
    return 0 if total_vram else 1


if __name__ == "__main__":
    raise SystemExit(main())
