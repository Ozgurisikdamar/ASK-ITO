"""Hardware / offload diagnostics (used by scripts/hw_diag.py; never by requests).

Reports what is actually present and in use rather than what is configured:
CPU, RAM, GPUs found by the vendor tools (nvidia-smi, rocm-smi) or the platform
(Apple Silicon), and, from the running Ollama, how much of each loaded model
sits in VRAM (`/api/ps` size_vram vs size). The verdict is one of:
  cpu_only            no GPU found; inference runs on the CPU
  gpu_present_unused  a GPU exists but no loaded model uses VRAM
  gpu_partial         part of the model is in VRAM (layer split CPU/GPU)
  gpu_full            the whole loaded model is in VRAM
  unknown             no model loaded / Ollama unreachable, GPU state only
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_ram() -> dict:
    cpu, flags = platform.processor() or "unknown", []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                flags = line.split(":", 1)[1].split()
                break
    except OSError:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    simd = [f for f in ("avx2", "avx512f", "avx512_vnni", "amx_tile") if f in flags]
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": ram_gb, "simd": simd,
            "os": f"{platform.system()} {platform.release()}", "arch": platform.machine()}


def nvidia_gpus(run=_run) -> list[dict]:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
               "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"vendor": "nvidia", "name": parts[0],
                         "vram_total_mb": int(float(parts[1])),
                         "vram_used_mb": int(float(parts[2])),
                         "driver": parts[3] if len(parts) > 3 else None})
    return gpus


def amd_gpus(run=_run) -> list[dict]:
    out = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if not out or "GPU" not in out:
        return []
    rows = [ln for ln in out.splitlines()[1:] if ln.strip()]
    return [{"vendor": "amd", "raw": ln.strip()} for ln in rows]


def apple_gpu() -> list[dict]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return [{"vendor": "apple", "name": "Apple Silicon (unified memory, Metal)"}]
    return []


def gpus(run=_run) -> list[dict]:
    return nvidia_gpus(run) + amd_gpus(run) + apple_gpu()


def device_nodes() -> list[str]:
    return sorted(str(p) for pat in ("/dev/nvidia*", "/dev/kfd", "/dev/dri/renderD*")
                  for p in Path("/").glob(pat.lstrip("/")))


def ollama_offload(ps_models: list[dict]) -> list[dict]:
    out = []
    for m in ps_models:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        out.append({"model": m.get("name"), "size_mb": round(size / 2**20),
                    "vram_mb": round(vram / 2**20),
                    "vram_fraction": round(vram / size, 3) if size else None,
                    "quantization": (m.get("details") or {}).get("quantization_level")})
    return out


def verdict(found_gpus: list[dict], offload: list[dict]) -> str:
    if offload:
        frac = max((o["vram_fraction"] or 0) for o in offload)
        if frac >= 0.999:
            return "gpu_full"
        if frac > 0:
            return "gpu_partial"
        return "gpu_present_unused" if found_gpus else "cpu_only"
    return "unknown" if found_gpus else "cpu_only"


def diagnose(ps_models: list[dict] | None, run=_run) -> dict:
    found = gpus(run)
    offload = ollama_offload(ps_models or [])
    return {"host": cpu_ram(), "gpus": found, "device_nodes": device_nodes(),
            "ollama_loaded": offload, "verdict": verdict(found, offload)}
