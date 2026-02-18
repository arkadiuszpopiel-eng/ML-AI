"""
System monitor - real-time CPU/RAM/GPU stats via WebSocket.
"""
import asyncio
import json
import logging
import platform
import shutil
import subprocess
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("neurostudio.monitor")


def get_cpu_info() -> dict:
    """Get CPU usage and info."""
    if not psutil:
        return {"error": "psutil not installed"}

    return {
        "percent": psutil.cpu_percent(interval=0),
        "count": psutil.cpu_count(),
        "count_logical": psutil.cpu_count(logical=True),
        "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None,
    }


def get_ram_info() -> dict:
    """Get RAM usage."""
    if not psutil:
        return {"error": "psutil not installed"}

    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "used_gb": round(mem.used / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
        "percent": mem.percent,
    }


def get_disk_info() -> dict:
    """Get disk usage for the main partition."""
    if not psutil:
        return {"error": "psutil not installed"}

    disk = psutil.disk_usage("/")
    return {
        "total_gb": round(disk.total / (1024**3), 1),
        "used_gb": round(disk.used / (1024**3), 1),
        "free_gb": round(disk.free / (1024**3), 1),
        "percent": disk.percent,
    }


def get_gpu_info_amd() -> Optional[dict]:
    """Try to get AMD GPU info via rocm-smi."""
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return None

    try:
        result = subprocess.run(
            [rocm_smi, "--showuse", "--showmeminfo", "vram", "--showtemp", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Parse rocm-smi JSON (format varies by version)
            gpu_data = {}
            for key, val in data.items():
                if key.startswith("card"):
                    gpu_data = val
                    break

            return {
                "vendor": "AMD",
                "temperature_c": gpu_data.get("Temperature (Sensor edge) (C)",
                                               gpu_data.get("temperature", "N/A")),
                "gpu_use_percent": gpu_data.get("GPU use (%)",
                                                 gpu_data.get("gpu_use", "N/A")),
                "vram_used_mb": gpu_data.get("VRAM Total Used Memory (B)", "N/A"),
                "vram_total_mb": gpu_data.get("VRAM Total Memory (B)", "N/A"),
            }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass

    # Fallback: plain text parsing
    try:
        result = subprocess.run(
            [rocm_smi], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return {"vendor": "AMD", "raw": result.stdout[:500]}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_gpu_info_nvidia() -> Optional[dict]:
    """Try to get NVIDIA GPU info via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5:
                return {
                    "vendor": "NVIDIA",
                    "name": parts[0].strip(),
                    "temperature_c": int(parts[1].strip()),
                    "gpu_use_percent": int(parts[2].strip()),
                    "vram_used_mb": int(parts[3].strip()),
                    "vram_total_mb": int(parts[4].strip()),
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return None


def get_gpu_info() -> dict:
    """Get GPU info (tries AMD first for RX 9070 XT, then NVIDIA)."""
    info = get_gpu_info_amd()
    if info:
        return info

    info = get_gpu_info_nvidia()
    if info:
        return info

    return {"vendor": "unknown", "message": "No GPU monitoring tool found (rocm-smi / nvidia-smi)"}


def get_system_snapshot() -> dict:
    """Get a complete system snapshot."""
    return {
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def get_process_info() -> list[dict]:
    """Get top processes by CPU/memory."""
    if not psutil:
        return []

    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            if info["cpu_percent"] > 0 or info["memory_percent"] > 0.5:
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": round(info["cpu_percent"], 1),
                    "mem": round(info["memory_percent"], 1),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda p: p["cpu"] + p["mem"], reverse=True)
    return procs[:15]
