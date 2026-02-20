"""
Setup utilities - automated download and installation of llama.cpp engine.
Can be triggered from the web UI so users don't need to run install.py manually.
"""
import asyncio
import json
import logging
import os
import platform
import stat
import urllib.request
import zipfile
from pathlib import Path

from .config import load_config, save_config, BASE_DIR

logger = logging.getLogger("neurostudio.setup")

BIN_DIR = BASE_DIR / "bin"
LLAMA_RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def get_engine_status() -> dict:
    """Check if llama-server binary is installed and ready."""
    from .config import get_llama_server_path

    server_path = get_llama_server_path()
    installed = server_path is not None

    return {
        "installed": installed,
        "server_path": server_path,
        "bin_dir": str(BIN_DIR),
        "platform": platform.system().lower(),
        "arch": platform.machine().lower(),
    }


def _get_download_url() -> tuple[str | None, str | None]:
    """Determine the correct llama.cpp download URL for this platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        target = "vulkan-x64.zip"
        fallback = "win-avx2-x64.zip"
    elif system == "linux":
        target = "ubuntu-x64.zip"
        fallback = target
    elif system == "darwin":
        target = "macos-arm64.zip" if "arm" in machine else "macos-x64.zip"
        fallback = target
    else:
        return None, None

    try:
        req = urllib.request.Request(
            LLAMA_RELEASES_URL,
            headers={"User-Agent": "NeuroForge-Setup/0.4"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read().decode())

        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if target.replace(".zip", "") in name.replace(".zip", ""):
                return asset["browser_download_url"], asset["name"]

        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if fallback.replace(".zip", "") in name.replace(".zip", ""):
                return asset["browser_download_url"], asset["name"]

        if system == "windows":
            for asset in release.get("assets", []):
                name = asset["name"].lower()
                if "vulkan" in name and name.endswith(".zip"):
                    return asset["browser_download_url"], asset["name"]

    except Exception as e:
        logger.error("Failed to fetch llama.cpp release info: %s", e)

    return None, None


async def install_engine(progress_callback=None) -> dict:
    """Download and install llama-server binary.

    Args:
        progress_callback: optional async callable(stage, pct, message)

    Returns:
        dict with 'success', 'server_path', 'message' keys.
    """

    async def report(stage: str, pct: int, msg: str):
        if progress_callback:
            await progress_callback(stage, pct, msg)

    await report("check", 0, "Sprawdzanie platformy...")

    # Check if already installed
    status = get_engine_status()
    if status["installed"]:
        return {
            "success": True,
            "server_path": status["server_path"],
            "message": "llama-server juz zainstalowany.",
            "already_installed": True,
        }

    await report("download", 5, "Pobieranie informacji o najnowszej wersji...")

    url, filename = await asyncio.to_thread(_get_download_url)
    if not url:
        return {
            "success": False,
            "server_path": None,
            "message": "Nie udalo sie znalezc adresu pobierania llama.cpp dla tej platformy.",
        }

    await report("download", 10, f"Pobieranie {filename}...")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BIN_DIR / (filename or "llama-cpp.zip")

    # Download in a thread to avoid blocking
    try:
        download_result = await asyncio.to_thread(_download_file, str(url), str(zip_path))
        if not download_result:
            return {
                "success": False,
                "server_path": None,
                "message": "Pobieranie nie powiodlo sie. Sprawdz polaczenie internetowe.",
            }
    except Exception as e:
        return {
            "success": False,
            "server_path": None,
            "message": f"Blad pobierania: {e}",
        }

    await report("extract", 70, "Rozpakowywanie...")

    # Extract
    try:
        def _extract():
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(BIN_DIR))
            os.remove(str(zip_path))

        await asyncio.to_thread(_extract)
    except Exception as e:
        return {
            "success": False,
            "server_path": None,
            "message": f"Blad rozpakowywania: {e}",
        }

    await report("configure", 85, "Konfigurowanie...")

    # Find llama-server binary
    server_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    server_path = None

    for root, dirs, files in os.walk(str(BIN_DIR)):
        if server_name in files:
            server_path = os.path.join(root, server_name)
            if platform.system() != "Windows":
                os.chmod(server_path, os.stat(server_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            break

    if not server_path:
        return {
            "success": False,
            "server_path": None,
            "message": "Rozpakowywanie zakonczone, ale binarka llama-server nie znaleziona.",
        }

    # Update config
    config = load_config()
    config.setdefault("inference", {})["llama_server_path"] = server_path
    save_config(config)

    await report("done", 100, "Instalacja zakonczona!")

    logger.info("llama-server installed at: %s", server_path)

    return {
        "success": True,
        "server_path": server_path,
        "message": "llama-server zainstalowany pomyslnie!",
    }


def _download_file(url: str, dest: str) -> bool:
    """Download a file (blocking, run in thread)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroForge-Setup/0.4"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info("Downloaded %s (%d bytes)", dest, downloaded)
        return True
    except Exception as e:
        logger.error("Download error: %s", e)
        return False
