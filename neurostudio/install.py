#!/usr/bin/env python3
"""
NeuroForge Auto-Installer
Downloads and sets up all dependencies:
1. Creates Python virtual environment
2. Installs Python packages
3. Downloads llama.cpp prebuilt binary (Vulkan backend for AMD GPU)
4. Creates launch scripts

Usage: python install.py
"""
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "bin")
VENV_DIR = os.path.join(SCRIPT_DIR, "venv")

# llama.cpp release info - Vulkan build (works with AMD, NVIDIA, Intel GPUs)
LLAMA_CPP_VERSION = "b5023"
LLAMA_CPP_RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def print_banner():
    print("=" * 60)
    print("  NeuroForge - Auto Installer")
    print("  Local AI Agent Studio")
    print("=" * 60)
    print()


def print_step(num, total, msg):
    print(f"  [{num}/{total}] {msg}")


def is_windows():
    return platform.system() == "Windows"


def get_llama_cpp_download_url():
    """Get the latest llama.cpp release download URL for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Determine the right binary
    if system == "windows":
        # Prefer Vulkan for AMD GPU compatibility
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

    # Try to fetch latest release info
    try:
        print("    Checking latest llama.cpp release...")
        req = urllib.request.Request(
            LLAMA_CPP_RELEASES_URL,
            headers={"User-Agent": "NeuroForge-Installer/0.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read().decode())

        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if target.replace(".zip", "") in name.replace(".zip", ""):
                return asset["browser_download_url"], asset["name"]

        # Fallback: try less specific match
        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if fallback.replace(".zip", "") in name.replace(".zip", ""):
                return asset["browser_download_url"], asset["name"]

        # Last resort: try any vulkan match for Windows
        if system == "windows":
            for asset in release.get("assets", []):
                name = asset["name"].lower()
                if "vulkan" in name and name.endswith(".zip"):
                    return asset["browser_download_url"], asset["name"]

    except Exception as e:
        print(f"    Warning: Could not fetch latest release: {e}")

    return None, None


def download_file(url, dest_path):
    """Download a file with progress indication."""
    print(f"    Downloading: {url}")
    print(f"    To: {dest_path}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroForge-Installer/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(f"\r    Progress: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

            print()  # newline after progress
            return True
    except Exception as e:
        print(f"\n    Error downloading: {e}")
        return False


def step_create_venv(step, total):
    """Create Python virtual environment."""
    print_step(step, total, "Creating Python virtual environment...")

    if os.path.exists(VENV_DIR):
        print("    Virtual environment already exists. Skipping.")
        return True

    try:
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        print("    Virtual environment created.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error creating venv: {e}")
        return False


def get_venv_python():
    """Get path to Python in the virtual environment."""
    if is_windows():
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def get_venv_pip():
    """Get path to pip in the virtual environment."""
    if is_windows():
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")


def step_install_packages(step, total):
    """Install Python packages."""
    print_step(step, total, "Installing Python packages...")

    pip = get_venv_pip()
    req_file = os.path.join(SCRIPT_DIR, "requirements.txt")

    if not os.path.exists(pip):
        print("    Error: pip not found in virtual environment.")
        return False

    try:
        subprocess.run([pip, "install", "--upgrade", "pip"], check=True, capture_output=True)
        subprocess.run([pip, "install", "-r", req_file], check=True)
        print("    All packages installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error installing packages: {e}")
        return False


def step_download_llama_cpp(step, total):
    """Download llama.cpp prebuilt binary."""
    print_step(step, total, "Downloading llama.cpp (Vulkan backend for GPU acceleration)...")

    os.makedirs(BIN_DIR, exist_ok=True)

    # Check if already downloaded
    server_name = "llama-server.exe" if is_windows() else "llama-server"
    for root, dirs, files in os.walk(BIN_DIR):
        if server_name in files:
            path = os.path.join(root, server_name)
            print(f"    llama-server already found at: {path}")
            _update_config_server_path(path)
            return True

    url, filename = get_llama_cpp_download_url()
    if not url:
        print("    Warning: Could not determine download URL for llama.cpp.")
        print("    You can manually download from: https://github.com/ggml-org/llama.cpp/releases")
        print(f"    Place the extracted files in: {BIN_DIR}")
        return False

    zip_path = os.path.join(BIN_DIR, filename or "llama-cpp.zip")

    if not download_file(url, zip_path):
        return False

    # Extract
    print("    Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BIN_DIR)
        os.remove(zip_path)
        print("    Extracted successfully.")
    except Exception as e:
        print(f"    Error extracting: {e}")
        return False

    # Find llama-server binary
    for root, dirs, files in os.walk(BIN_DIR):
        if server_name in files:
            path = os.path.join(root, server_name)
            if not is_windows():
                os.chmod(path, 0o755)
            print(f"    llama-server found at: {path}")
            _update_config_server_path(path)
            return True

    print("    Warning: llama-server binary not found after extraction.")
    return False


def _update_config_server_path(path):
    """Update config.yaml with the llama-server path."""
    import yaml  # Available after pip install

    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        config.setdefault("inference", {})["llama_server_path"] = path
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"    Config updated with server path.")
    except Exception:
        # yaml might not be installed yet, write path manually
        pass


def step_create_models_dir(step, total):
    """Create models directory."""
    print_step(step, total, "Creating models directory...")

    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    gitkeep = os.path.join(models_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        with open(gitkeep, "w") as f:
            pass

    print(f"    Models directory: {models_dir}")
    print("    Place .gguf model files here, or download them from the web UI.")
    return True


def step_create_launcher(step, total):
    """Create platform-specific launch scripts."""
    print_step(step, total, "Creating launcher scripts...")

    if is_windows():
        # Windows batch launcher
        bat_path = os.path.join(SCRIPT_DIR, "NeuroForge.bat")
        python_path = get_venv_python()
        run_path = os.path.join(SCRIPT_DIR, "run.py")

        with open(bat_path, "w") as f:
            f.write(f"""@echo off
echo ============================================================
echo   NeuroForge - Local AI Agent Studio
echo   Starting server...
echo ============================================================
echo.
echo   Open your browser at: http://localhost:7860
echo   On mobile/other device: http://YOUR_IP:7860
echo.
echo   Press Ctrl+C to stop.
echo.
"{python_path}" "{run_path}"
pause
""")
        print(f"    Created: {bat_path}")
    else:
        # Unix shell launcher
        sh_path = os.path.join(SCRIPT_DIR, "neurostudio.sh")
        python_path = get_venv_python()
        run_path = os.path.join(SCRIPT_DIR, "run.py")

        with open(sh_path, "w") as f:
            f.write(f"""#!/bin/bash
echo "============================================================"
echo "  NeuroForge - Local AI Agent Studio"
echo "  Starting server..."
echo "============================================================"
echo ""
echo "  Open your browser at: http://localhost:7860"
echo "  On mobile/other device: http://YOUR_IP:7860"
echo ""
echo "  Press Ctrl+C to stop."
echo ""
"{python_path}" "{run_path}"
""")
        os.chmod(sh_path, 0o755)
        print(f"    Created: {sh_path}")

    return True


def main():
    print_banner()

    total_steps = 5
    results = []

    results.append(step_create_venv(1, total_steps))
    results.append(step_install_packages(2, total_steps))
    results.append(step_download_llama_cpp(3, total_steps))
    results.append(step_create_models_dir(4, total_steps))
    results.append(step_create_launcher(5, total_steps))

    print()
    print("=" * 60)
    if all(results):
        print("  Installation complete!")
        print()
        if is_windows():
            print("  To start NeuroForge:")
            print(f"    Double-click: NeuroForge.bat")
            print(f"    Or run: python run.py")
        else:
            print("  To start NeuroForge:")
            print(f"    Run: ./neurostudio.sh")
            print(f"    Or: python run.py")
        print()
        print("  Then open: http://localhost:7860")
    else:
        print("  Installation completed with warnings.")
        print("  Some steps may have failed - check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
