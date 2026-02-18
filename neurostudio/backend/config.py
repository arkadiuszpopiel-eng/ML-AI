"""Configuration management for NeuroForge."""
import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: dict):
    """Save configuration to YAML file."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_models_dir() -> Path:
    """Get the models directory path."""
    config = load_config()
    models_dir = config.get("models", {}).get("directory", "models")
    path = BASE_DIR / models_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llama_server_path() -> str | None:
    """Get path to llama-server binary."""
    config = load_config()
    path = config.get("inference", {}).get("llama_server_path")
    if path and os.path.exists(path):
        return path
    # Try auto-detection in common locations
    for candidate in [
        BASE_DIR / "bin" / "llama-server.exe",
        BASE_DIR / "bin" / "llama-server",
        BASE_DIR / "bin" / "llama" / "llama-server.exe",
        BASE_DIR / "bin" / "llama" / "llama-server",
    ]:
        if candidate.exists():
            return str(candidate)
    return None
