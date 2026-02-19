"""Shared test fixtures for NeuroForge tests."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path so we can import neurostudio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _patch_base_dir(monkeypatch, tmp_path):
    """Redirect all data operations to a temp directory per test."""
    import neurostudio.backend.config as config_mod
    import neurostudio.backend.storage as storage_mod
    import neurostudio.backend.rag.engine as rag_mod
    import neurostudio.backend.templates as templates_mod

    # Create a minimal config.yaml in tmp
    test_config = {
        "server": {"host": "127.0.0.1", "port": 7860},
        "inference": {"context_size": 4096, "gpu_layers": 0, "threads": 4},
        "tools": {
            "filesystem": {
                "allowed_dirs": [str(tmp_path)],
                "blocked_dirs": ["/etc", "/usr", "/sys"],
            },
            "shell": {
                "timeout": 10,
                "blocked_commands": ["rm -rf /", "format", "shutdown", "reboot"],
            },
        },
        "agent": {
            "system_prompt": "You are a test assistant.",
            "max_tool_calls": 5,
        },
    }

    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(test_config), encoding="utf-8")

    # Patch BASE_DIR and CONFIG_PATH
    monkeypatch.setattr(config_mod, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", config_path)

    # Patch module-level paths that derive from BASE_DIR
    monkeypatch.setattr(storage_mod, "HISTORY_DIR", tmp_path / "data" / "conversations")
    monkeypatch.setattr(rag_mod, "DOCS_DIR", tmp_path / "data" / "documents")
    monkeypatch.setattr(rag_mod, "INDEX_PATH", tmp_path / "data" / "rag_index.json")
    monkeypatch.setattr(templates_mod, "TEMPLATES_PATH", tmp_path / "data" / "templates.json")
