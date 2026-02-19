"""Tests for configuration management."""
from neurostudio.backend.config import load_config, save_config, get_models_dir


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        config = load_config()
        assert isinstance(config, dict)
        assert "server" in config

    def test_returns_dict(self, tmp_path):
        config = load_config()
        assert isinstance(config, dict)


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path):
        config = load_config()
        config["test_key"] = "test_value"
        save_config(config)

        reloaded = load_config()
        assert reloaded["test_key"] == "test_value"


class TestGetModelsDir:
    def test_creates_directory(self, tmp_path):
        models_dir = get_models_dir()
        assert models_dir.exists()
        assert models_dir.is_dir()
