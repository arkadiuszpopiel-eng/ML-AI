"""Tests for engine setup and installation."""
import os
from unittest.mock import patch, MagicMock
import pytest

from neurostudio.backend.setup import get_engine_status, _get_download_url, _download_file, _strip_archive_ext


class TestGetEngineStatus:
    def test_returns_dict_with_required_keys(self):
        status = get_engine_status()
        assert isinstance(status, dict)
        assert "installed" in status
        assert "server_path" in status
        assert "bin_dir" in status
        assert "platform" in status
        assert "arch" in status

    def test_installed_is_bool(self):
        status = get_engine_status()
        assert isinstance(status["installed"], bool)

    @patch("neurostudio.backend.config.get_llama_server_path", return_value=None)
    def test_not_installed_when_no_path(self, mock_path):
        status = get_engine_status()
        assert status["installed"] is False
        assert status["server_path"] is None

    @patch("neurostudio.backend.config.get_llama_server_path", return_value="/usr/bin/llama-server")
    def test_installed_when_path_exists(self, mock_path):
        status = get_engine_status()
        assert status["installed"] is True
        assert status["server_path"] == "/usr/bin/llama-server"

    def test_platform_is_string(self):
        status = get_engine_status()
        assert isinstance(status["platform"], str)
        assert len(status["platform"]) > 0


class TestStripArchiveExt:
    def test_strip_zip(self):
        assert _strip_archive_ext("file.zip") == "file"

    def test_strip_tar_gz(self):
        assert _strip_archive_ext("file.tar.gz") == "file"

    def test_strip_tar_xz(self):
        assert _strip_archive_ext("file.tar.xz") == "file"

    def test_strip_tar_bz2(self):
        assert _strip_archive_ext("file.tar.bz2") == "file"

    def test_no_ext(self):
        assert _strip_archive_ext("file") == "file"

    def test_real_llama_filename_zip(self):
        assert _strip_archive_ext("llama-b8117-bin-win-vulkan-x64.zip") == "llama-b8117-bin-win-vulkan-x64"

    def test_real_llama_filename_targz(self):
        assert _strip_archive_ext("llama-b8117-bin-ubuntu-vulkan-x64.tar.gz") == "llama-b8117-bin-ubuntu-vulkan-x64"


def _mock_github_api(assets):
    """Helper to mock GitHub API response with given assets."""
    import json
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps({"assets": assets}).encode()
    return mock_resp


class TestGetDownloadUrl:
    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_returns_tuple(self, mock_urlopen):
        mock_urlopen.return_value = _mock_github_api([])

        url, filename = _get_download_url()
        assert url is None
        assert filename is None

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_finds_linux_asset_zip(self, mock_urlopen):
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b1234-bin-ubuntu-x64.zip", "browser_download_url": "https://example.com/ubuntu.zip"}
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Linux"):
            url, filename = _get_download_url()
            assert url == "https://example.com/ubuntu.zip"

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_finds_linux_asset_targz(self, mock_urlopen):
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b8117-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://example.com/ubuntu.tar.gz"}
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Linux"):
            url, filename = _get_download_url()
            assert url == "https://example.com/ubuntu.tar.gz"

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_windows_picks_win_not_ubuntu(self, mock_urlopen):
        """Critical test: Windows must NOT match ubuntu-vulkan-x64."""
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b8117-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://example.com/ubuntu.tar.gz"},
            {"name": "llama-b8117-bin-win-vulkan-x64.zip", "browser_download_url": "https://example.com/win.zip"},
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Windows"), \
             patch("neurostudio.backend.setup.platform.machine", return_value="AMD64"):
            url, filename = _get_download_url()
            assert url == "https://example.com/win.zip"
            assert "win" in filename.lower()

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_windows_picks_win_even_if_ubuntu_first(self, mock_urlopen):
        """Ubuntu asset listed first should NOT be picked on Windows."""
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b8117-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://example.com/ubuntu.tar.gz"},
            {"name": "llama-b8117-bin-win-vulkan-x64.zip", "browser_download_url": "https://example.com/win.zip"},
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Windows"), \
             patch("neurostudio.backend.setup.platform.machine", return_value="AMD64"):
            url, filename = _get_download_url()
            assert "ubuntu" not in url
            assert "win" in url

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_linux_picks_ubuntu_not_win(self, mock_urlopen):
        """Linux must pick ubuntu, not win."""
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b8117-bin-win-vulkan-x64.zip", "browser_download_url": "https://example.com/win.zip"},
            {"name": "llama-b8117-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "https://example.com/ubuntu.tar.gz"},
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Linux"), \
             patch("neurostudio.backend.setup.platform.machine", return_value="x86_64"):
            url, filename = _get_download_url()
            assert "ubuntu" in url
            assert "win" not in url

    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_macos_arm64(self, mock_urlopen):
        mock_urlopen.return_value = _mock_github_api([
            {"name": "llama-b8117-bin-macos-arm64.zip", "browser_download_url": "https://example.com/macos-arm64.zip"},
            {"name": "llama-b8117-bin-macos-x64.zip", "browser_download_url": "https://example.com/macos-x64.zip"},
        ])
        with patch("neurostudio.backend.setup.platform.system", return_value="Darwin"), \
             patch("neurostudio.backend.setup.platform.machine", return_value="arm64"):
            url, filename = _get_download_url()
            assert "arm64" in url

    @patch("neurostudio.backend.setup.urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_returns_none_on_network_error(self, mock_urlopen):
        url, filename = _get_download_url()
        assert url is None
        assert filename is None


class TestDownloadFile:
    @patch("neurostudio.backend.setup.urllib.request.urlopen")
    def test_downloads_to_file(self, mock_urlopen, tmp_path):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"Content-Length": "5"}
        mock_resp.read.side_effect = [b"hello", b""]
        mock_urlopen.return_value = mock_resp

        dest = str(tmp_path / "test.zip")
        result = _download_file("https://example.com/test.zip", dest)
        assert result is True
        assert os.path.exists(dest)

    @patch("neurostudio.backend.setup.urllib.request.urlopen", side_effect=Exception("fail"))
    def test_returns_false_on_error(self, mock_urlopen, tmp_path):
        dest = str(tmp_path / "test.zip")
        result = _download_file("https://example.com/test.zip", dest)
        assert result is False


class TestInstallEngine:
    @pytest.mark.asyncio
    @patch("neurostudio.backend.setup.get_engine_status")
    async def test_returns_early_if_already_installed(self, mock_status):
        mock_status.return_value = {
            "installed": True,
            "server_path": "/path/to/llama-server",
            "bin_dir": "/path/to/bin",
            "platform": "linux",
            "arch": "x86_64",
        }
        from neurostudio.backend.setup import install_engine
        result = await install_engine()
        assert result["success"] is True
        assert result["already_installed"] is True

    @pytest.mark.asyncio
    @patch("neurostudio.backend.setup.get_engine_status")
    @patch("neurostudio.backend.setup._get_download_url", return_value=(None, None))
    async def test_fails_when_no_download_url(self, mock_url, mock_status):
        mock_status.return_value = {
            "installed": False,
            "server_path": None,
            "bin_dir": "/tmp/bin",
            "platform": "linux",
            "arch": "x86_64",
        }
        from neurostudio.backend.setup import install_engine
        result = await install_engine()
        assert result["success"] is False
        assert "Nie udalo" in result["message"]
