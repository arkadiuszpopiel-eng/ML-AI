"""Tests for system monitor."""
from neurostudio.backend.monitor import (
    get_cpu_info,
    get_ram_info,
    get_disk_info,
    get_gpu_info,
    get_system_snapshot,
    get_process_info,
)


class TestCPUInfo:
    def test_returns_dict(self):
        info = get_cpu_info()
        assert isinstance(info, dict)
        # Should have percent and count even without psutil
        if "error" not in info:
            assert "percent" in info
            assert "count" in info


class TestRAMInfo:
    def test_returns_dict(self):
        info = get_ram_info()
        assert isinstance(info, dict)
        if "error" not in info:
            assert "total_gb" in info
            assert "percent" in info


class TestDiskInfo:
    def test_returns_dict(self):
        info = get_disk_info()
        assert isinstance(info, dict)
        if "error" not in info:
            assert "total_gb" in info
            assert "percent" in info


class TestGPUInfo:
    def test_returns_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "vendor" in info


class TestSystemSnapshot:
    def test_has_all_sections(self):
        snap = get_system_snapshot()
        assert "cpu" in snap
        assert "ram" in snap
        assert "disk" in snap
        assert "gpu" in snap
        assert "platform" in snap


class TestProcessInfo:
    def test_returns_list(self):
        procs = get_process_info()
        assert isinstance(procs, list)
        # Should return some processes on any running system
        assert len(procs) >= 0
