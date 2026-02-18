"""
Shell tool - execute system commands directly.
"""
import asyncio
import os
import sys
from typing import Any

from .base import Tool, registry
from ..config import load_config


class ShellTool(Tool):
    name = "run_command"
    description = (
        "Execute a shell command on the local system. "
        "On Windows this uses cmd.exe, on Linux/Mac it uses bash. "
        "Returns stdout, stderr, and exit code. "
        "Use for: git, npm, pip, system utilities, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command (default: user home).",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        command = kwargs["command"]
        working_dir = kwargs.get("working_dir", os.path.expanduser("~"))

        config = load_config()
        shell_config = config.get("tools", {}).get("shell", {})
        timeout = shell_config.get("timeout", 30)
        blocked = shell_config.get("blocked_commands", [])

        # Safety check
        cmd_lower = command.lower().strip()
        for blocked_cmd in blocked:
            if blocked_cmd.lower() in cmd_lower:
                return {"success": False, "error": f"Command blocked for safety: contains '{blocked_cmd}'"}

        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "bash", "-c", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds.",
                    "timeout": True,
                }

            stdout_str = stdout.decode("utf-8", errors="replace")[:10000]
            stderr_str = stderr.decode("utf-8", errors="replace")[:5000]

            return {
                "success": proc.returncode == 0,
                "result": stdout_str,
                "stderr": stderr_str,
                "exit_code": proc.returncode,
                "command": command,
            }
        except Exception as e:
            return {"success": False, "error": f"Command execution failed: {str(e)}"}


class ProcessManagerTool(Tool):
    name = "manage_process"
    description = (
        "List running processes or get system resource usage. "
        "Useful for monitoring system state during AI workloads."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_top", "system_info", "gpu_info"],
                "description": "Action: list_top (top processes by CPU/RAM), system_info (CPU/RAM/disk), gpu_info (GPU utilization).",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        import psutil

        action = kwargs["action"]

        try:
            if action == "list_top":
                procs = []
                for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                    try:
                        info = p.info
                        procs.append({
                            "pid": info["pid"],
                            "name": info["name"],
                            "cpu_percent": info["cpu_percent"] or 0,
                            "memory_mb": round((info["memory_info"].rss if info["memory_info"] else 0) / 1048576, 1),
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                procs.sort(key=lambda x: x["memory_mb"], reverse=True)
                return {"success": True, "result": procs[:20]}

            elif action == "system_info":
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                return {
                    "success": True,
                    "result": {
                        "cpu_percent": cpu,
                        "cpu_count": psutil.cpu_count(),
                        "ram_total_gb": round(mem.total / (1024**3), 1),
                        "ram_used_gb": round(mem.used / (1024**3), 1),
                        "ram_percent": mem.percent,
                        "disk_total_gb": round(disk.total / (1024**3), 1),
                        "disk_used_gb": round(disk.used / (1024**3), 1),
                        "disk_percent": disk.percent,
                    },
                }

            elif action == "gpu_info":
                # Try to get GPU info (works if vulkan-tools or nvidia-smi available)
                info = "GPU info not available via psutil. Use 'run_command' with 'vulkaninfo' or 'nvidia-smi' for details."
                return {"success": True, "result": info}

            return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


registry.register(ShellTool())
registry.register(ProcessManagerTool())
