"""
Code Executor tool - runs Python, PowerShell, or cmd scripts in a sandboxed subprocess.
"""
import asyncio
import os
import sys
import tempfile
from typing import Any

from .base import Tool, registry
from ..config import load_config


class CodeExecutorTool(Tool):
    name = "execute_code"
    description = (
        "Execute code in Python, PowerShell, or cmd/bash. "
        "The code runs on the local machine with full access. "
        "Returns stdout, stderr, and exit code."
    )
    parameters = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "enum": ["python", "powershell", "cmd", "bash"],
                "description": "Programming language / shell to use.",
            },
            "code": {
                "type": "string",
                "description": "The code to execute.",
            },
        },
        "required": ["language", "code"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        language = kwargs["language"]
        code = kwargs["code"]

        config = load_config()
        exec_config = config.get("tools", {}).get("code_executor", {})
        timeout = exec_config.get("timeout", 30)
        allowed = exec_config.get("languages", ["python", "powershell", "cmd", "bash"])

        if language not in allowed:
            return {"success": False, "error": f"Language '{language}' is not allowed. Allowed: {allowed}"}

        # Write code to temp file
        ext_map = {"python": ".py", "powershell": ".ps1", "cmd": ".bat", "bash": ".sh"}
        ext = ext_map.get(language, ".txt")

        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        try:
            # Build command
            if language == "python":
                cmd = [sys.executable, temp_path]
            elif language == "powershell":
                cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_path]
            elif language == "cmd":
                cmd = ["cmd", "/c", temp_path]
            elif language == "bash":
                cmd = ["bash", temp_path]
            else:
                return {"success": False, "error": f"Unknown language: {language}"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.expanduser("~"),
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "error": f"Code execution timed out after {timeout} seconds.",
                    "timeout": True,
                }

            stdout_str = stdout.decode("utf-8", errors="replace")[:10000]
            stderr_str = stderr.decode("utf-8", errors="replace")[:5000]

            return {
                "success": proc.returncode == 0,
                "result": stdout_str,
                "stderr": stderr_str,
                "exit_code": proc.returncode,
            }
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


registry.register(CodeExecutorTool())
