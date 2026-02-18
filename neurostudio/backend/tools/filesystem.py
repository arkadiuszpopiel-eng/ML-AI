"""
Filesystem tools - read, write, list, search files on the local machine.
"""
import os
import glob as globmod
from pathlib import Path
from typing import Any

from .base import Tool, registry
from ..config import load_config


def _check_path_allowed(path: str) -> str | None:
    """Check if a path is allowed by configuration. Returns error message or None."""
    config = load_config()
    fs_config = config.get("tools", {}).get("filesystem", {})
    blocked = fs_config.get("blocked_dirs", [])
    abs_path = os.path.abspath(path)

    for blocked_dir in blocked:
        if abs_path.startswith(os.path.abspath(blocked_dir)):
            return f"Access denied: {blocked_dir} is a blocked directory."

    allowed = fs_config.get("allowed_dirs", [])
    if allowed:
        for allowed_dir in allowed:
            if abs_path.startswith(os.path.abspath(allowed_dir)):
                return None
        return f"Access denied: path is outside allowed directories."
    return None


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the file content as text."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 500).",
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        path = kwargs["path"]
        max_lines = kwargs.get("max_lines", 500)

        error = _check_path_allowed(path)
        if error:
            return {"success": False, "error": error}

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)
            return {"success": True, "result": "".join(lines), "path": os.path.abspath(path)}
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append to file instead of overwriting (default: false).",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        path = kwargs["path"]
        content = kwargs["content"]
        append = kwargs.get("append", False)

        error = _check_path_allowed(path)
        if error:
            return {"success": False, "error": error}

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "result": f"File written: {os.path.abspath(path)}", "bytes": len(content.encode())}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and directories in a given path. Shows names, sizes, and types."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory).",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter results (e.g. '*.py', '**/*.txt').",
            },
        },
        "required": [],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        path = kwargs.get("path", ".")
        pattern = kwargs.get("pattern")

        error = _check_path_allowed(path)
        if error:
            return {"success": False, "error": error}

        try:
            if pattern:
                full_pattern = os.path.join(path, pattern)
                entries = globmod.glob(full_pattern, recursive=True)
                items = []
                for entry in sorted(entries)[:200]:
                    stat = os.stat(entry)
                    items.append({
                        "name": os.path.relpath(entry, path),
                        "type": "dir" if os.path.isdir(entry) else "file",
                        "size": stat.st_size,
                    })
            else:
                items = []
                for entry in sorted(os.listdir(path))[:200]:
                    full = os.path.join(path, entry)
                    try:
                        stat = os.stat(full)
                        items.append({
                            "name": entry,
                            "type": "dir" if os.path.isdir(full) else "file",
                            "size": stat.st_size,
                        })
                    except OSError:
                        items.append({"name": entry, "type": "unknown", "size": 0})

            return {"success": True, "result": items, "path": os.path.abspath(path), "count": len(items)}
        except FileNotFoundError:
            return {"success": False, "error": f"Directory not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search for text content within files. Returns matching lines with file paths and line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in.",
            },
            "query": {
                "type": "string",
                "description": "Text or regex pattern to search for.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to search (e.g. '*.py'). Default: all files.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default: 50).",
            },
        },
        "required": ["path", "query"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        import re

        path = kwargs["path"]
        query = kwargs["query"]
        file_pattern = kwargs.get("file_pattern", "*")
        max_results = kwargs.get("max_results", 50)

        error = _check_path_allowed(path)
        if error:
            return {"success": False, "error": error}

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        results = []
        search_pattern = os.path.join(path, "**", file_pattern)
        for filepath in globmod.glob(search_pattern, recursive=True):
            if os.path.isdir(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append({
                                "file": os.path.relpath(filepath, path),
                                "line": line_num,
                                "content": line.rstrip()[:200],
                            })
                            if len(results) >= max_results:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(results) >= max_results:
                break

        return {"success": True, "result": results, "count": len(results)}


# Register all filesystem tools
for tool_class in [ReadFileTool, WriteFileTool, ListDirectoryTool, SearchFilesTool]:
    registry.register(tool_class())
