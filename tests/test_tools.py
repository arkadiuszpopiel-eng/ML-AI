"""Tests for agent tools: filesystem, shell, base registry."""
import os
import pytest

from neurostudio.backend.tools.base import Tool, ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()

        class DummyTool(Tool):
            name = "dummy"
            description = "A dummy tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return {"success": True}

        tool = DummyTool()
        reg.register(tool)

        assert reg.get("dummy") is tool
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = ToolRegistry()

        class T1(Tool):
            name = "t1"
            description = "Tool 1"
            parameters = {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return {"success": True}

        class T2(Tool):
            name = "t2"
            description = "Tool 2"
            parameters = {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return {"success": True}

        reg.register(T1())
        reg.register(T2())

        tools = reg.list_tools()
        assert len(tools) == 2

    def test_to_openai_tools_format(self):
        reg = ToolRegistry()

        class DummyTool(Tool):
            name = "test_tool"
            description = "Does testing"
            parameters = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
            async def execute(self, **kwargs):
                return {"success": True}

        reg.register(DummyTool())
        openai_tools = reg.to_openai_tools()

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "test_tool"
        assert openai_tools[0]["function"]["description"] == "Does testing"


class TestFilesystemTools:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        from neurostudio.backend.tools.filesystem import ReadFileTool

        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, World!\nLine 2\n", encoding="utf-8")

        tool = ReadFileTool()
        result = await tool.execute(path=str(test_file))

        assert result["success"] is True
        assert "Hello, World!" in result["result"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        from neurostudio.backend.tools.filesystem import ReadFileTool

        tool = ReadFileTool()
        result = await tool.execute(path=str(tmp_path / "nonexistent.txt"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        from neurostudio.backend.tools.filesystem import WriteFileTool

        target = tmp_path / "output.txt"
        tool = WriteFileTool()
        result = await tool.execute(path=str(target), content="Test content")

        assert result["success"] is True
        assert target.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_file_append(self, tmp_path):
        from neurostudio.backend.tools.filesystem import WriteFileTool

        target = tmp_path / "append.txt"
        target.write_text("Line 1\n", encoding="utf-8")

        tool = WriteFileTool()
        result = await tool.execute(path=str(target), content="Line 2\n", append=True)

        assert result["success"] is True
        assert target.read_text() == "Line 1\nLine 2\n"

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        from neurostudio.backend.tools.filesystem import ListDirectoryTool

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        tool = ListDirectoryTool()
        result = await tool.execute(path=str(tmp_path))

        assert result["success"] is True
        names = {item["name"] for item in result["result"]}
        assert "a.txt" in names
        assert "b.py" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_search_files(self, tmp_path):
        from neurostudio.backend.tools.filesystem import SearchFilesTool

        (tmp_path / "code.py").write_text("def hello():\n    print('world')\n")
        (tmp_path / "other.txt").write_text("nothing here\n")

        tool = SearchFilesTool()
        result = await tool.execute(path=str(tmp_path), query="hello")

        assert result["success"] is True
        assert result["count"] >= 1
        assert any("hello" in r["content"] for r in result["result"])

    @pytest.mark.asyncio
    async def test_blocked_path(self, tmp_path):
        from neurostudio.backend.tools.filesystem import ReadFileTool

        tool = ReadFileTool()
        result = await tool.execute(path="/etc/passwd")

        assert result["success"] is False
        assert "denied" in result["error"].lower()


class TestPathSecurity:
    def test_is_path_under(self):
        from neurostudio.backend.tools.filesystem import _is_path_under

        # Positive cases
        assert _is_path_under("/home/user/file.txt", "/home/user") is True
        assert _is_path_under("/home/user/sub/file.txt", "/home/user") is True
        assert _is_path_under("/home/user", "/home/user") is True

        # Negative cases - the old startswith bug
        assert _is_path_under("/etc.backup/file", "/etc") is False
        assert _is_path_under("/etcfoo/bar", "/etc") is False
        assert _is_path_under("/home/other/file.txt", "/home/user") is False


class TestShellTool:
    @pytest.mark.asyncio
    async def test_run_simple_command(self, tmp_path):
        from neurostudio.backend.tools.shell import ShellTool

        tool = ShellTool()
        result = await tool.execute(command="echo hello", working_dir=str(tmp_path))

        assert result["success"] is True
        assert "hello" in result["result"]

    @pytest.mark.asyncio
    async def test_blocked_command(self, tmp_path):
        from neurostudio.backend.tools.shell import ShellTool

        tool = ShellTool()
        result = await tool.execute(command="rm -rf /", working_dir=str(tmp_path))

        assert result["success"] is False
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_blocked_command_with_extra_spaces(self, tmp_path):
        from neurostudio.backend.tools.shell import ShellTool

        tool = ShellTool()
        # Extra spaces should still be caught after normalization
        result = await tool.execute(command="rm  -rf  /", working_dir=str(tmp_path))

        assert result["success"] is False
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_command_exit_code(self, tmp_path):
        from neurostudio.backend.tools.shell import ShellTool

        tool = ShellTool()
        result = await tool.execute(command="false", working_dir=str(tmp_path))

        assert result["success"] is False
        assert result["exit_code"] != 0
