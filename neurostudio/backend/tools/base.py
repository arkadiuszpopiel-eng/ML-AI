"""
Base class for all tools available to the AI agent.
Each tool provides a name, description, parameter schema, and execute method.
"""
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does (shown to the model)."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with given parameters.
        Returns a dict with at least 'success' (bool) and 'result' or 'error' keys.
        """

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_tool() for tool in self._tools.values()]


# Global registry
registry = ToolRegistry()
