"""
AnthropicProvider - connects to Anthropic API (Claude Sonnet, Haiku, Opus).
Translates between OpenAI format (used internally) and Anthropic Messages API.
"""
import logging
import httpx
from .providers import BaseProvider

logger = logging.getLogger("neurostudio.providers.anthropic")

ANTHROPIC_MODELS = [
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "context": "200K"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "context": "200K"},
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "context": "200K"},
]


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API."""

    provider_id = "anthropic"
    provider_name = "Anthropic (Claude)"
    requires_api_key = True

    API_BASE = "https://api.anthropic.com/v1"

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format.
        Returns (system_prompt, messages_without_system).
        """
        system = None
        converted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role == "user":
                converted.append({"role": "user", "content": content})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content or ""})
            elif role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }],
                })

        return system, converted

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI tool format to Anthropic tool format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        return anthropic_tools

    def _convert_response(self, data: dict) -> dict:
        """Convert Anthropic response to OpenAI format."""
        content_blocks = data.get("content", [])
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": __import__("json").dumps(block.get("input", {})),
                    },
                })

        message = {"role": "assistant", "content": "\n".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "choices": [{"message": message, "finish_reason": data.get("stop_reason", "stop")}],
            "usage": data.get("usage", {}),
        }

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> dict:
        api_key = self.get_api_key()
        if not api_key:
            raise RuntimeError("Brak klucza API Anthropic. Dodaj klucz w panelu Providerow.")

        model = model or self._active_model or "claude-sonnet-4-5-20250929"
        system, converted_messages = self._convert_messages(messages)

        payload = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.API_BASE}/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return self._convert_response(resp.json())

    def list_models(self) -> list[dict]:
        return ANTHROPIC_MODELS
