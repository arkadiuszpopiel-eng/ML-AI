"""
OpenAIProvider - connects to OpenAI API (GPT-4o, GPT-4o-mini, etc.)
Ready to use once an API key is provided in config.
"""
import logging
import httpx
from .providers import BaseProvider

logger = logging.getLogger("neurostudio.providers.openai")

OPENAI_MODELS = [
    {"id": "gpt-4o", "name": "GPT-4o", "context": "128K"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "context": "128K"},
    {"id": "gpt-4.1", "name": "GPT-4.1", "context": "1M"},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "context": "1M"},
    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "context": "1M"},
    {"id": "o3-mini", "name": "o3-mini (reasoning)", "context": "200K"},
]


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API."""

    provider_id = "openai"
    provider_name = "OpenAI"
    requires_api_key = True

    API_BASE = "https://api.openai.com/v1"

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
            raise RuntimeError("Brak klucza API OpenAI. Dodaj klucz w panelu Providerow.")

        model = model or self._active_model or "gpt-4o-mini"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.API_BASE}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> list[dict]:
        return OPENAI_MODELS
