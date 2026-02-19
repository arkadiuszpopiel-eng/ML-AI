"""
OpenRouterProvider - one API key for 100+ models via OpenRouter gateway.
Uses OpenAI-compatible API format.
"""
import logging
import httpx
from .providers import BaseProvider

logger = logging.getLogger("neurostudio.providers.openrouter")

OPENROUTER_MODELS = [
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "context": "200K"},
    {"id": "openai/gpt-4o", "name": "GPT-4o", "context": "128K"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "context": "128K"},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "context": "1M"},
    {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B", "context": "128K"},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "context": "64K"},
    {"id": "qwen/qwen-2.5-72b-instruct", "name": "Qwen 2.5 72B", "context": "128K"},
    {"id": "mistralai/mistral-large-2411", "name": "Mistral Large", "context": "128K"},
]


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter - one key, 100+ models."""

    provider_id = "openrouter"
    provider_name = "OpenRouter (multi-model)"
    requires_api_key = True

    API_BASE = "https://openrouter.ai/api/v1"

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
            raise RuntimeError("Brak klucza API OpenRouter. Dodaj klucz w panelu Providerow.")

        model = model or self._active_model or "openai/gpt-4o-mini"

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
                    "HTTP-Referer": "http://localhost:7860",
                    "X-Title": "NeuroForge",
                },
            )
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> list[dict]:
        return OPENROUTER_MODELS
