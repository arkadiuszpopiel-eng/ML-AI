"""
OllamaProvider - connects to a local Ollama instance.
Ollama exposes OpenAI-compatible API at http://localhost:11434.
"""
import logging
import httpx
from .providers import BaseProvider

logger = logging.getLogger("neurostudio.providers.ollama")


class OllamaProvider(BaseProvider):
    """Provider for local Ollama instance."""

    provider_id = "ollama"
    provider_name = "Ollama (lokalny)"
    requires_api_key = False

    @property
    def api_base(self) -> str:
        return self.config.get("api_base", "http://localhost:11434")

    @property
    def is_available(self) -> bool:
        """Ollama is available if the server responds."""
        # Optimistic check - real availability tested on first call
        return True

    async def _check_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_base}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.ReadError):
            return False

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> dict:
        model = model or self._active_model or "llama3.1"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = tools

        # Ollama has OpenAI-compatible endpoint
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.api_base}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **({"tools": tools} if tools else {}),
                },
            )
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> list[dict]:
        """Return placeholder list. Real list fetched via API at runtime."""
        return [
            {"id": "llama3.1", "name": "Llama 3.1 8B", "context": "128K"},
            {"id": "qwen2.5", "name": "Qwen 2.5 7B", "context": "128K"},
            {"id": "mistral", "name": "Mistral 7B", "context": "32K"},
            {"id": "codellama", "name": "Code Llama 7B", "context": "16K"},
            {"id": "deepseek-coder-v2", "name": "DeepSeek Coder V2", "context": "128K"},
        ]
