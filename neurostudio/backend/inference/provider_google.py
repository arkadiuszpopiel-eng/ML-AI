"""
GoogleProvider - connects to Google Gemini API.
Translates between OpenAI format (used internally) and Gemini format.
"""
import json
import logging
import httpx
from .providers import BaseProvider

logger = logging.getLogger("neurostudio.providers.google")

GEMINI_MODELS = [
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "context": "1M"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "context": "1M"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context": "1M"},
]


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini API."""

    provider_id = "google"
    provider_name = "Google (Gemini)"
    requires_api_key = True

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Gemini format."""
        system = None
        contents = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content or ""}]})
            elif role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"[Tool result]: {content}"}],
                })

        return system, contents

    def _convert_response(self, data: dict) -> dict:
        """Convert Gemini response to OpenAI format."""
        candidates = data.get("candidates", [{}])
        if not candidates:
            return {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if "text" in p)

        return {
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": candidate.get("finishReason", "STOP").lower(),
            }],
            "usage": data.get("usageMetadata", {}),
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
            raise RuntimeError("Brak klucza API Google. Dodaj klucz w panelu Providerow.")

        model = model or self._active_model or "gemini-2.0-flash"
        system, contents = self._convert_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self.API_BASE}/models/{model}:generateContent?key={api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return self._convert_response(resp.json())

    def list_models(self) -> list[dict]:
        return GEMINI_MODELS
