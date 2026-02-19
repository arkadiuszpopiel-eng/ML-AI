"""
LocalProvider - wraps the existing llama.cpp InferenceEngine as a provider.
"""
from .providers import BaseProvider
from .engine import engine


class LocalProvider(BaseProvider):
    """Provider for local llama.cpp inference."""

    provider_id = "local"
    provider_name = "Lokalny (llama.cpp)"
    requires_api_key = False

    @property
    def is_available(self) -> bool:
        return True

    @property
    def active_model(self) -> str | None:
        if engine.current_model:
            from pathlib import Path
            return Path(engine.current_model).name
        return self._active_model

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> dict:
        """Delegate to llama.cpp engine."""
        if not engine.is_running:
            raise RuntimeError("Lokalny model nie jest zaladowany. Zaladuj model najpierw.")

        return await engine.chat_completion(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

    def list_models(self) -> list[dict]:
        """List locally available GGUF models."""
        from .model_manager import list_models
        return [
            {"id": m.filename, "name": m.base_name, "size": m.size_display}
            for m in list_models()
        ]
