"""
Provider abstraction layer for AI inference.

Defines a common interface (BaseProvider) for all inference sources:
- LocalProvider: wraps llama.cpp engine (existing)
- OpenAIProvider: OpenAI API (GPT-4o, GPT-4o-mini)
- AnthropicProvider: Anthropic API (Claude)
- GoogleProvider: Google Gemini API
- OllamaProvider: local Ollama instance
- OpenRouterProvider: OpenRouter gateway (100+ models)

Each provider implements chat_completion() with the same signature,
so the agent loop doesn't care which backend generates the response.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("neurostudio.providers")


class BaseProvider(ABC):
    """Abstract base class for all AI inference providers."""

    # Subclasses must set these
    provider_id: str = ""
    provider_name: str = ""
    requires_api_key: bool = False

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._active_model: str | None = None

    @property
    def is_available(self) -> bool:
        """Check if this provider is configured and ready to use."""
        if self.requires_api_key:
            return bool(self.get_api_key())
        return True

    @property
    def active_model(self) -> str | None:
        """Currently selected model for this provider."""
        return self._active_model

    def get_api_key(self) -> str | None:
        """Get the API key from provider config."""
        return self.config.get("api_key")

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> dict:
        """
        Send a chat completion request.

        Args:
            messages: Conversation history in OpenAI format
            tools: Tool definitions in OpenAI format (optional)
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            model: Model to use (provider-specific)

        Returns:
            OpenAI-compatible chat completion response dict
        """
        ...

    @abstractmethod
    def list_models(self) -> list[dict]:
        """
        List available models for this provider.

        Returns:
            List of dicts with at least: {"id": str, "name": str}
        """
        ...

    def to_dict(self) -> dict:
        """Serialize provider info (safe for API response, no secrets)."""
        return {
            "id": self.provider_id,
            "name": self.provider_name,
            "requires_api_key": self.requires_api_key,
            "has_api_key": bool(self.get_api_key()) if self.requires_api_key else True,
            "is_available": self.is_available,
            "active_model": self._active_model,
            "models": self.list_models(),
        }


class ProviderRegistry:
    """Registry of all available inference providers."""

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._active_provider_id: str | None = None

    def register(self, provider: BaseProvider):
        """Register a provider."""
        self._providers[provider.provider_id] = provider
        logger.info("Registered provider: %s (%s)", provider.provider_id, provider.provider_name)

    def get(self, provider_id: str) -> BaseProvider | None:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def list_providers(self) -> list[BaseProvider]:
        """List all registered providers."""
        return list(self._providers.values())

    @property
    def active_provider(self) -> BaseProvider | None:
        """Get the currently active provider."""
        if self._active_provider_id:
            return self._providers.get(self._active_provider_id)
        return None

    def set_active(self, provider_id: str) -> bool:
        """Set the active provider. Returns True if successful."""
        if provider_id not in self._providers:
            logger.warning("Unknown provider: %s", provider_id)
            return False
        provider = self._providers[provider_id]
        if not provider.is_available:
            logger.warning("Provider not available (missing API key?): %s", provider_id)
            return False
        self._active_provider_id = provider_id
        logger.info("Active provider set to: %s", provider_id)
        return True

    def to_dict(self) -> dict:
        """Serialize registry state."""
        return {
            "active_provider": self._active_provider_id,
            "providers": [p.to_dict() for p in self._providers.values()],
        }


# Singleton registry
provider_registry = ProviderRegistry()
