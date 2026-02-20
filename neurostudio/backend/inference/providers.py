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
    """Registry of all available inference providers.

    Supports:
    - Single active provider (classic mode)
    - Fallback chain: ordered list of providers to try if primary fails
    - Smart routing: route simple queries to local, complex to cloud
    """

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._active_provider_id: str | None = None
        self._fallback_chain: list[str] = []
        self._smart_routing: bool = False

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

    # ── Fallback chain ──

    @property
    def fallback_chain(self) -> list[str]:
        return list(self._fallback_chain)

    def set_fallback_chain(self, chain: list[str]):
        """Set ordered list of provider IDs to try as fallbacks."""
        valid = [pid for pid in chain if pid in self._providers]
        self._fallback_chain = valid
        logger.info("Fallback chain set: %s", valid)

    def get_fallback_providers(self) -> list[BaseProvider]:
        """Get ordered list of available fallback providers (excluding active)."""
        result = []
        for pid in self._fallback_chain:
            if pid == self._active_provider_id:
                continue
            provider = self._providers.get(pid)
            if provider and provider.is_available:
                result.append(provider)
        return result

    # ── Smart routing ──

    @property
    def smart_routing(self) -> bool:
        return self._smart_routing

    def set_smart_routing(self, enabled: bool):
        self._smart_routing = enabled
        logger.info("Smart routing %s", "enabled" if enabled else "disabled")

    def route_message(self, message: str) -> BaseProvider | None:
        """Pick the best provider for a message based on complexity.

        Simple heuristic:
        - Short casual messages (< 80 chars, no code markers) → local
        - Longer or technical messages → cloud (active provider)
        """
        if not self._smart_routing:
            return self.active_provider

        # Check if a cloud provider is available
        cloud = self.active_provider
        if not cloud or cloud.provider_id == "local":
            return self.active_provider

        # Heuristic: is this a simple query?
        code_markers = ["```", "def ", "function ", "class ", "import ", "SELECT ", "async "]
        is_complex = (
            len(message) > 80
            or any(m in message for m in code_markers)
            or message.count("\n") > 2
        )

        if is_complex:
            return cloud  # Use cloud API for complex queries
        else:
            # Check if local engine is running before routing there
            local = self._providers.get("local")
            if local and local.is_available:
                return local
            return cloud  # Fallback to cloud if local not available

    def to_dict(self) -> dict:
        """Serialize registry state."""
        return {
            "active_provider": self._active_provider_id,
            "providers": [p.to_dict() for p in self._providers.values()],
            "fallback_chain": self._fallback_chain,
            "smart_routing": self._smart_routing,
        }


# Singleton registry
provider_registry = ProviderRegistry()
