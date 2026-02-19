"""
Inference module - providers, engine, router, model manager.

Initializes and registers all providers on import.
"""
from ..config import load_config
from .providers import provider_registry
from .provider_local import LocalProvider
from .provider_openai import OpenAIProvider
from .provider_anthropic import AnthropicProvider
from .provider_google import GoogleProvider
from .provider_ollama import OllamaProvider
from .provider_openrouter import OpenRouterProvider


def init_providers():
    """Register all providers with their config from config.yaml."""
    config = load_config()
    providers_config = config.get("providers", {})

    # Local is always registered
    provider_registry.register(LocalProvider(providers_config.get("local", {})))

    # Cloud providers
    provider_registry.register(OpenAIProvider(providers_config.get("openai", {})))
    provider_registry.register(AnthropicProvider(providers_config.get("anthropic", {})))
    provider_registry.register(GoogleProvider(providers_config.get("google", {})))
    provider_registry.register(OllamaProvider(providers_config.get("ollama", {})))
    provider_registry.register(OpenRouterProvider(providers_config.get("openrouter", {})))

    # Set active provider (default to local)
    active = providers_config.get("active", "local")
    provider_registry.set_active(active)


# Auto-init on import
init_providers()
