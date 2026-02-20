"""Tests for Provider abstraction layer."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from neurostudio.backend.inference.providers import (
    BaseProvider, ProviderRegistry, provider_registry,
)
from neurostudio.backend.inference.provider_local import LocalProvider
from neurostudio.backend.inference.provider_openai import OpenAIProvider
from neurostudio.backend.inference.provider_anthropic import AnthropicProvider
from neurostudio.backend.inference.provider_google import GoogleProvider
from neurostudio.backend.inference.provider_ollama import OllamaProvider
from neurostudio.backend.inference.provider_openrouter import OpenRouterProvider


class TestProviderRegistry:
    def test_register_provider(self):
        registry = ProviderRegistry()
        provider = LocalProvider()
        registry.register(provider)
        assert registry.get("local") is provider

    def test_list_providers(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.register(OpenAIProvider())
        assert len(registry.list_providers()) == 2

    def test_set_active(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        assert registry.set_active("local") is True
        assert registry.active_provider.provider_id == "local"

    def test_set_active_unknown(self):
        registry = ProviderRegistry()
        assert registry.set_active("nonexistent") is False

    def test_active_provider_default_none(self):
        registry = ProviderRegistry()
        assert registry.active_provider is None

    def test_to_dict(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.set_active("local")
        data = registry.to_dict()
        assert data["active_provider"] == "local"
        assert len(data["providers"]) == 1
        assert data["providers"][0]["id"] == "local"


class TestLocalProvider:
    def test_provider_id(self):
        p = LocalProvider()
        assert p.provider_id == "local"
        assert p.provider_name == "Lokalny (llama.cpp)"

    def test_is_available(self):
        p = LocalProvider()
        assert p.is_available is True

    def test_requires_no_api_key(self):
        p = LocalProvider()
        assert p.requires_api_key is False

    @patch("neurostudio.backend.inference.model_manager.list_models")
    def test_list_models(self, mock_list):
        mock_list.return_value = [
            MagicMock(filename="test.gguf", base_name="test", size_display="1 GB"),
        ]
        p = LocalProvider()
        models = p.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "test.gguf"

    def test_to_dict(self):
        p = LocalProvider()
        d = p.to_dict()
        assert d["id"] == "local"
        assert d["is_available"] is True
        assert d["requires_api_key"] is False


class TestOpenAIProvider:
    def test_provider_id(self):
        p = OpenAIProvider()
        assert p.provider_id == "openai"
        assert p.requires_api_key is True

    def test_not_available_without_key(self):
        p = OpenAIProvider()
        assert p.is_available is False

    def test_available_with_key(self):
        p = OpenAIProvider({"api_key": "sk-test123"})
        assert p.is_available is True

    def test_list_models(self):
        p = OpenAIProvider()
        models = p.list_models()
        assert len(models) >= 3
        assert any(m["id"] == "gpt-4o" for m in models)

    def test_to_dict_hides_key(self):
        p = OpenAIProvider({"api_key": "sk-secret"})
        d = p.to_dict()
        assert d["has_api_key"] is True
        assert "sk-secret" not in str(d)


class TestAnthropicProvider:
    def test_provider_id(self):
        p = AnthropicProvider()
        assert p.provider_id == "anthropic"

    def test_list_models(self):
        p = AnthropicProvider()
        models = p.list_models()
        assert len(models) >= 2
        assert any("claude" in m["id"] for m in models)

    def test_convert_messages(self):
        p = AnthropicProvider()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        system, converted = p._convert_messages(messages)
        assert system == "You are helpful"
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_convert_tools(self):
        p = AnthropicProvider()
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        result = p._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"

    def test_convert_tools_none(self):
        p = AnthropicProvider()
        assert p._convert_tools(None) is None

    def test_convert_response(self):
        p = AnthropicProvider()
        resp = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
        }
        result = p._convert_response(resp)
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_convert_response_with_tool_use(self):
        p = AnthropicProvider()
        resp = {
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/tmp"}},
            ],
            "stop_reason": "tool_use",
        }
        result = p._convert_response(resp)
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "read_file"


class TestGoogleProvider:
    def test_provider_id(self):
        p = GoogleProvider()
        assert p.provider_id == "google"

    def test_list_models(self):
        p = GoogleProvider()
        models = p.list_models()
        assert len(models) >= 2
        assert any("gemini" in m["id"] for m in models)

    def test_convert_messages(self):
        p = GoogleProvider()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ]
        system, contents = p._convert_messages(messages)
        assert system == "System prompt"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_convert_response(self):
        p = GoogleProvider()
        resp = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello!"}]}, "finishReason": "STOP"}
            ]
        }
        result = p._convert_response(resp)
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_convert_response_empty(self):
        p = GoogleProvider()
        result = p._convert_response({"candidates": []})
        assert result["choices"][0]["message"]["content"] == ""


class TestOllamaProvider:
    def test_provider_id(self):
        p = OllamaProvider()
        assert p.provider_id == "ollama"
        assert p.requires_api_key is False

    def test_api_base_default(self):
        p = OllamaProvider()
        assert p.api_base == "http://localhost:11434"

    def test_api_base_custom(self):
        p = OllamaProvider({"api_base": "http://192.168.1.100:11434"})
        assert p.api_base == "http://192.168.1.100:11434"

    def test_list_models(self):
        p = OllamaProvider()
        models = p.list_models()
        assert len(models) >= 3


class TestOpenRouterProvider:
    def test_provider_id(self):
        p = OpenRouterProvider()
        assert p.provider_id == "openrouter"
        assert p.requires_api_key is True

    def test_list_models(self):
        p = OpenRouterProvider()
        models = p.list_models()
        assert len(models) >= 5
        assert any("claude" in m["id"] for m in models)
        assert any("gpt" in m["id"] for m in models)


class TestGlobalRegistry:
    def test_global_registry_has_providers(self):
        """The global registry should be populated by __init__.py."""
        assert provider_registry.get("local") is not None
        assert provider_registry.get("openai") is not None
        assert provider_registry.get("anthropic") is not None
        assert provider_registry.get("google") is not None
        assert provider_registry.get("ollama") is not None
        assert provider_registry.get("openrouter") is not None

    def test_global_registry_active_default(self):
        """Default active provider should be 'local'."""
        assert provider_registry.active_provider is not None
        assert provider_registry.active_provider.provider_id == "local"


class TestFallbackChain:
    def test_empty_chain_by_default(self):
        registry = ProviderRegistry()
        assert registry.fallback_chain == []

    def test_set_fallback_chain(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.register(OpenAIProvider({"api_key": "sk-test"}))
        registry.set_fallback_chain(["openai", "local"])
        assert registry.fallback_chain == ["openai", "local"]

    def test_set_fallback_chain_filters_unknown(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.set_fallback_chain(["nonexistent", "local"])
        assert registry.fallback_chain == ["local"]

    def test_get_fallback_providers_excludes_active(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        oai = OpenAIProvider({"api_key": "sk-test"})
        registry.register(oai)
        registry.set_active("openai")
        registry.set_fallback_chain(["openai", "local"])
        fallbacks = registry.get_fallback_providers()
        ids = [p.provider_id for p in fallbacks]
        assert "openai" not in ids
        assert "local" in ids

    def test_get_fallback_providers_empty_when_no_chain(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        assert registry.get_fallback_providers() == []

    def test_to_dict_includes_fallback_chain(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.set_active("local")
        registry.set_fallback_chain(["local"])
        data = registry.to_dict()
        assert "fallback_chain" in data
        assert data["fallback_chain"] == ["local"]


class TestSmartRouting:
    def test_disabled_by_default(self):
        registry = ProviderRegistry()
        assert registry.smart_routing is False

    def test_enable_smart_routing(self):
        registry = ProviderRegistry()
        registry.set_smart_routing(True)
        assert registry.smart_routing is True

    def test_route_message_returns_active_when_disabled(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.set_active("local")
        result = registry.route_message("hello")
        assert result.provider_id == "local"

    def test_route_simple_to_local_when_enabled(self):
        registry = ProviderRegistry()
        local = LocalProvider()
        oai = OpenAIProvider({"api_key": "sk-test"})
        registry.register(local)
        registry.register(oai)
        registry.set_active("openai")
        registry.set_smart_routing(True)

        # Short simple message should route to local (if available)
        result = registry.route_message("Czesc, jak sie masz?")
        assert result.provider_id == "local"

    def test_route_complex_to_cloud_when_enabled(self):
        registry = ProviderRegistry()
        local = LocalProvider()
        oai = OpenAIProvider({"api_key": "sk-test"})
        registry.register(local)
        registry.register(oai)
        registry.set_active("openai")
        registry.set_smart_routing(True)

        # Long message with code markers should go to cloud
        result = registry.route_message("Napisz mi funkcje Python ```def calculate_fibonacci(n): ...")
        assert result.provider_id == "openai"

    def test_route_falls_back_to_cloud_when_local_unavailable(self):
        registry = ProviderRegistry()
        oai = OpenAIProvider({"api_key": "sk-test"})
        registry.register(oai)
        registry.set_active("openai")
        registry.set_smart_routing(True)

        # No local provider registered - should use cloud even for simple
        result = registry.route_message("Hi")
        assert result.provider_id == "openai"

    def test_to_dict_includes_smart_routing(self):
        registry = ProviderRegistry()
        registry.register(LocalProvider())
        registry.set_active("local")
        registry.set_smart_routing(True)
        data = registry.to_dict()
        assert "smart_routing" in data
        assert data["smart_routing"] is True
