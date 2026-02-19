"""
Usage tracking for AI providers.
Tracks tokens consumed, request count, and estimated cost per provider.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_config, save_config

logger = logging.getLogger("neurostudio.usage")

# Approximate pricing per 1M tokens (input/output) in USD
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "o3-mini": {"input": 1.10, "output": 4.40},
    },
    "anthropic": {
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    },
    "google": {
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    },
    "openrouter": {
        # OpenRouter has dynamic pricing; use approximate values
        "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "google/gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
        "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
        "qwen/qwen-2.5-72b-instruct": {"input": 0.36, "output": 0.40},
        "mistralai/mistral-large-2411": {"input": 2.00, "output": 6.00},
    },
    # Local and Ollama are free
    "local": {},
    "ollama": {},
}


@dataclass
class ProviderUsage:
    """Usage stats for a single provider."""
    provider_id: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    last_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "last_used": self.last_used,
        }


class UsageTracker:
    """Tracks token usage and costs per provider."""

    def __init__(self):
        self._usage: dict[str, ProviderUsage] = {}
        self._session_tokens: int = 0

    def record(self, provider_id: str, model: str | None, usage_data: dict):
        """
        Record usage from a chat completion response.

        Args:
            provider_id: Provider that served the request
            model: Model used
            usage_data: The 'usage' dict from the API response
        """
        if provider_id not in self._usage:
            self._usage[provider_id] = ProviderUsage(provider_id=provider_id)

        stats = self._usage[provider_id]
        stats.requests += 1
        stats.last_used = time.time()

        # Extract token counts (OpenAI format or Anthropic format)
        input_tok = usage_data.get("prompt_tokens", 0) or usage_data.get("input_tokens", 0)
        output_tok = usage_data.get("completion_tokens", 0) or usage_data.get("output_tokens", 0)
        total = input_tok + output_tok

        stats.input_tokens += input_tok
        stats.output_tokens += output_tok
        stats.total_tokens += total
        self._session_tokens += total

        # Estimate cost
        if model and provider_id in PRICING:
            model_pricing = PRICING[provider_id].get(model, {})
            if model_pricing:
                input_cost = (input_tok / 1_000_000) * model_pricing.get("input", 0)
                output_cost = (output_tok / 1_000_000) * model_pricing.get("output", 0)
                stats.estimated_cost_usd += input_cost + output_cost

        logger.debug(
            "Usage recorded: provider=%s model=%s in=%d out=%d cost=$%.6f",
            provider_id, model, input_tok, output_tok, stats.estimated_cost_usd,
        )

    @property
    def session_tokens(self) -> int:
        """Total tokens consumed in this session (all providers)."""
        return self._session_tokens

    @property
    def session_cost(self) -> float:
        """Total estimated cost in this session."""
        return sum(u.estimated_cost_usd for u in self._usage.values())

    def get_usage(self, provider_id: str) -> ProviderUsage | None:
        return self._usage.get(provider_id)

    def get_all_usage(self) -> list[dict]:
        return [u.to_dict() for u in self._usage.values()]

    def to_dict(self) -> dict:
        return {
            "session_tokens": self._session_tokens,
            "session_cost_usd": round(self.session_cost, 6),
            "providers": self.get_all_usage(),
        }

    def reset(self):
        """Reset all usage stats."""
        self._usage.clear()
        self._session_tokens = 0


# Singleton
usage_tracker = UsageTracker()
