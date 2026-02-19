"""Tests for Usage Tracker."""
import time
from neurostudio.backend.usage import UsageTracker, PRICING


class TestUsageTracker:
    def setup_method(self):
        self.tracker = UsageTracker()

    def test_initial_state(self):
        assert self.tracker.session_tokens == 0
        assert self.tracker.session_cost == 0.0
        assert self.tracker.get_all_usage() == []

    def test_record_openai_usage(self):
        self.tracker.record("openai", "gpt-4o-mini", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })
        assert self.tracker.session_tokens == 150
        usage = self.tracker.get_usage("openai")
        assert usage is not None
        assert usage.requests == 1
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_record_accumulates(self):
        self.tracker.record("openai", "gpt-4o", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })
        self.tracker.record("openai", "gpt-4o", {
            "prompt_tokens": 200,
            "completion_tokens": 100,
        })
        usage = self.tracker.get_usage("openai")
        assert usage.requests == 2
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150
        assert usage.total_tokens == 450

    def test_multiple_providers(self):
        self.tracker.record("openai", "gpt-4o-mini", {
            "prompt_tokens": 100, "completion_tokens": 50,
        })
        self.tracker.record("anthropic", "claude-sonnet-4-5-20250929", {
            "input_tokens": 200, "output_tokens": 100,
        })
        assert self.tracker.session_tokens == 450
        assert len(self.tracker.get_all_usage()) == 2
        assert self.tracker.get_usage("openai").total_tokens == 150
        assert self.tracker.get_usage("anthropic").total_tokens == 300

    def test_anthropic_token_format(self):
        """Anthropic uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens."""
        self.tracker.record("anthropic", "claude-haiku-4-5-20251001", {
            "input_tokens": 500,
            "output_tokens": 200,
        })
        usage = self.tracker.get_usage("anthropic")
        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.total_tokens == 700

    def test_cost_estimation_openai(self):
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        self.tracker.record("openai", "gpt-4o-mini", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        })
        usage = self.tracker.get_usage("openai")
        expected_cost = 0.15 + 0.60  # $0.75
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_estimation_anthropic(self):
        # Claude Sonnet 4.5: $3.00/1M input, $15.00/1M output
        self.tracker.record("anthropic", "claude-sonnet-4-5-20250929", {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        })
        usage = self.tracker.get_usage("anthropic")
        expected_cost = 3.00 + 15.00
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_zero_for_local(self):
        self.tracker.record("local", "some-model.gguf", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        usage = self.tracker.get_usage("local")
        assert usage.estimated_cost_usd == 0.0

    def test_cost_zero_for_ollama(self):
        self.tracker.record("ollama", "llama3.1", {
            "prompt_tokens": 5000,
            "completion_tokens": 2000,
        })
        usage = self.tracker.get_usage("ollama")
        assert usage.estimated_cost_usd == 0.0

    def test_session_cost_sums_providers(self):
        self.tracker.record("openai", "gpt-4o-mini", {
            "prompt_tokens": 1_000_000, "completion_tokens": 0,
        })
        self.tracker.record("anthropic", "claude-haiku-4-5-20251001", {
            "input_tokens": 1_000_000, "output_tokens": 0,
        })
        # gpt-4o-mini input: $0.15, claude-haiku input: $0.80
        expected = 0.15 + 0.80
        assert abs(self.tracker.session_cost - expected) < 0.001

    def test_reset(self):
        self.tracker.record("openai", "gpt-4o", {
            "prompt_tokens": 500, "completion_tokens": 200,
        })
        assert self.tracker.session_tokens > 0
        self.tracker.reset()
        assert self.tracker.session_tokens == 0
        assert self.tracker.session_cost == 0.0
        assert self.tracker.get_all_usage() == []

    def test_to_dict(self):
        self.tracker.record("openai", "gpt-4o-mini", {
            "prompt_tokens": 100, "completion_tokens": 50,
        })
        d = self.tracker.to_dict()
        assert d["session_tokens"] == 150
        assert "session_cost_usd" in d
        assert len(d["providers"]) == 1
        assert d["providers"][0]["provider_id"] == "openai"

    def test_last_used_timestamp(self):
        before = time.time()
        self.tracker.record("openai", "gpt-4o", {
            "prompt_tokens": 10, "completion_tokens": 5,
        })
        after = time.time()
        usage = self.tracker.get_usage("openai")
        assert before <= usage.last_used <= after

    def test_unknown_model_no_crash(self):
        """Recording usage for an unknown model should work, just no cost."""
        self.tracker.record("openai", "gpt-99-turbo", {
            "prompt_tokens": 100, "completion_tokens": 50,
        })
        usage = self.tracker.get_usage("openai")
        assert usage.total_tokens == 150
        assert usage.estimated_cost_usd == 0.0

    def test_empty_usage_data(self):
        """Empty usage dict should record request but no tokens."""
        self.tracker.record("openai", "gpt-4o", {})
        usage = self.tracker.get_usage("openai")
        assert usage.requests == 1
        assert usage.total_tokens == 0

    def test_provider_usage_to_dict(self):
        self.tracker.record("anthropic", "claude-sonnet-4-5-20250929", {
            "input_tokens": 250, "output_tokens": 125,
        })
        usage = self.tracker.get_usage("anthropic")
        d = usage.to_dict()
        assert d["provider_id"] == "anthropic"
        assert d["requests"] == 1
        assert d["input_tokens"] == 250
        assert d["output_tokens"] == 125
        assert d["total_tokens"] == 375
        assert d["estimated_cost_usd"] >= 0


class TestPricingTable:
    def test_openai_models_have_pricing(self):
        assert "gpt-4o" in PRICING["openai"]
        assert "gpt-4o-mini" in PRICING["openai"]

    def test_anthropic_models_have_pricing(self):
        assert "claude-sonnet-4-5-20250929" in PRICING["anthropic"]
        assert "claude-haiku-4-5-20251001" in PRICING["anthropic"]

    def test_google_models_have_pricing(self):
        assert "gemini-2.5-flash" in PRICING["google"]
        assert "gemini-2.0-flash" in PRICING["google"]

    def test_local_has_no_pricing(self):
        assert PRICING["local"] == {}

    def test_ollama_has_no_pricing(self):
        assert PRICING["ollama"] == {}

    def test_all_prices_positive(self):
        for provider_id, models in PRICING.items():
            for model_id, prices in models.items():
                assert prices.get("input", 0) >= 0, f"{provider_id}/{model_id} input < 0"
                assert prices.get("output", 0) >= 0, f"{provider_id}/{model_id} output < 0"
