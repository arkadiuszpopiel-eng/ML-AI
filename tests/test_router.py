"""Tests for Semantic Router - task detection, scoring, and configuration."""
import pytest
from unittest.mock import patch, MagicMock
from neurostudio.backend.inference.router import (
    detect_task_type,
    get_task_scores,
    SemanticScorer,
    TASK_PROFILES,
    _detect_task_type_keywords,
    get_router_config,
    update_router_config,
)


class TestSemanticScorer:
    def test_build(self):
        scorer = SemanticScorer()
        scorer.build(TASK_PROFILES)
        assert scorer._built is True
        assert len(scorer._profile_vectors) == 4
        assert len(scorer._idf) > 0

    def test_score_returns_all_types(self):
        scorer = SemanticScorer()
        scorer.build(TASK_PROFILES)
        scores = scorer.score("write a python function")
        assert set(scores.keys()) == {"coding", "analysis", "creative", "chat"}

    def test_score_empty_message(self):
        scorer = SemanticScorer()
        scorer.build(TASK_PROFILES)
        scores = scorer.score("")
        assert all(v == 0.0 for v in scores.values())

    def test_score_values_are_bounded(self):
        scorer = SemanticScorer()
        scorer.build(TASK_PROFILES)
        scores = scorer.score("implement a sorting algorithm in python")
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_auto_builds_on_first_score(self):
        scorer = SemanticScorer()
        assert scorer._built is False
        scorer.score("hello")
        assert scorer._built is True


class TestDetectTaskType:
    def test_coding_message(self):
        result = detect_task_type("Write a Python function to sort a list")
        assert result == "coding"

    def test_analysis_message(self):
        result = detect_task_type("Analyze the performance benchmark data")
        assert result == "analysis"

    def test_creative_message(self):
        result = detect_task_type("Write a short story about a dragon")
        assert result == "creative"

    def test_chat_message(self):
        result = detect_task_type("Hello, how are you?")
        assert result == "chat"

    def test_empty_message_defaults_to_chat(self):
        result = detect_task_type("")
        assert result == "chat"

    def test_ambiguous_defaults_to_best_match(self):
        result = detect_task_type("help me debug this code")
        assert result in ("coding", "chat")

    def test_coding_keywords(self):
        result = detect_task_type("implement a REST API endpoint in javascript")
        assert result == "coding"

    def test_analysis_keywords(self):
        result = detect_task_type("compare and evaluate the statistics and metrics")
        assert result == "analysis"


class TestGetTaskScores:
    def test_returns_dict(self):
        scores = get_task_scores("write code in python")
        assert isinstance(scores, dict)
        assert "coding" in scores

    def test_coding_message_scores_highest_for_coding(self):
        scores = get_task_scores("implement a function in python to sort data")
        assert scores["coding"] >= scores["chat"]
        assert scores["coding"] >= scores["creative"]


class TestKeywordFallback:
    def test_coding_keywords(self):
        result = _detect_task_type_keywords("debug this python function")
        assert result == "coding"

    def test_analysis_keywords(self):
        result = _detect_task_type_keywords("analyze the performance data")
        assert result == "analysis"

    def test_creative_keywords(self):
        result = _detect_task_type_keywords("write a poem about nature")
        assert result == "creative"

    def test_chat_keywords(self):
        result = _detect_task_type_keywords("hello, help me")
        assert result == "chat"

    def test_empty_defaults_to_chat(self):
        result = _detect_task_type_keywords("")
        assert result == "chat"

    def test_unknown_defaults_to_chat(self):
        result = _detect_task_type_keywords("xyzzy foobar baz")
        assert result == "chat"


class TestRouterConfig:
    @patch("neurostudio.backend.inference.router.load_config")
    @patch("neurostudio.backend.inference.router.list_models")
    def test_get_router_config(self, mock_models, mock_config):
        mock_config.return_value = {
            "router": {"enabled": True, "coding": "model.gguf"}
        }
        mock_models.return_value = []

        result = get_router_config()
        assert result["enabled"] is True
        assert result["assignments"]["coding"] == "model.gguf"
        assert result["assignments"]["chat"] is None
        assert "task_types" in result
        assert len(result["task_types"]) == 4

    @patch("neurostudio.backend.config.save_config")
    @patch("neurostudio.backend.inference.router.load_config")
    @patch("neurostudio.backend.inference.router.list_models")
    def test_update_router_config_enable(self, mock_models, mock_config, mock_save):
        mock_config.return_value = {"router": {"enabled": False}}
        mock_models.return_value = []

        result = update_router_config(enabled=True)
        assert result["enabled"] is True
        mock_save.assert_called_once()

    @patch("neurostudio.backend.config.save_config")
    @patch("neurostudio.backend.inference.router.load_config")
    @patch("neurostudio.backend.inference.router.list_models")
    def test_update_router_assignments(self, mock_models, mock_config, mock_save):
        mock_config.return_value = {"router": {"enabled": True}}
        mock_models.return_value = []

        result = update_router_config(assignments={"coding": "qwen-coder.gguf"})
        assert result["assignments"]["coding"] == "qwen-coder.gguf"
