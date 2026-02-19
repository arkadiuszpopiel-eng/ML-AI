"""Tests for conversation storage."""
import json
from pathlib import Path

from neurostudio.backend.storage import (
    save_conversation,
    load_conversation,
    list_conversations,
    delete_conversation,
    update_conversation_title,
    HISTORY_DIR,
)


class TestSaveConversation:
    def test_save_creates_file(self, tmp_path):
        messages = [
            {"role": "user", "content": "Cześć"},
            {"role": "assistant", "content": "Hej!"},
        ]
        meta = save_conversation("test-session-1", messages)

        assert meta["session_id"] == "test-session-1"
        assert meta["title"] == "Cześć"
        assert meta["message_count"] == 2

    def test_save_generates_title_from_first_user_message(self, tmp_path):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Jak działa Python?"},
        ]
        meta = save_conversation("test-session-2", messages)
        assert meta["title"] == "Jak działa Python?"

    def test_save_uses_provided_title(self, tmp_path):
        messages = [{"role": "user", "content": "Hello"}]
        meta = save_conversation("test-session-3", messages, title="Custom Title")
        assert meta["title"] == "Custom Title"

    def test_save_default_title_when_no_user_messages(self, tmp_path):
        messages = [{"role": "system", "content": "System only"}]
        meta = save_conversation("test-session-4", messages)
        assert meta["title"] == "Nowa rozmowa"

    def test_save_preserves_created_at_on_update(self, tmp_path):
        messages = [{"role": "user", "content": "First"}]
        meta1 = save_conversation("test-session-5", messages)
        created_at_1 = meta1["created_at"]

        messages.append({"role": "assistant", "content": "Reply"})
        meta2 = save_conversation("test-session-5", messages)

        assert meta2["created_at"] == created_at_1
        assert meta2["updated_at"] >= meta1["updated_at"]


class TestLoadConversation:
    def test_load_existing(self, tmp_path):
        messages = [{"role": "user", "content": "Test msg"}]
        save_conversation("load-test", messages)

        data = load_conversation("load-test")
        assert data is not None
        assert data["meta"]["session_id"] == "load-test"
        assert len(data["messages"]) == 1

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_conversation("nonexistent") is None


class TestListConversations:
    def test_list_empty(self, tmp_path):
        result = list_conversations()
        assert result == []

    def test_list_multiple(self, tmp_path):
        save_conversation("s1", [{"role": "user", "content": "First"}])
        save_conversation("s2", [{"role": "user", "content": "Second"}])

        result = list_conversations()
        assert len(result) == 2
        # Most recent first
        assert result[0]["session_id"] == "s2"


class TestDeleteConversation:
    def test_delete_existing(self, tmp_path):
        save_conversation("del-test", [{"role": "user", "content": "To delete"}])
        assert delete_conversation("del-test") is True
        assert load_conversation("del-test") is None

    def test_delete_nonexistent(self, tmp_path):
        assert delete_conversation("nonexistent") is False


class TestUpdateTitle:
    def test_update_existing(self, tmp_path):
        save_conversation("title-test", [{"role": "user", "content": "Original"}])
        assert update_conversation_title("title-test", "New Title") is True

        data = load_conversation("title-test")
        assert data["meta"]["title"] == "New Title"

    def test_update_nonexistent(self, tmp_path):
        assert update_conversation_title("nonexistent", "Title") is False
