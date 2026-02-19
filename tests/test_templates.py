"""Tests for prompt templates."""
from neurostudio.backend.templates import (
    get_all_templates,
    get_template,
    add_custom_template,
    delete_custom_template,
    DEFAULT_TEMPLATES,
)


class TestDefaultTemplates:
    def test_has_builtin_templates(self):
        assert len(DEFAULT_TEMPLATES) >= 10

    def test_each_template_has_required_fields(self):
        for t in DEFAULT_TEMPLATES:
            assert "id" in t
            assert "name" in t
            assert "prompt" in t
            assert "category" in t


class TestGetAllTemplates:
    def test_returns_builtins(self, tmp_path):
        templates = get_all_templates()
        assert len(templates) >= len(DEFAULT_TEMPLATES)
        assert all(t.get("builtin") is True for t in templates[:len(DEFAULT_TEMPLATES)])


class TestGetTemplate:
    def test_get_existing(self, tmp_path):
        t = get_template("code-review")
        assert t is not None
        assert t["name"] == "Przeglad kodu"

    def test_get_nonexistent(self, tmp_path):
        assert get_template("nonexistent-id") is None


class TestCustomTemplates:
    def test_add_custom(self, tmp_path):
        t = add_custom_template({
            "name": "My template",
            "prompt": "Do something: {input}",
            "category": "custom",
            "variables": ["input"],
        })
        assert "id" in t

        all_t = get_all_templates()
        custom = [x for x in all_t if not x.get("builtin")]
        assert len(custom) >= 1

    def test_delete_custom(self, tmp_path):
        t = add_custom_template({
            "id": "delete-me",
            "name": "To delete",
            "prompt": "Test",
            "category": "custom",
        })

        assert delete_custom_template("delete-me") is True
        assert delete_custom_template("delete-me") is False

    def test_cannot_delete_builtin(self, tmp_path):
        # Builtin templates are not in the custom file, so delete returns False
        assert delete_custom_template("code-review") is False
