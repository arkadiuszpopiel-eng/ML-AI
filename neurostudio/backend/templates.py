"""
Prompt templates - pre-built prompts for common tasks.
"""
import json
from pathlib import Path
from typing import Optional

from .config import BASE_DIR

TEMPLATES_PATH = BASE_DIR / "data" / "templates.json"

# Built-in templates
DEFAULT_TEMPLATES = [
    {
        "id": "code-review",
        "name": "Przeglad kodu",
        "icon": "&#128270;",
        "category": "coding",
        "prompt": "Przeanalizuj ponizszy kod. Znajdz bledy, zaproponuj ulepszenia dotyczace wydajnosci, czytelnosci i bezpieczenstwa:\n\n```\n{code}\n```",
        "variables": ["code"],
    },
    {
        "id": "explain-code",
        "name": "Wyjasnij kod",
        "icon": "&#128218;",
        "category": "coding",
        "prompt": "Wyjasnij co robi ponizszy kod, linia po linii. Uzyj prostego jezyka:\n\n```\n{code}\n```",
        "variables": ["code"],
    },
    {
        "id": "write-tests",
        "name": "Napisz testy",
        "icon": "&#9989;",
        "category": "coding",
        "prompt": "Napisz testy jednostkowe dla ponizszego kodu. Uzyj pytest:\n\n```python\n{code}\n```",
        "variables": ["code"],
    },
    {
        "id": "refactor",
        "name": "Refaktoryzacja",
        "icon": "&#9881;",
        "category": "coding",
        "prompt": "Zrefaktoryzuj ponizszy kod. Popraw czytelnosc, usun duplikacje, zastosuj dobre praktyki:\n\n```\n{code}\n```",
        "variables": ["code"],
    },
    {
        "id": "summarize",
        "name": "Podsumowanie tekstu",
        "icon": "&#128196;",
        "category": "text",
        "prompt": "Podsumuj ponizszy tekst w 3-5 punktach. Wyodrebnij najwazniejsze informacje:\n\n{text}",
        "variables": ["text"],
    },
    {
        "id": "translate-en-pl",
        "name": "Tlumaczenie EN->PL",
        "icon": "&#127760;",
        "category": "text",
        "prompt": "Przetlumacz ponizszy tekst z angielskiego na polski. Zachowaj naturalny styl:\n\n{text}",
        "variables": ["text"],
    },
    {
        "id": "translate-pl-en",
        "name": "Tlumaczenie PL->EN",
        "icon": "&#127760;",
        "category": "text",
        "prompt": "Translate the following text from Polish to English. Keep natural style:\n\n{text}",
        "variables": ["text"],
    },
    {
        "id": "analyze-file",
        "name": "Analiza pliku",
        "icon": "&#128193;",
        "category": "tools",
        "prompt": "Przeczytaj plik {filepath} i dokonaj jego analizy. Opisz zawartosc, strukture i potencjalne problemy.",
        "variables": ["filepath"],
    },
    {
        "id": "project-structure",
        "name": "Struktura projektu",
        "icon": "&#128194;",
        "category": "tools",
        "prompt": "Wylistuj i opisz strukture katalogu {directory}. Podaj drzewo plikow i krotki opis kazdego elementu.",
        "variables": ["directory"],
    },
    {
        "id": "system-check",
        "name": "Stan systemu",
        "icon": "&#128187;",
        "category": "tools",
        "prompt": "Sprawdz stan systemu: zuzycie CPU, RAM, dysku. Wylistuj top 10 procesow zuzycie zasobow. Podaj informacje o GPU jesli dostepne.",
        "variables": [],
    },
    {
        "id": "web-research",
        "name": "Szukaj w sieci",
        "icon": "&#128269;",
        "category": "tools",
        "prompt": "Wyszukaj w internecie informacje na temat: {topic}. Podaj zwiezle podsumowanie z linkami do zrodel.",
        "variables": ["topic"],
    },
    {
        "id": "git-status",
        "name": "Status Git",
        "icon": "&#128204;",
        "category": "tools",
        "prompt": "Wykonaj 'git status' i 'git log --oneline -10' w katalogu {directory}. Opisz stan repozytorium.",
        "variables": ["directory"],
    },
]


def _ensure_dir():
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def _load_custom_templates() -> list[dict]:
    """Load user-created templates."""
    if not TEMPLATES_PATH.exists():
        return []
    try:
        with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_custom_templates(templates: list[dict]):
    """Save user-created templates."""
    _ensure_dir()
    with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)


def get_all_templates() -> list[dict]:
    """Get all templates (built-in + custom)."""
    custom = _load_custom_templates()
    # Mark built-in vs custom
    result = [dict(t, builtin=True) for t in DEFAULT_TEMPLATES]
    result.extend(dict(t, builtin=False) for t in custom)
    return result


def get_template(template_id: str) -> Optional[dict]:
    """Get a specific template by ID."""
    for t in get_all_templates():
        if t["id"] == template_id:
            return t
    return None


def add_custom_template(template: dict) -> dict:
    """Add a user-created template."""
    custom = _load_custom_templates()
    # Auto-generate ID if missing
    if "id" not in template:
        template["id"] = f"custom-{len(custom) + 1}"
    custom.append(template)
    _save_custom_templates(custom)
    return template


def delete_custom_template(template_id: str) -> bool:
    """Delete a user-created template."""
    custom = _load_custom_templates()
    new_custom = [t for t in custom if t.get("id") != template_id]
    if len(new_custom) == len(custom):
        return False
    _save_custom_templates(new_custom)
    return True
