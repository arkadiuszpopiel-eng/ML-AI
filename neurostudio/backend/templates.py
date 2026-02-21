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
    # ── Nowe solidne szablony ──
    {
        "id": "debug-error",
        "name": "Debuguj blad",
        "icon": "&#128027;",
        "category": "coding",
        "prompt": "Pomoz mi zdebugowac nastepujacy blad. Przeanalizuj stacktrace, zidentyfikuj przyczyne i zaproponuj rozwiazanie krok po kroku:\n\n```\n{error}\n```\n\nKontekst: {context}",
        "variables": ["error", "context"],
    },
    {
        "id": "api-design",
        "name": "Projektuj API",
        "icon": "&#128640;",
        "category": "coding",
        "prompt": "Zaprojektuj RESTful API dla: {description}.\n\nUwzglednij:\n1. Endpointy (metoda HTTP, sciezka, opis)\n2. Request/Response body (JSON schema)\n3. Kody odpowiedzi HTTP\n4. Autentykacje i autoryzacje\n5. Paginacje i filtrowanie\n6. Obsluge bledow",
        "variables": ["description"],
    },
    {
        "id": "sql-query",
        "name": "Napisz zapytanie SQL",
        "icon": "&#128451;",
        "category": "coding",
        "prompt": "Napisz zoptymalizowane zapytanie SQL dla nastepujacego zadania:\n\n{task}\n\nStruktura bazy danych: {schema}\n\nUwzglednij wydajnosc, indeksy i dobre praktyki SQL.",
        "variables": ["task", "schema"],
    },
    {
        "id": "security-audit",
        "name": "Audyt bezpieczenstwa",
        "icon": "&#128274;",
        "category": "analysis",
        "prompt": "Przeprowadz audyt bezpieczenstwa nastepujacego kodu. Szukaj podatnosci OWASP Top 10:\n- SQL Injection\n- XSS\n- CSRF\n- Broken Authentication\n- Sensitive Data Exposure\n- Insecure Deserialization\n- Command Injection\n\n```\n{code}\n```\n\nPodaj konkretne podatnosci, ich waznosc (krytyczna/wysoka/srednia/niska) i rekomendacje naprawy.",
        "variables": ["code"],
    },
    {
        "id": "performance-analysis",
        "name": "Analiza wydajnosci",
        "icon": "&#9889;",
        "category": "analysis",
        "prompt": "Przeanalizuj wydajnosc nastepujacego kodu. Zidentyfikuj:\n1. Bottlenecki i wezle gardla\n2. Zlozonosc obliczeniowa (Big O)\n3. Uzycie pamieci\n4. Mozliwosci optymalizacji\n5. Problemy z I/O\n6. Mozliwosci rownoleglego przetwarzania\n\n```\n{code}\n```\n\nZaproponuj zoptymalizowana wersje z wyjasneniem zmian.",
        "variables": ["code"],
    },
    {
        "id": "architecture-review",
        "name": "Przeglad architektury",
        "icon": "&#127959;",
        "category": "analysis",
        "prompt": "Przeanalizuj architekture projektu w katalogu {directory}. Ocen:\n1. Strukture katalogow i organizacje kodu\n2. Wzorce projektowe (design patterns)\n3. Separacje warstw (MVC/MVVM/Clean Architecture)\n4. Zarzadzanie zaleznosc (dependency injection)\n5. Skalowalisc i maintainability\n6. Testowalnosc\n\nPodaj konkretne rekomendacje ulepszen z przykladami.",
        "variables": ["directory"],
    },
    {
        "id": "write-documentation",
        "name": "Generuj dokumentacje",
        "icon": "&#128214;",
        "category": "text",
        "prompt": "Wygeneruj kompletna dokumentacje dla nastepujacego kodu/modulu:\n\n```\n{code}\n```\n\nUwzglednij:\n1. Opis ogolny (co robi, do czego sluzy)\n2. Instalacje i konfiguracje\n3. API Reference (funkcje, klasy, parametry, zwracane wartosci)\n4. Przyklady uzycia\n5. FAQ / Troubleshooting",
        "variables": ["code"],
    },
    {
        "id": "creative-brainstorm",
        "name": "Brainstorming pomyslow",
        "icon": "&#128161;",
        "category": "creative",
        "prompt": "Przeprowadz sesje brainstormingu na temat: {topic}\n\nWygeneruj:\n1. 10 kreatywnych pomyslow (od konwencjonalnych po nieszablonowe)\n2. Dla kazdego pomyslu: krotki opis, zalety, wady, trudnosc implementacji (1-5)\n3. Top 3 rekomendacje z uzasadnieniem\n4. Plan dzialania dla najlepszego pomyslu",
        "variables": ["topic"],
    },
    {
        "id": "email-professional",
        "name": "Email profesjonalny",
        "icon": "&#9993;",
        "category": "creative",
        "prompt": "Napisz profesjonalny email w kontekscie: {context}\n\nOdbiorca: {recipient}\nCel: {goal}\n\nEmail powinien byc:\n- Zwiezly i klarowny\n- Profesjonalny w tonie\n- Z jasnym call-to-action\n- Z odpowiednim powitaniem i zakonczeniem",
        "variables": ["context", "recipient", "goal"],
    },
    {
        "id": "data-analysis",
        "name": "Analiza danych",
        "icon": "&#128202;",
        "category": "analysis",
        "prompt": "Przeanalizuj nastepujace dane:\n\n{data}\n\nWykonaj:\n1. Podsumowanie statystyczne (srednia, mediana, odchylenie std)\n2. Identyfikacje trendow i wzorcow\n3. Wykrycie anomalii i wartosci odstajacych\n4. Wnioski i rekomendacje\n5. Jezeli to mozliwe - wygeneruj kod Python do wizualizacji",
        "variables": ["data"],
    },
    {
        "id": "compare-technologies",
        "name": "Porownaj technologie",
        "icon": "&#9878;",
        "category": "analysis",
        "prompt": "Porownaj nastepujace technologie/narzedzia: {technologies}\n\nKryteria porownania:\n1. Wydajnosc i skalowalnosc\n2. Latwosc nauki (learning curve)\n3. Ekosystem i spolecznosc\n4. Dokumentacja\n5. Wsparcie korporacyjne\n6. Koszty (licencje, hosting)\n7. Przypadki uzycia (kiedy wybrac co)\n\nPodaj rekomendacje w zaleznosci od kontekstu projektu.",
        "variables": ["technologies"],
    },
    {
        "id": "regex-builder",
        "name": "Buduj wyrażenie regularne",
        "icon": "&#128269;",
        "category": "tools",
        "prompt": "Stworz wyrazenie regularne (regex) dla nastepujacego zadania:\n\n{task}\n\nPodaj:\n1. Wyrazenie regularne\n2. Wyjasnienie kazdej czesci wyrazenia\n3. Przyklady tekstu ktory PASUJE\n4. Przyklady tekstu ktory NIE PASUJE\n5. Wersje dla Python, JavaScript i inne jezyki jezeli sie roznia",
        "variables": ["task"],
    },
]


def _ensure_dir():
    TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)


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
