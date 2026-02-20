# NeuroForge ML/AI - Handoff dla nowego AI

## TL;DR
**NeuroForge Local AI Studio v0.6.0** - kompletna aplikacja do uruchamiania lokalnych LLM i zewnetrznych API AI na GPU AMD RX 9070 XT.
FastAPI backend + vanilla JS frontend + llama.cpp inference + 6 providerow AI + system multi-agent.
**Kod jest GOTOWY i KOMPLETNY** - ~60 plikow, ~12000 linii + 221 testow.

---

## Lokalizacja i Git

- **Repo lokalne:** `/home/user/ML-AI/`
- **Repo docelowe na GitHub:** `https://github.com/arkadiuszpopiel-eng/ML-AI.git`
- **Tymczasowy branch w Projekt-G.01.:** `claude/ml-ai-project-TExUS` (orphan branch, tylko pliki ML-AI)
- **Status:** Commit gotowy lokalnie. Push do ML-AI repo wymaga sesji powiązanej z tym repo (proxy blokuje).

### Jak wrzucić do ML-AI repo:
```bash
git clone https://github.com/arkadiuszpopiel-eng/ML-AI.git
cd ML-AI
git remote add source https://github.com/arkadiuszpopiel-eng/Projekt-G.01..git
git fetch source claude/ml-ai-project-TExUS
git reset --hard source/claude/ml-ai-project-TExUS
git push -u origin main
```

---

## Struktura projektu

```
ML-AI/
├── .gitignore
├── README.md                          # Opis projektu (PL)
├── HANDOFF.md                         # Ten plik
├── rx-9070-xt-ai-ml-analysis.md       # Analiza GPU RX 9070 XT dla AI/ML (12KB)
└── neurostudio/                       # Główna aplikacja NeuroForge
    ├── config.yaml                    # Konfiguracja serwera/modeli/narzędzi
    ├── requirements.txt               # 11 zależności Python
    ├── start.bat                      # All-in-one: install + run (Windows)
    ├── start.sh                       # All-in-one: install + run (Linux/macOS)
    ├── install.py                     # Autoinstalator (używany przez start.*)
    ├── install.bat                    # Stary installer (zachowany dla kompatybilności)
    ├── run.py                         # Launcher serwera (używany przez start.*)
    ├── models/                        # Tu trafiają pliki .gguf
    ├── data/                          # Konwersacje, dokumenty, indeks RAG
    ├── backend/
    │   ├── app.py                     # FastAPI - 40+ endpointów + 3 WebSockety (870+ linii)
    │   ├── config.py                  # Zarządzanie config.yaml (49 linii)
    │   ├── storage.py                 # Persystencja konwersacji JSON (191 linii)
    │   ├── usage.py                   # Śledzenie tokenów/kosztów per provider (151 linii)
    │   ├── monitor.py                 # Monitor CPU/RAM/GPU - AMD + NVIDIA (194 linii)
    │   ├── setup.py                   # Auto-download llama.cpp binary (218 linii)
    │   ├── templates.py               # 12 wbudowanych szablonów promptów PL (171 linii)
    │   ├── inference/
    │   │   ├── __init__.py            # Rejestracja providerów przy imporcie
    │   │   ├── engine.py              # Zarządzanie procesem llama.cpp (191 linii)
    │   │   ├── model_manager.py       # Odkrywanie i pobieranie modeli GGUF (160 linii)
    │   │   ├── router.py              # Routing modeli wg typu zadania (81 linii)
    │   │   ├── providers.py           # BaseProvider ABC + ProviderRegistry (222 linii)
    │   │   ├── provider_local.py      # Wrapper na llama.cpp engine
    │   │   ├── provider_openai.py     # OpenAI API (GPT-4o, GPT-4.1)
    │   │   ├── provider_anthropic.py  # Anthropic API (Claude Opus/Sonnet/Haiku)
    │   │   ├── provider_google.py     # Google Gemini API
    │   │   ├── provider_ollama.py     # Lokalna instancja Ollama
    │   │   └── provider_openrouter.py # OpenRouter (100+ modeli)
    │   ├── agent/
    │   │   ├── loop.py                # Pętla agenta z narzędziami (275 linii)
    │   │   ├── roles.py               # [v0.6] Role agentów: planner/coder/reviewer/researcher
    │   │   ├── shared_context.py      # [v0.6] Pamięć współdzielona, message passing, artefakty
    │   │   └── orchestrator.py        # [v0.6] Orkiestrator multi-agent z równoległym wykonywaniem
    │   ├── rag/
    │   │   └── engine.py              # Silnik TF-IDF RAG (307 linii)
    │   └── tools/
    │       ├── base.py                # Klasa bazowa + rejestr narzędzi (71 linii)
    │       ├── filesystem.py          # Odczyt/zapis/szukanie plików (244 linii)
    │       ├── code_executor.py       # Wykonywanie kodu Python/Shell (104 linii)
    │       ├── web_search.py          # Wyszukiwanie DuckDuckGo (61 linii)
    │       ├── web_fetch.py           # Pobieranie stron HTML (93 linii)
    │       ├── shell.py               # Komendy systemowe + procesy (162 linii)
    │       └── rag_search.py          # Wyszukiwanie w dokumentach RAG (52 linii)
    └── frontend/
        ├── index.html                 # Interfejs UI (400+ linii)
        ├── css/style.css              # Dark theme, responsive (2000+ linii)
        └── js/app.js                  # WebSocket + logika UI (1900+ linii)
```

---

## Co jest GOTOWE (100%)

### Backend
- **FastAPI app** z 40+ REST endpointami + 3 WebSockety (chat + monitor + multi-agent)
- **Inference engine** - zarzadzanie procesem llama-server (start/stop/health check)
- **Model manager** - 13 rekomendowanych modeli z kategoriami, pobieranie z HuggingFace z prawdziwym postepem (SSE)
- **Agent loop** - petla rozumowania z wywolaniami narzedzi, streaming eventow, fallback chain
- **6 providerow AI:** Local (llama.cpp), OpenAI, Anthropic (Claude), Google Gemini, Ollama, OpenRouter
- **Fallback chain** - automatyczne przelaczanie na zapasowego providera gdy glowny zawiedzie
- **Smart routing** - proste pytania -> model lokalny, zlozone -> API w chmurze
- **Usage tracking** - sledzenie tokenow i kosztow per provider z szacunkami cen
- **8 narzedzi:** filesystem, code executor, web search, web fetch, shell, process monitor, RAG search
- **RAG engine** - TF-IDF, chunking z overlap, trwaly indeks na dysku (v2 - instant restart)
- **Storage** - persystencja konwersacji jako JSON + eksport do Markdown/JSON
- **Monitor** - CPU/RAM/dysk/GPU (AMD rocm-smi + NVIDIA nvidia-smi)
- **12 szablonow promptow** po polsku (code review, testy, tlumaczenia, refaktoring...)
- **Auto-setup** - instalacja llama-server z poziomu UI (bez recznego uruchamiania install.py)
- **Multi-Agent System** (v0.6):
  - **Orkiestrator** - koordynacja podzadan, rownolegle wykonywanie, fallback na awarie
  - **4 role agentow:** Planner (planowanie), Coder (kodowanie), Reviewer (recenzja), Researcher (badania)
  - **Shared Context** - pamiec wspoldzielona, message passing miedzy agentami, artefakty
  - **Planner-driven decomposition** - automatyczny podzial zlozonych zadan na podzadania z zaleznosциami
  - **Per-role tool filtering** - kazda rola ma dostep tylko do swoich narzedzi

### Frontend
- **Dark theme** z fioletowym akcentem (#6c5ce7)
- **Real-time chat** przez WebSocket ze streamingiem
- **Syntax highlighting** - kolorowanie kodu (highlight.js) z przyciskiem Kopiuj
- **Panel boczny:** historia konwersacji (z eksportem MD/JSON), zarzadzanie modelami, ustawienia GPU
- **Panel Provider AI** - wybor providera, fallback chain, smart routing, klucze API
- **Dedykowana zakladka Klucze API** - formularze per provider z zapisem/usuwaniem
- **Katalog modeli** - 13 modeli w 5 kategoriach (kodowanie, ogolne, kreatywne, lekkie, powerhouse)
- **Prawdziwy pasek postepu** pobierania modeli (%, MB/s, ETA)
- **Status bar** aktywnego providera z zuzyciem tokenow i kosztem
- **Panel szablonow** z kategoriami (coding, text, tools)
- **Monitor systemowy** z auto-odswiezaniem co 2s
- **Panel Multi-Agent** - uruchamianie zespolu agentow, wizualizacja przeplywu pracy
- **Workflow visualization** - real-time widok podzadan, statusy, wyniki agentow
- **Responsywny** - dziala na mobile

### Instalacja
- **install.py** - tworzy venv, instaluje zaleznosci, pobiera llama.cpp binary (Vulkan)
- **Auto-install z UI** - przycisk "Zainstaluj llama-server" w panelu bocznym
- **install.bat** - one-click dla Windows
- **run.py** - startuje serwer, otwiera przegladarke

---

## Zależności (requirements.txt)

| Pakiet | Wersja | Do czego |
|--------|--------|----------|
| fastapi | ≥0.104.0 | Framework webowy |
| uvicorn[standard] | ≥0.24.0 | Serwer ASGI |
| httpx | ≥0.25.0 | Async HTTP do llama.cpp |
| pyyaml | ≥6.0 | Parsowanie config.yaml |
| aiofiles | ≥23.0 | Async file I/O |
| websockets | ≥12.0 | Protokół WebSocket |
| huggingface-hub | ≥0.20.0 | Pobieranie modeli |
| duckduckgo-search | ≥4.0 | Wyszukiwanie web |
| beautifulsoup4 | ≥4.12.0 | Parsowanie HTML |
| psutil | ≥5.9.0 | Monitoring systemu |
| python-multipart | ≥0.0.6 | Upload plików |

**Runtime:** Python 3.10+, llama.cpp (auto-download przez installer)

---

## API Endpointy

### Modele
- `GET /api/models` - lista lokalnych modeli .gguf
- `GET /api/models/recommended` - 13 rekomendowanych z kategoriami i statusem pobrania
- `POST /api/models/load` - zaladuj model (body: `{filename, gpu_layers, context_size, threads}`)
- `POST /api/models/unload` - wyladuj model
- `POST /api/models/download` - pobierz z HuggingFace (SSE z postepem: %, MB/s, ETA)
- `GET /api/status` - status silnika i systemu

### Providery AI
- `GET /api/providers` - lista providerow z ich statusem i modelami
- `POST /api/providers/activate` - aktywuj providera (body: `{provider_id, model}`)
- `POST /api/providers/key` - ustaw klucz API (body: `{provider_id, api_key}`)
- `DELETE /api/providers/key/{id}` - usun klucz API
- `POST /api/providers/fallback` - ustaw lancuch fallback (body: `{chain: [...]}`)
- `POST /api/providers/smart-routing` - wlacz/wylacz smart routing (body: `{enabled}`)

### Usage Tracking
- `GET /api/usage` - statystyki zuzycia per provider (tokeny, koszt)
- `POST /api/usage/reset` - resetuj statystyki

### Engine Setup
- `GET /api/setup/engine-status` - czy llama-server jest zainstalowany
- `POST /api/setup/install-engine` - automatyczna instalacja llama-server

### Chat
- `WS /ws/chat` - WebSocket real-time chat z agentem
- `POST /api/chat` - HTTP fallback (body: `{message, session_id}`)

### Konwersacje
- `GET /api/conversations` - lista zapisanych
- `GET /api/conversations/{id}` - zaladuj konkretna
- `GET /api/conversations/{id}/export?format=markdown|json` - eksport konwersacji
- `DELETE /api/conversations/{id}` - usun
- `PATCH /api/conversations/{id}` - zmien tytul

### Monitor
- `GET /api/monitor` - snapshot CPU/RAM/dysk/GPU
- `GET /api/monitor/processes` - top 15 procesow
- `WS /ws/monitor` - real-time co 2s

### RAG/Dokumenty
- `GET /api/documents` - lista zindeksowanych
- `POST /api/documents/upload` - upload i indeksuj
- `POST /api/documents/index-text` - indeksuj surowy tekst
- `DELETE /api/documents/{id}` - usun dokument
- `GET /api/documents/search` - szukaj w dokumentach

### Router
- `GET /api/router` - konfiguracja routera
- `POST /api/router` - aktualizuj (body: `{enabled, assignments}`)
- `POST /api/router/test` - testuj klasyfikacje (body: `{message}`)

### Szablony
- `GET /api/templates` - lista szablonow
- `POST /api/templates` - utworz wlasny
- `DELETE /api/templates/{id}` - usun

### Multi-Agent (v0.6)
- `GET /api/agents/roles` - lista dostepnych rol agentow
- `POST /api/agents/run` - uruchom workflow multi-agent (SSE stream z eventami)
- `GET /api/agents/workflows` - lista aktywnych/zakonczonych workflows
- `GET /api/agents/workflows/{id}` - szczegoly workflow (podzadania, wiadomosci, artefakty)
- `WS /ws/agents` - WebSocket do real-time multi-agent (alternatywa dla SSE)

### Pliki i konfiguracja
- `POST /api/upload` - upload pliku do chatu (max 50MB)
- `GET /api/config` - pobierz konfiguracje
- `POST /api/config` - zaktualizuj konfiguracje

---

## Rekomendowane modele (13 wbudowanych, 5 kategorii)

| Model | Rozmiar | Kategoria | Zastosowanie |
|-------|---------|-----------|-------------|
| Qwen2.5-Coder-7B Q4_K_M | 4.7 GB | Kodowanie | Kod Python, JS, TS |
| Qwen2.5-Coder-14B Q4_K_M | 8.9 GB | Kodowanie | Code review, refaktoring |
| DeepSeek-Coder-V2-Lite Q4_K_M | 9.4 GB | Kodowanie | Debugging, generowanie |
| Qwen2.5-7B-Instruct Q4_K_M | 4.7 GB | Ogolne | Szybki, polski |
| Qwen2.5-14B-Instruct Q4_K_M | 8.9 GB | Ogolne | Lepsza jakosc |
| Llama-3.1-8B-Instruct Q4_K_M | 4.9 GB | Ogolne | Tool use, angielski |
| Gemma-2-9B-Instruct Q4_K_M | 5.8 GB | Ogolne | Analiza, rozumowanie |
| Mistral-Nemo-12B Q4_K_M | 7.1 GB | Ogolne | Function calling |
| Phi-4-14B Q4_K_M | 8.4 GB | Ogolne | Matematyka, rozumowanie |
| Mistral-Small-24B Q4_K_M | 14.1 GB | Kreatywne | Dlugie teksty, tlumaczenia |
| Llama-3.1-70B-Instruct Q4_K_M | 42.0 GB | Powerhouse | Poziom GPT-4 (48GB+ RAM) |
| Qwen2.5-3B-Instruct Q4_K_M | 2.0 GB | Lekkie | Slabsze komputery |
| Llama-3.2-3B-Instruct Q4_K_M | 2.0 GB | Lekkie | Proste pytania |

---

## Kluczowe decyzje architektoniczne

1. **llama.cpp (Vulkan)** zamiast ROCm - stabilniejszy na RX 9070 XT, działa out-of-box
2. **TF-IDF RAG** zamiast wektorowej bazy - zero dodatkowych zależności, działa lokalnie
3. **Vanilla JS** zamiast React/Vue - brak build stepu, prostota, zero zależności frontend
4. **WebSocket** dla chatu - streaming tokenów w real-time
5. **DuckDuckGo** zamiast Google/Bing - bez klucza API
6. **JSON file storage** zamiast SQLite - prostota, czytelność

---

## Architektura Multi-Agent (v0.6)

### Przeplyw workflow (3 fazy)

```
Uzytkownik: "Przeanalizuj ten kod i popraw bledy"
         │
    ┌────▼────┐
    │ PLANNER  │  Faza 1: Planowanie
    │          │  - Analizuje zadanie
    │  JSON:   │  - Generuje liste podzadan
    │ subtasks │  - Przypisuje role i zaleznosci
    └────┬─────┘
         │
    ┌────▼────────────────────────────┐
    │    ORCHESTRATOR (Faza 2)        │
    │                                 │
    │  Dependency graph:              │
    │  [1] Research (researcher) ──┐  │
    │  [2] Code fix (coder) ◄──────┤  │
    │  [3] Review (reviewer) ◄─────┘  │
    │                                 │
    │  Parallel: [1] runs alone       │
    │  Then: [2] after [1] completes  │
    │  Then: [3] after [2] completes  │
    └────┬────────────────────────────┘
         │
    ┌────▼────┐
    │ SYNTH.  │  Faza 3: Synteza
    │         │  - Laczy wyniki agentow
    │  Final  │  - Tworzy spojna odpowiedz
    │ response│  - Wysyla do uzytkownika
    └─────────┘
```

### Role agentow

| Rola | Narzedzia | Kolor | Cel |
|------|-----------|-------|-----|
| **Planner** | brak | #6c5ce7 | Analizuje zadanie, tworzy plan JSON z podzadaniami |
| **Coder** | read_file, execute_code, run_command, search_documents | #00b894 | Pisze/debuguje kod, uzywa narzedzi |
| **Reviewer** | read_file, search_documents | #fdcb6e | Recenzuje kod (read-only), szuka bledow |
| **Researcher** | web_search, web_fetch, search_documents, read_file | #74b9ff | Zbiera informacje z internetu i dokumentow |

### Shared Context (pamiec wspoldzielona)

Kazdy workflow ma wlasna instancje `SharedContext` zawierajaca:
- **Subtasks** - lista podzadan ze statusami (pending/running/completed/failed/skipped)
- **Messages** - log wiadomosci miedzy agentami (typed: TASK, RESULT, QUESTION, INFO, ERROR)
- **Artifacts** - named storage na dane produkowane przez agentow (kod, wyniki, dokumenty)
- **Context Summary** - automatycznie budowany kontekst dla kazdego agenta (wyniki poprzednikow + wiadomosci)

### Eventy SSE (Server-Sent Events)

Workflow emituje real-time eventy przez SSE/WebSocket:

```
workflow_start  → {workflow_id, task}
planning        → {status: "Planner analizuje..."}
plan_ready      → {subtasks: [{id, title, role, description, depends_on}]}
subtask_start   → {subtask: {...}, agent: "coder"}
subtask_complete → {subtask_id, result}
subtask_error   → {subtask_id, error}
progress_update → {progress: {total, completed, failed, running, pending, percent}}
synthesis       → {status: "Lacze wyniki..."}
workflow_complete → {workflow_id, result, progress}
workflow_error  → {message}
```

### Pliki modulu multi-agent

| Plik | Klasy/Funkcje | Rozmiar |
|------|---------------|---------|
| `agent/roles.py` | AgentRole, PLANNER/CODER/REVIEWER/RESEARCHER, get_role(), list_roles() | ~120 linii |
| `agent/shared_context.py` | SharedContext, Subtask, AgentMessage, MessageType, SubtaskStatus | ~200 linii |
| `agent/orchestrator.py` | MultiAgentOrchestrator.run_workflow(), _run_planner(), _run_subtask(), _synthesize() | ~250 linii |

---

## Naprawione bugi (v0.6.0)

- **[FIX] Windows pobierał plik ubuntu zamiast win** - matchowanie `vulkan-x64` było za ogólne,
  łapało zarówno `win-vulkan-x64` jak i `ubuntu-vulkan-x64`. Naprawiono: platform-specific keywords
  (`win-vulkan-x64` dla Windows, `ubuntu-vulkan-x64` dla Linux).
- **[FIX] Ekstrakcja .tar.gz jako .zip** - llama.cpp przeszedł z .zip na .tar.gz dla Linux.
  Kod używał `zipfile.ZipFile` co failowało na .tar.gz. Naprawiono: auto-detekcja formatu archiwum
  (.zip, .tar.gz, .tar.xz, .tar.bz2) w install.py i setup.py.
- **Dotyczy:** `install.py` (start.bat/start.sh) + `backend/setup.py` (auto-install z UI)

---

## Znane ograniczenia (NIE bugi)

- Indeks RAG jest w pamięci (dokumenty zapisane na dysku, ale indeks przebudowywany po restarcie)
- Model router wyłączony domyślnie (opcjonalna funkcja)
- GPU monitoring wymaga rocm-smi lub nvidia-smi (graceful fallback)
- WSL2 + GPU = niestabilne (zalecany natywny Windows)

---

## Mapa rozwoju (Roadmap)

### v0.1 - Fundament (GOTOWE)
> Podstawowa aplikacja: chat z lokalnym LLM przez przeglądarkę.

- [x] FastAPI backend z 20+ REST endpointami + 2 WebSockety
- [x] Inference engine - zarządzanie procesem llama-server (start/stop/health)
- [x] Model manager - 5 rekomendowanych modeli, pobieranie z HuggingFace
- [x] Agent loop - pętla rozumowania z wywołaniami narzędzi
- [x] 8 narzędzi: filesystem, code executor, web search, web fetch, shell, process monitor, RAG search
- [x] 12 szablonów promptów PL (code review, testy, tłumaczenia, refaktoring...)
- [x] Dark theme UI z fioletowym akcentem, responsive, WebSocket chat
- [x] Historia konwersacji, zarządzanie modelami, ustawienia GPU
- [x] Monitor systemowy (CPU/RAM/dysk/GPU) z auto-odświeżaniem
- [x] Instalator: install.py + install.bat + run.py

### v0.2 - Semantic Router (GOTOWE)
> Inteligentne przełączanie modeli na podstawie typu zadania.

- [x] Detekcja typu zadania (coding/analysis/creative/chat) wg słów kluczowych
- [x] Router z config.yaml (przypisanie modelu do typu zadania)
- [x] Automatyczne przełączanie modelu w agent loop
- [x] **Panel UI** do konfiguracji routera (włącz/wyłącz, przypisania modeli per task type)
- [x] **Router domyślnie włączony** (enabled: true w config.yaml)
- [x] **TF-IDF semantic scoring** (cosine similarity z profilami zadań, keyword fallback)
- [x] **API:** GET/POST /api/router + POST /api/router/test (klasyfikacja z wizualizacją)
- [x] **Testy:** 24 testy jednostkowe (SemanticScorer, detect, config, fallback)

### v0.3 - RAG + Quality (GOTOWE)
> Baza wiedzy z dokumentów + testy + bezpieczeństwo + Docker.

- [x] Silnik TF-IDF RAG - chunking z overlap, indeksowanie, search
- [x] Upload i indeksowanie dokumentów przez UI
- [x] Automatyczne wstrzykiwanie kontekstu RAG do promptów agenta
- [x] API: upload, index-text, search, delete dokumentów
- [x] **Testy jednostkowe (pytest)** - 120 testów w 8 modułach
- [x] **Docker Compose** - Dockerfile + docker-compose.yml z named volumes
- [x] **Bezpieczeństwo:** fix XSS, path traversal, command injection, onclick injection
- [x] **Temperature** podłączony: UI slider -> WebSocket -> agent -> inference
- [x] **Atomic writes** w storage (temp file + rename)
- [x] **Logging** zamiast cichego połykania błędów w monitorze GPU
- [x] **.gitignore** + pyproject.toml + zunifikowane start.bat/start.sh

### v0.4 - External AI APIs (GOTOWE)
> Podpiecie zewnetrznych providerow AI obok lokalnego llama.cpp.

- [x] **Abstrakcja providerow** - BaseProvider ABC + ProviderRegistry (wspolny interfejs)
- [x] **6 providerow:** Local, OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter
- [x] **UI: panel providerow** - wybor, model selector, klucze API, status bar
- [x] **Usage tracking** - tokeny/koszty per provider z szacunkami cen
- [x] **Testy:** 58 testow (36 providers + 22 usage tracking)

### v0.5 - Hybrid Mode + UX (GOTOWE)
> Tryb hybrydowy, lepszy UX, nowe funkcje.

- [x] **Fallback chain** - automatyczne przelaczanie na zapasowego providera
- [x] **Smart routing** - proste pytania → model lokalny, zlozone → API w chmurze
- [x] **Auto-install llama-server** - instalacja silnika z poziomu UI (bez install.py)
- [x] **Dedykowana zakladka Klucze API** - formularze per provider z zapisem/usuwaniem
- [x] **Rozszerzony katalog modeli** - 13 modeli w 5 kategoriach z opisami PL
- [x] **Prawdziwy postep pobierania** - SSE stream z %, MB/s, ETA (nie fake progress)
- [x] **Syntax highlighting** - kolorowanie kodu w czacie (highlight.js + atom-one-dark)
- [x] **Przycisk Kopiuj kod** - na kazdym bloku kodu w czacie
- [x] **Eksport konwersacji** - pobieranie rozmow jako Markdown lub JSON
- [x] **Trwaly indeks RAG v2** - pelny zapis TF-IDF na dysk (instant restart)
- [x] **Karta aktywnego providera** - info + przycisk dezaktywacji
- [x] **Testy:** 167 testow razem

### v0.6 - Multi-Agent (GOTOWE - aktualny stan)
> Kilka agentow AI wspolpracuje nad zlozonym zadaniem.

- [x] **Orkiestrator agentow** - 3-fazowy workflow: Planning → Execution → Synthesis
- [x] **4 role agentow:** Planner (planowanie), Coder (kod), Reviewer (recenzja), Researcher (badania)
- [x] **Komunikacja miedzy agentami** - typed message passing (TASK, RESULT, QUESTION, INFO, ERROR)
- [x] **Shared context** - pamiec wspoldzielona, artefakty, context summary per agent
- [x] **Per-role tool filtering** - kazda rola ma dostep tylko do swoich narzedzi (np. Reviewer = read-only)
- [x] **Rownolegle wykonywanie** - podzadania bez zaleznosci wykonuja sie jednoczesnie (asyncio.gather)
- [x] **Dependency graph** - podzadania z depends_on, automatyczne schedulowanie
- [x] **Preferred provider per role** - kazdy agent moze uzywac innego modelu/providera
- [x] **UI: panel Multi-Agent** w sidebarze - opis zadania, uruchomienie zespolu, pasek postepu
- [x] **UI: workflow visualization** - floating panel z real-time widokiem podzadan i wynikow
- [x] **API:** GET /api/agents/roles, POST /api/agents/run (SSE), GET /api/agents/workflows
- [x] **WebSocket:** WS /ws/agents - real-time workflow events
- [x] **Testy:** 54 nowe testy (role, shared context, orchestrator) = 221 testow razem

### v0.7 - Vision (DO ZROBIENIA)
> Analiza obrazow i screenshotow przez modele multimodalne.

- [ ] Obsluga modeli multimodalnych (LLaVA, Qwen-VL, GPT-4o vision)
- [ ] Upload i analiza obrazow w chacie
- [ ] Screenshot tool - przechwytywanie ekranu
- [ ] OCR z obrazow (wyciaganie tekstu)
- [ ] Generowanie opisow obrazow
- [ ] UI: podglad obrazow w konwersacji

### v1.0 - Production Release (DO ZROBIENIA)
> Dopracowany produkt gotowy do codziennego uzytku.

- [ ] **System pluginow** - dynamiczne ladowanie narzedzi z katalogu plugins/
- [ ] **Profile uzytkownikow** - ustawienia, historia, preferencje per user
- [ ] **Polished UI** - animacje, onboarding, lepszy UX
- [ ] **i18n** - wielojezyczne szablony i interfejs (PL + EN)
- [ ] **Eksport PDF** - eksport konwersacji jako PDF
- [ ] **Rate limiting + auth token** - bezpieczenstwo API
- [ ] **PWA** - manifest + service worker (offline mode)
- [ ] **Wiecej narzedzi** - git, baza danych SQL, API caller, image gen

---

## Uruchomienie

**Jeden plik robi wszystko** (instalacja + uruchomienie):

### Windows
```cmd
cd neurostudio
start.bat
```

### Linux/macOS
```bash
cd neurostudio
chmod +x start.sh
./start.sh
```

Skrypty automatycznie:
1. Sprawdzają Pythona
2. Tworzą venv (jeśli brak)
3. Instalują/aktualizują zależności (skip jeśli requirements.txt się nie zmienił)
4. Pobierają llama.cpp (jeśli brak)
5. Tworzą katalogi danych
6. Uruchamiają serwer

Aplikacja startuje na `http://localhost:7860`

---

## Kontekst: RX 9070 XT

Plik `rx-9070-xt-ai-ml-analysis.md` zawiera pełną analizę GPU:
- ROCm support timeline (gfx1201) - marzec 2025 → styczeń 2026
- Benchmarki llama.cpp: prompt processing 5055 t/s, generowanie 101.3 t/s
- Porównanie z NVIDIA (63% lepszy stosunek cena/wydajność)
- Zalecenie: **Vulkan backend** (nie ROCm/HIP) dla stabilności
- Ekosystem: Ollama, LM Studio, PyTorch DirectML, vLLM
