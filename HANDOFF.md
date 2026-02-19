# NeuroForge ML/AI - Handoff dla nowego AI

## TL;DR
**NeuroForge Local AI Studio v0.3.0** - kompletna aplikacja do uruchamiania lokalnych LLM na GPU AMD RX 9070 XT.
FastAPI backend + vanilla JS frontend + llama.cpp inference. **Kod jest GOTOWY i KOMPLETNY** - ~45 plików, ~6500 linii + 60 testów.

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
    │   ├── app.py                     # FastAPI - 20+ endpointów + 2 WebSockety (466 linii)
    │   ├── config.py                  # Zarządzanie config.yaml (49 linii)
    │   ├── storage.py                 # Persystencja konwersacji JSON (94 linii)
    │   ├── monitor.py                 # Monitor CPU/RAM/GPU - AMD + NVIDIA (184 linii)
    │   ├── templates.py               # 12 wbudowanych szablonów promptów PL (171 linii)
    │   ├── inference/
    │   │   ├── engine.py              # Zarządzanie procesem llama.cpp (191 linii)
    │   │   ├── model_manager.py       # Odkrywanie i pobieranie modeli GGUF (160 linii)
    │   │   └── router.py             # Routing modeli wg typu zadania (81 linii)
    │   ├── agent/
    │   │   └── loop.py                # Pętla agenta z narzędziami (173 linii)
    │   ├── rag/
    │   │   └── engine.py              # Silnik TF-IDF RAG (276 linii)
    │   └── tools/
    │       ├── base.py                # Klasa bazowa + rejestr narzędzi (71 linii)
    │       ├── filesystem.py          # Odczyt/zapis/szukanie plików (244 linii)
    │       ├── code_executor.py       # Wykonywanie kodu Python/Shell (104 linii)
    │       ├── web_search.py          # Wyszukiwanie DuckDuckGo (61 linii)
    │       ├── web_fetch.py           # Pobieranie stron HTML (93 linii)
    │       ├── shell.py               # Komendy systemowe + procesy (162 linii)
    │       └── rag_search.py          # Wyszukiwanie w dokumentach RAG (52 linii)
    └── frontend/
        ├── index.html                 # Interfejs UI (226 linii)
        ├── css/style.css              # Dark theme, responsive (1075 linii)
        └── js/app.js                  # WebSocket + logika UI (861 linii)
```

---

## Co jest GOTOWE (100%)

### Backend
- **FastAPI app** z 20+ REST endpointami + 2 WebSockety (chat + monitor)
- **Inference engine** - zarządzanie procesem llama-server (start/stop/health check)
- **Model manager** - 5 rekomendowanych modeli, pobieranie z HuggingFace
- **Agent loop** - pętla rozumowania z wywołaniami narzędzi, streaming eventów
- **8 narzędzi:** filesystem, code executor, web search, web fetch, shell, process monitor, RAG search
- **RAG engine** - TF-IDF, chunking z overlap, indeksowanie dokumentów
- **Storage** - persystencja konwersacji jako JSON
- **Monitor** - CPU/RAM/dysk/GPU (AMD rocm-smi + NVIDIA nvidia-smi)
- **12 szablonów promptów** po polsku (code review, testy, tłumaczenia, refaktoring...)

### Frontend
- **Dark theme** z fioletowym akcentem (#6c5ce7)
- **Real-time chat** przez WebSocket ze streamingiem
- **Panel boczny:** historia konwersacji, zarządzanie modelami, ustawienia GPU, upload dokumentów
- **Panel szablonów** z kategoriami (coding, text, tools)
- **Monitor systemowy** z auto-odświeżaniem co 2s
- **Responsywny** - działa na mobile

### Instalacja
- **install.py** - tworzy venv, instaluje zależności, pobiera llama.cpp binary (Vulkan)
- **install.bat** - one-click dla Windows
- **run.py** - startuje serwer, otwiera przeglądarkę

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
- `GET /api/models/recommended` - 5 rekomendowanych z statusem pobrania
- `POST /api/models/load` - załaduj model (body: `{filename, gpu_layers, context_size, threads}`)
- `POST /api/models/unload` - wyładuj model
- `POST /api/models/download` - pobierz z HuggingFace (body: `{repo_id, filename}`)
- `GET /api/status` - status silnika i systemu

### Chat
- `WS /ws/chat` - WebSocket real-time chat z agentem
- `POST /api/chat` - HTTP fallback (body: `{message, session_id}`)

### Konwersacje
- `GET /api/conversations` - lista zapisanych
- `GET /api/conversations/{id}` - załaduj konkretną
- `DELETE /api/conversations/{id}` - usuń
- `PATCH /api/conversations/{id}` - zmień tytuł

### Monitor
- `GET /api/monitor` - snapshot CPU/RAM/dysk/GPU
- `GET /api/monitor/processes` - top 15 procesów
- `WS /ws/monitor` - real-time co 2s

### RAG/Dokumenty
- `GET /api/documents` - lista zindeksowanych
- `POST /api/documents/upload` - upload i indeksuj
- `POST /api/documents/index-text` - indeksuj surowy tekst
- `DELETE /api/documents/{id}` - usuń dokument
- `GET /api/documents/search` - szukaj w dokumentach

### Szablony
- `GET /api/templates` - lista szablonów
- `POST /api/templates` - utwórz własny
- `DELETE /api/templates/{id}` - usuń

### Pliki i konfiguracja
- `POST /api/upload` - upload pliku do chatu (max 50MB)
- `GET /api/config` - pobierz konfigurację
- `POST /api/config` - zaktualizuj konfigurację

---

## Rekomendowane modele (wbudowane)

| Model | Rozmiar | Zastosowanie |
|-------|---------|-------------|
| Qwen2.5-7B-Instruct Q4_K_M | 4.7 GB | Ogólny, szybki |
| Qwen2.5-14B-Instruct Q4_K_M | 8.9 GB | Lepszy ogólny |
| Qwen2.5-Coder-7B-Instruct Q4_K_M | 4.7 GB | Kodowanie |
| Llama-3.1-8B-Instruct Q4_K_M | 4.9 GB | Meta, ogólny |
| Mistral-Nemo-12B-Instruct Q4_K_M | 7.1 GB | Kreatywny |

---

## Kluczowe decyzje architektoniczne

1. **llama.cpp (Vulkan)** zamiast ROCm - stabilniejszy na RX 9070 XT, działa out-of-box
2. **TF-IDF RAG** zamiast wektorowej bazy - zero dodatkowych zależności, działa lokalnie
3. **Vanilla JS** zamiast React/Vue - brak build stepu, prostota, zero zależności frontend
4. **WebSocket** dla chatu - streaming tokenów w real-time
5. **DuckDuckGo** zamiast Google/Bing - bez klucza API
6. **JSON file storage** zamiast SQLite - prostota, czytelność

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

### v0.4 - External AI APIs (GOTOWE - aktualny stan)
> Podpięcie zewnętrznych providerów AI obok lokalnego llama.cpp.

- [x] **Abstrakcja providerów** - BaseProvider ABC + ProviderRegistry (wspólny interfejs)
- [x] **LocalProvider** - wrapper na istniejący llama.cpp engine
- [x] **OpenAI API** - GPT-4o, GPT-4o-mini, GPT-4.1, o3-mini (gotowe po dodaniu klucza)
- [x] **Anthropic API** - Claude Sonnet 4.5, Haiku 4.5, Opus 4.6 (z translacją formatów)
- [x] **Google Gemini API** - Gemini 2.5 Flash/Pro, 2.0 Flash (z translacją formatów)
- [x] **Ollama** - integracja z lokalnym Ollama (OpenAI-compatible endpoint)
- [x] **OpenRouter** - jeden klucz API → 8+ prekonfigurowanych modeli
- [x] **UI: panel providerów** - wybór providera, model selector, dodawanie kluczy API
- [x] **Agent loop integration** - cloud provider lub local engine, transparentnie
- [x] **API:** GET /api/providers, POST activate/key, DELETE key + GET /api/usage, POST reset
- [x] **Config:** sekcja providers w config.yaml z persystencją kluczy
- [x] **Provider status bar** - widoczny na górze chatu (nazwa providera, model, tokeny, koszt)
- [x] **Usage tracking** - śledzenie tokenów/kosztów per provider z szacunkiem cen
- [x] **Panel zużycia** - karta per provider z: requests, tokeny in/out, koszt szacunkowy
- [x] **Keys grid** - chipy w sidebarze pokazujące które API są skonfigurowane
- [x] **Real-time updates** - WebSocket `usage_update` aktualizuje status bar na bieżąco
- [x] **Testy:** 58 testów (36 providers + 22 usage tracking), 142 razem
- [ ] **TODO (v0.5):** Routing po providerze (coding→Qwen, creative→Claude)
- [ ] **TODO (v0.5):** Tryb hybrydowy - lokalne dla prostych, chmurowe dla trudnych

### v0.5 - Multi-Agent (DO ZROBIENIA)
> Kilka agentów AI współpracuje nad złożonym zadaniem.

- [ ] Orkiestrator agentów - koordynacja zadań między agentami
- [ ] Role agentów (planer, coder, reviewer, researcher)
- [ ] Komunikacja między agentami (message passing)
- [ ] Każdy agent może używać innego modelu/providera (lokalne + chmurowe)
- [ ] UI: wizualizacja przepływu pracy agentów
- [ ] Równoległe wykonywanie podzadań
- [ ] Shared context / pamięć współdzielona między agentami

### v0.6 - Vision (DO ZROBIENIA)
> Analiza obrazów i screenshotów przez modele multimodalne.

- [ ] Obsługa modeli multimodalnych (LLaVA, Qwen-VL, GPT-4o vision)
- [ ] Upload i analiza obrazów w chacie
- [ ] Screenshot tool - przechwytywanie ekranu
- [ ] OCR z obrazów (wyciąganie tekstu)
- [ ] Generowanie opisów obrazów
- [ ] UI: podgląd obrazów w konwersacji

### v1.0 - Production Release (DO ZROBIENIA)
> Dopracowany produkt gotowy do codziennego użytku.

- [ ] **System pluginów** - dynamiczne ładowanie narzędzi z katalogu plugins/
- [ ] **Profile użytkowników** - ustawienia, historia, preferencje per user
- [ ] **Polished UI** - animacje, onboarding, lepszy UX
- [ ] **i18n** - wielojęzyczne szablony i interfejs (PL + EN)
- [ ] **Eksport konwersacji** - markdown, PDF, JSON
- [ ] **Rate limiting + auth token** - bezpieczeństwo API
- [ ] **PWA** - manifest + service worker (offline mode)
- [ ] **Persistent RAG index** - zapis na dysk zamiast rebuild po restarcie
- [ ] **Streaming HTTP** - SSE w REST endpoint (nie tylko WebSocket)
- [ ] **Więcej narzędzi** - git, baza danych SQL, API caller, image gen

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
