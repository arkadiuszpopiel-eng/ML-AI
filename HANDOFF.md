# NeuroForge ML/AI - Handoff dla nowego AI

## TL;DR
**NeuroForge Local AI Studio v0.2.0** - kompletna aplikacja do uruchamiania lokalnych LLM na GPU AMD RX 9070 XT.
FastAPI backend + vanilla JS frontend + llama.cpp inference. **Kod jest GOTOWY i KOMPLETNY** - 36 plików, ~5700 linii.

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
    ├── install.py                     # Autoinstalator (Linux/macOS/Windows)
    ├── install.bat                    # One-click installer Windows
    ├── run.py                         # Launcher aplikacji
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

## Co MOŻNA robić dalej (pomysły na rozwój)

- [ ] Testy jednostkowe (pytest)
- [ ] Docker compose
- [ ] Persistent RAG index (zamiast rebuild po restarcie)
- [ ] Streaming response w HTTP endpoint (nie tylko WebSocket)
- [ ] Więcej narzędzi (git, baza danych, API caller)
- [ ] Eksport konwersacji (markdown, PDF)
- [ ] Wielojęzyczne szablony (EN obok PL)
- [ ] Plugin system dla narzędzi
- [ ] Bezpieczeństwo: rate limiting, auth token
- [ ] PWA manifest + service worker

---

## Uruchomienie

### Windows
```cmd
cd neurostudio
install.bat
NeuroForge.bat
```

### Linux/macOS
```bash
cd neurostudio
python install.py
python run.py
```

Aplikacja startuje na `http://localhost:7860`

---

## Kontekst: RX 9070 XT

Plik `rx-9070-xt-ai-ml-analysis.md` zawiera pełną analizę GPU:
- ROCm support timeline (gfx1201) - marzec 2025 → styczeń 2026
- Benchmarki llama.cpp: prompt processing 5055 t/s, generowanie 101.3 t/s
- Porównanie z NVIDIA (63% lepszy stosunek cena/wydajność)
- Zalecenie: **Vulkan backend** (nie ROCm/HIP) dla stabilności
- Ekosystem: Ollama, LM Studio, PyTorch DirectML, vLLM
