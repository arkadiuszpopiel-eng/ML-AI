# ML-AI

Repozytorium projektow AI/ML z lokalnym uruchamianiem na GPU.

## Projekty

### NeuroForge - Local AI Studio
Pelne lokalne studio AI z agentem, narzediami i interfejsem webowym.
Dziala na Twoim komputerze bez chmury - zero API keys, pelna prywatnosc.

**Funkcjonalnosci:**
- Chat z lokalnymi modelami LLM (llama.cpp + Vulkan GPU)
- System agenta z narzediami (pliki, kod, shell, web, RAG)
- Historia konwersacji z auto-zapisem
- System RAG - indeksowanie dokumentow i wyszukiwanie kontekstu
- Monitor systemu (CPU/RAM/GPU) w czasie rzeczywistym
- 12 wbudowanych szablonow promptow
- Upload plikow i zalaczniki w czacie
- Responsywny dark-theme UI

**Szybki start:**
```bash
cd neurostudio
# Windows:
install.bat
NeuroForge.bat

# Linux/macOS:
python install.py
./neurostudio.sh
```

**Wymagania:**
- Python 3.10+
- 8 GB RAM (16 GB zalecane)
- GPU z Vulkan (AMD RX 9070 XT, NVIDIA, Intel Arc)

### Dokumentacja
- `rx-9070-xt-ai-ml-analysis.md` - Analiza AI/ML dla AMD RX 9070 XT

## Technologie
- Python, FastAPI, WebSocket
- llama.cpp (Vulkan backend)
- Vanilla JS, CSS Grid/Flexbox
- TF-IDF RAG engine
- psutil, rocm-smi/nvidia-smi
