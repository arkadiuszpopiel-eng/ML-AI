# AMD Radeon RX 9070 XT: Stan wsparcia AI/ML (luty 2026)

**RX 9070 XT (RDNA 4, gfx1201) stala sie realna karta do lokalnego AI inference, ale wymaga cierpliwosci i technicznej bieglosci -- szczegolnie na Windows.** Oficjalne wsparcie ROCm pojawilo sie dopiero w maju 2025 (ROCm 6.4.1), niemal trzy miesiace po premierze karty. Od tego czasu ekosystem szybko dojrzewa: ROCm 7.2 (styczen 2026) oferuje pelne wsparcie na Linuxie i PyTorch na Windows, llama.cpp dziala zarowno przez HIP, jak i Vulkan, a LM Studio oficjalnie obsluguje karte. Mimo to, w porownaniu z NVIDIA CUDA, setup AMD wciaz wymaga wiecej recznej konfiguracji -- zwlaszcza na Windows, gdzie pelny stack ROCm nie jest dostepny, a narzedzia takie jak Ollama wymagaja workaroundow.

---

## Spis tresci

1. [ROCm dla RDNA 4: historia wsparcia](#rocm-dla-rdna-4-historia-wsparcia)
2. [llama.cpp: backendy i benchmarki](#llamacpp-backendy-i-benchmarki)
3. [Ekosystem narzedzi AI](#ekosystem-narzedzi-ai)
4. [Porownanie z NVIDIA](#porownanie-z-nvidia)
5. [WSL2 i Windows 11](#wsl2-i-windows-11)
6. [Rekomendowany workflow Windows 11](#rekomendowany-workflow-windows-11)
7. [Wnioski](#wnioski)

---

## ROCm dla RDNA 4: historia wsparcia

Premiera RX 9070 XT 6 marca 2025 roku odbyla sie **bez oficjalnego wsparcia ROCm** -- AMD obiecalo je "po premierze". Chronologia wsparcia:

| Wersja ROCm | Data (przybliz.) | Status RDNA 4 |
|---|---|---|
| 6.3.x | Poczatek 2025 | gfx1201 w build scripts; nieoficjalne, czesciowe wsparcie |
| 6.4.0 | Kwiecien 2025 | Brak oficjalnego wsparcia RDNA 4 |
| **6.4.1** | **Maj 2025** | **Pierwsze oficjalne wsparcie (tylko Linux)** |
| 6.4.4 | ~Wrzesien 2025 | Pierwszy preview PyTorch na Windows dla RDNA 3+4 |
| 7.0.2 | Koniec 2025 | hipBLAS dla gfx1201; dodano RX 9060 |
| 7.1.1 | Listopad 2025 | Walidacja modeli AI (Phi-4, Qwen QwQ-32B) na gfx1201 |
| **7.2.0** | **~Styczen 2026** | **Aktualna wersja produkcyjna; pelna macierz kompatybilnosci** |

### Kluczowe informacje

- **Minimalna wersja z dzialajacym wsparciem:** ROCm 6.4.1
- **Rekomendowana wersja:** ROCm 7.0+ (implementuje WMMA -- Wave Matrix Multiply Accumulate dla RDNA 4; bez niej wydajnosc jest drastycznie nizsza)
- **Linux:** wymagany kernel 6.14+ (domyslny kernel 6.8 w Ubuntu 24.04 nie rozpoznaje karty). Wspierane dystrybucje: Ubuntu 22.04.5/24.04.3 oraz RHEL 9.7/10.1
- **Windows:** pelny stack ROCm **nie jest dostepny**. Jedynym oficjalnie wspieranym komponentem jest PyTorch 2.9 z ROCm 7.2 na Windows 11

### Ograniczenia na Windows

- Batch size LLM ograniczony do 1
- Brak `torch.distributed`
- Koniecznosc wylaczenia Smart App Control
- Sporadyczne crashe przy jednoczesnym uruchamianiu innych aplikacji

### Znane problemy ROCm na RDNA 4

- **Bug idle power w HIP** -- karta utrzymuje podwyzszone zegary po zakonczeniu obliczen (GitHub: ROCm/ROCm#5706)
- **Suboptymalna wydajnosc MUL_MAT** -- kompilator ROCm generuje kod osiagajacy ~3.09 TFLOPS zamiast ~4.94 TFLOPS po recznym patchu AMD (GitHub: ROCm/ROCm#5727)
- Bledy konwolucji MIOpen na gfx1201
- Niestabilnosc przy multi-GPU

---

## llama.cpp: backendy i benchmarki

Wsparcie gfx1201 w llama.cpp dodano w **PR #12372** (marzec 2025, autor: inzynier AMD). Karta jest wykrywana jako `AMD Radeon RX 9070 XT, gfx1201 (0x1201), Wave Size: 32`.

### Backend Vulkan

- **Najprostsza sciezka** -- dziala out-of-the-box na Windows i Linuxie bez instalacji ROCm
- Karta obsluguje `KHR_coopmat` (cooperative matrix) -- umozliwia akceleracje macierzowa
- Na Linuxie z Mesa 25.3+ i kernelem 6.17 wydajnosc Vulkan dorownuje lub przewyzsza Windows

### Backend HIP/ROCm

- Wyzsza wydajnosc prompt processing niz Vulkan
- Wymaga ROCm 7.0+ dla pelnej funkcjonalnosci WMMA
- Na ROCm 6.4 flash attention z WMMA **nie kompiluje sie** (blad `mma_sync` -- issue #13110)
- AMD udostepnia **oficjalne, prekompilowane binaria** llama.cpp z ROCm 7.2 dla Windows: `repo.radeon.com/rocm/llama.cpp/windows/rocm-rel-7.2/`
- Alternatywa: projekt **lemonade-sdk/llamacpp-rocm** -- nightly builds z wbudowanymi bibliotekami ROCm 7

### Benchmarki: Llama 2 7B Q4_0 (HIP, ROCm 7+)

| Konfiguracja | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| RX 9070 XT (bez FA) | 4,065 | 89.9 |
| RX 9070 XT (z FA, ROCm 7.1) | **5,055** | **101.3** |
| RX 7900 XTX (bez FA) | 3,552 | 167.1 |
| RX 7800 XT (bez FA) | 2,152 | 100.9 |

**Prompt processing (pp512):** RX 9070 XT dominuje -- do 5,055 t/s z flash attention vs 3,552 t/s na 7900 XTX. Zasluga ulepszonej architektury compute w RDNA 4.

**Token generation (tg128):** 7900 XTX jest ~65% szybszy (~167 vs ~101 t/s), poniewaz generowanie tokenow jest ograniczone przepustowoscia pamieci: 9070 XT ma **256 GB/s** (szyna 256-bit) vs **960 GB/s** na 7900 XTX (384-bit). To fundamentalne ograniczenie hardwareowe.

### Benchmarki mniejsze modele (LocalScore.ai)

- Llama 3.2 1B Q4_K_M: **163 t/s**
- Llama 3.2 3B Q3_K_L: **81 t/s**
- Modele 70B: wymagaja agresywnej kwantyzacji i offloadu na CPU -- przy 16 GB VRAM nie sa praktyczne

### Kompilacja na Linuxie (ROCm 7+)

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
-DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j $(nproc)
```

---

## Ekosystem narzedzi AI

### Ollama

| Platforma | Status | Uwagi |
|---|---|---|
| Linux | Dziala | Obraz Docker `ollama/ollama:rocm` z ROCm 6.4.1+ |
| Windows | Problemy | Oficjalnie nie rozpoznaje gfx1201, spada na CPU |

**Obejscia na Windows:**
1. Community fork: `likelovewant/ollama-for-amd` z wbudowanymi bibliotekami ROCm dla gfx1201
2. Tryb Vulkan: `OLLAMA_VULKAN=1` -- omija ROCm calkowicie

Otwarte issues: #10430, #11542, #12573

### LM Studio

**Dziala na obu platformach.** AMD oficjalnie prezentuje LM Studio 0.3.21b4 z llama.cpp runtime 1.44 na RX 9070 XT (blog AMD, sierpien 2025). Issue #574 w bug trackerze zamkniety z etykieta "beta-build-available". Na Windows korzysta z backendu llama.cpp (ROCm lub Vulkan).

### PyTorch

| Platforma | Status | Wersja |
|---|---|---|
| Linux | Wsparcie od ROCm 6.4+ | Stabilne |
| Windows | PyTorch 2.9 z ROCm 7.2 | Tylko Windows 11, brak torch.distributed |

FP8 wspierane **tylko na RDNA 4**. Instalacja na Windows przez `repo.radeon.com`.

### DirectML

Dziala z kazdym GPU DirectX 12 -- w tym z RX 9070 XT -- ale jest **4-6x wolniejszy od ROCm**. Uzyteczny jako universalny fallback na Windows, nie nadaje sie do wydajnego inference.

### Vulkan compute

Szybko dojrzewajacy backend. Obsluga `KHR_coopmat` na RDNA 4 umozliwia akceleracje macierzowa. Na Linuxie z najnowszym RADV (Mesa 25.3+) wydajnosc dorownuje lub przewyzsza Windows. Ollama, llama.cpp, text-generation-webui i KoboldCpp oferuja backendy Vulkan.

### vLLM

Eksperymentalnie -- oficjalnie targetuje GPU Instinct (MI series). Czlonek community zdolal uruchomic natywne FP8 WMMA na gfx1201 przez patche kerneli Triton, osiagajac dwukrotny wzrost prefill performance. Nie jest upstream (GitHub: vllm-project/vllm#28649).

### text-generation-webui (oobabooga)

Dziala -- dostepne **portable buildy Vulkan** dla AMD/Intel, nie wymagajace instalacji. Na Linuxie dostepne rowniez buildy ROCm.

### KoboldCpp-ROCm

**Nie dziala** na RX 9070 XT z domyslnymi targetami -- wymaga rebuildu z dodanym gfx1201. Standardowy KoboldCpp z backendem Vulkan moze dzialac.

---

## Porownanie z NVIDIA

### RX 9070 XT vs RTX 4070 Ti Super (16 GB)

| Benchmark | Wynik |
|---|---|
| Stable Diffusion (UL Procyon) | NVIDIA prowadzi o ~14% (3,031 vs 2,646 pkt) |
| Geekbench AI (single precision) | RX 9070 XT: 35,496 |
| Geekbench AI (quantized) | RX 9070 XT: 30,662 -- **9% lepsza** od RTX 5080 FE |
| MLPerf Client 0.5 (LLaMA 2 7B) | AMD: wyzszy throughput (t/s), gorsza latencja TTFT |

### RX 9070 XT vs RTX 4080

Deficyt wyrazniejszy -- NVIDIA prowadzi o ~28% w agregatowych benchmarkach. W Procyon AI Text Generation (DirectML) RX 9070 XT traci 30-50% vs RTX 4070 (test nie wykorzystuje ROCm).

### Kluczowa przewaga AMD: cena

- RX 9070 XT: ~$550-600
- RTX 4070 Ti Super: ~$800+
- RTX 4080: ~$1,000+
- **63% lepszy stosunek ceny do wydajnosci** (wg Technical.city)

### Realne wyniki llama.cpp

- RX 9070 XT: **~90-101 t/s** dla 7B Q4_0 (tg128)
- RX 7900 XTX (24 GB, szersza szyna): **~167 t/s**
- **Waskie gardlo:** przepustowosc pamieci 256 GB/s
- **Dominacja** w prompt processing (compute-bound)

---

## WSL2 i Windows 11

**WSL2 + ROCm dla RX 9070 XT to loteria w lutym 2026.**

- Oficjalna dokumentacja AMD opisuje instalacje przez `amdgpu-install` ze sterownikiem Adrenalin 25.8.1 dla WSL2
- ROCm 7.2 explicite wymienia wsparcie WSL2
- W praktyce: liczne raporty (GitHub issues #4471, #4490, Microsoft WSL #13535) dokumentuja **fundamentalny problem z ladowaniem modulu kernela amdgpu** w WSL2
- Jeden uzytkownik potwierdzil dzialanie rocminfo i PyTorch na WSL2 (Windows 10, Ubuntu 22.04, ROCm 6.4.1), ale kolejni nie sa w stanie powtorzyc sukcesu

**Nie jest to stabilna sciezka w lutym 2026.**

---

## Rekomendowany workflow Windows 11

### LLM inference (najlepsza sciezka)

1. Zainstaluj **LM Studio** (0.3.21b4+) -- oficjalnie wspiera RX 9070 XT
2. LUB pobierz **prekompilowane binaria llama.cpp z ROCm 7.2** od AMD (`repo.radeon.com`)
3. LUB llama.cpp z backendem **Vulkan** (zero konfiguracji)
4. Wlacz **Flash Attention** i ogranicz context length do 8192 tokenow dla modeli bliskich limitu 16 GB VRAM

### Ollama na Windows

- Uzyj forka `likelovewant/ollama-for-amd`
- LUB trybu `OLLAMA_VULKAN=1`

### PyTorch

- Zainstaluj PyTorch 2.9 z ROCm 7.2 z oficjalnego repo AMD

### Rekomendowane modele (16 GB VRAM)

- **7B-14B z kwantyzacja Q4-Q8** -- komfortowo mieszcza sie w 16 GB VRAM
- Modele MoE (np. GPT-OSS 20B z ~3.6B aktywnych parametrow) rowniez dobrze dzialaja

### Czego unikac

- Pelnego ROCm stack na Windows (niedostepny)
- WSL2 (niestabilny)
- Modeli 70B (wymagaja offloadu na CPU, niepraktyczne)

### Dla zaawansowanych workloadow

Linux dual-boot pozostaje rekomendacja -- pelny ROCm stack, stabilne wsparcie, lepsza wydajnosc.

---

## Wnioski

RX 9070 XT w lutym 2026 to **realnie uzyteczna karta do lokalnego AI inference** z kilkoma istotnymi zastrzezeniami:

**Zalety:**
- **16 GB VRAM za ~$550** -- atrakcyjna propozycja cenowa
- Wydajnosc compute (prompt processing) przewyzsza nawet znacznie drozsza 7900 XTX
- Rosnacy ekosystem: LM Studio, llama.cpp (HIP + Vulkan), PyTorch na Windows

**Ograniczenia:**
- **256 GB/s przepustowosci pamieci** -- obniza predkosc generowania tokenow (najistotniejszy parametr w interaktywnych czatach)
- Gap wobec CUDA: setup wymaga wiecej krokow, nie wszystkie narzedzia dzialaja out-of-box na Windows
- Bugi: idle power (ROCm#5706), MUL_MAT performance (ROCm#5727)

**Perspektywy:**
- AMD deklaruje, ze pierwsza polowa 2026 przyniesie "zmiane podejscia" w nowej major wersji ROCm
- Minor releases co 6 tygodni beda systematycznie rozszerzac wsparcie
- Dla uzytkownika sklonnego poswiecic czas na konfiguracje (szczegolnie na Linuxie) -- 9070 XT oferuje najlepszy stosunek VRAM-do-ceny na rynku
- Dla tych, ktorzy cenia bezproblemowoscf -- NVIDIA wciaz jest bezpieczniejszym wyborem

---

## Zrodla

- ROCm 7.2.0 Release Notes: https://rocm.docs.amd.com/en/latest/about/release-notes.html
- Windows support matrices: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
- llama.cpp PR #12372 (gfx1201 support): https://github.com/ggml-org/llama.cpp/pull/12372
- llama.cpp ROCm HIP performance: https://github.com/ggml-org/llama.cpp/discussions/15021
- llama.cpp Vulkan performance: https://github.com/ggml-org/llama.cpp/discussions/10879
- AMD llama.cpp ROCm binaries: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/windows/llm/llamacpp.html
- lemonade-sdk/llamacpp-rocm: https://github.com/lemonade-sdk/llamacpp-rocm
- LM Studio RX 9070 XT issue: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/574
- Ollama AMD fork: https://github.com/likelovewant/ollama-for-amd/releases
- LocalScore.ai RX 9070 XT results: https://www.localscore.ai/accelerator/585
- Phoronix Linux compute review: https://www.phoronix.com/review/amd-radeon-rx9070-linux-compute
- ROCm idle power bug: https://github.com/ROCm/ROCm/issues/5706
- ROCm MUL_MAT performance: https://github.com/ROCm/ROCm/issues/5727
- WSL2 issues: https://github.com/ROCm/ROCm/issues/4490
