"""
Model Manager - discovers, downloads, and manages GGUF model files.
"""
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import get_models_dir

logger = logging.getLogger("neurostudio.models")


@dataclass
class ModelInfo:
    """Information about a local GGUF model."""
    filename: str
    path: str
    size_bytes: int
    size_display: str
    quantization: str
    base_name: str

    def to_dict(self) -> dict:
        return asdict(self)


# Common recommended models with HuggingFace download info
RECOMMENDED_MODELS = [
    # ── Coding ──
    {
        "name": "Qwen2.5-Coder-7B (Q4_K_M)",
        "description": "Specjalizowany model do programowania. Swietny w generowaniu kodu Python, JS, TS. Szybki, 4.7 GB.",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size": "4.7 GB",
        "category": "coding",
        "capabilities": ["coding"],
    },
    {
        "name": "Qwen2.5-Coder-14B (Q4_K_M)",
        "description": "Mocniejszy model do kodu. Lepiej rozumie kontekst, refaktoryzacje i code review. 8.9 GB.",
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "size": "8.9 GB",
        "category": "coding",
        "capabilities": ["coding", "analysis"],
    },
    {
        "name": "DeepSeek-Coder-V2-Lite (Q4_K_M)",
        "description": "Model DeepSeek do kodu. Dobry w generowaniu, debugowaniu i wyjasniananiu kodu. 9.4 GB.",
        "repo": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "size": "9.4 GB",
        "category": "coding",
        "capabilities": ["coding", "analysis"],
    },
    # ── General / Chat ──
    {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "description": "Uniwersalny model do czatu, analizy i kodowania. Szybki, dobrze po polsku. 4.7 GB.",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size": "4.7 GB",
        "category": "general",
        "capabilities": ["coding", "chat", "analysis"],
    },
    {
        "name": "Qwen2.5-14B-Instruct (Q4_K_M)",
        "description": "Mocniejszy uniwersalny model. Lepsze rozumienie, lepsza jakosc tekstu. 8.9 GB.",
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size": "8.9 GB",
        "category": "general",
        "capabilities": ["coding", "chat", "analysis", "creative"],
    },
    {
        "name": "Llama-3.1-8B-Instruct (Q4_K_M)",
        "description": "Model Meta. Swietny do czatu i narzedzi (tool use). Dobrze zna angielski. 4.9 GB.",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size": "4.9 GB",
        "category": "general",
        "capabilities": ["chat", "analysis", "coding"],
    },
    {
        "name": "Gemma-2-9B-Instruct (Q4_K_M)",
        "description": "Model Google. Bardzo dobry w analizie, rozumowaniu i podazaniu za instrukcjami. 5.8 GB.",
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size": "5.8 GB",
        "category": "general",
        "capabilities": ["analysis", "chat", "creative"],
    },
    {
        "name": "Mistral-Nemo-12B-Instruct (Q4_K_M)",
        "description": "Model Mistral 12B. Silne function calling, dobry wielojezyczny. 7.1 GB.",
        "repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "size": "7.1 GB",
        "category": "general",
        "capabilities": ["chat", "coding", "analysis"],
    },
    {
        "name": "Phi-4-14B (Q4_K_M)",
        "description": "Model Microsoft. Wybitny w rozumowaniu i matematyce. Kompaktowy i szybki. 8.4 GB.",
        "repo": "bartowski/phi-4-GGUF",
        "filename": "phi-4-Q4_K_M.gguf",
        "size": "8.4 GB",
        "category": "general",
        "capabilities": ["analysis", "coding", "chat"],
    },
    # ── Creative / Text ──
    {
        "name": "Mistral-Small-24B (Q4_K_M)",
        "description": "Duzy model Mistral. Doskonaly do dlugich tekstow, tlumazen i kreatywnego pisania. 14.1 GB.",
        "repo": "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        "filename": "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        "size": "14.1 GB",
        "category": "creative",
        "capabilities": ["creative", "chat", "analysis"],
    },
    {
        "name": "Llama-3.1-70B-Instruct (Q4_K_M)",
        "description": "Najsilniejszy otwarty model Meta. Wymaga 48GB+ RAM. Poziom GPT-4 klasy. 42 GB.",
        "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "size": "42.0 GB",
        "category": "powerhouse",
        "capabilities": ["coding", "chat", "analysis", "creative"],
    },
    # ── Lightweight / Fast ──
    {
        "name": "Qwen2.5-3B-Instruct (Q4_K_M)",
        "description": "Bardzo maly i szybki. Idealny na slabsze komputery lub jako pomocniczy model. 2.0 GB.",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size": "2.0 GB",
        "category": "lightweight",
        "capabilities": ["chat"],
    },
    {
        "name": "Llama-3.2-3B-Instruct (Q4_K_M)",
        "description": "Maly model Meta. Szybki, do prostych pytan i narzedzi. 2.0 GB.",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "2.0 GB",
        "category": "lightweight",
        "capabilities": ["chat"],
    },
]


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def _extract_quantization(filename: str) -> str:
    """Extract quantization type from filename."""
    patterns = [
        r'[_-](Q\d[\w_]*)',
        r'[_-](q\d[\w_]*)',
        r'[_-](f16|f32|bf16)',
        r'[_-](IQ\d[\w_]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "unknown"


def _extract_base_name(filename: str) -> str:
    """Extract base model name from filename."""
    name = filename.replace(".gguf", "")
    # Remove quantization suffix
    name = re.sub(r'[_-](?:Q\d[\w_]*|q\d[\w_]*|f16|f32|bf16|IQ\d[\w_]*)\s*$', '', name, flags=re.IGNORECASE)
    return name


def list_models() -> list[ModelInfo]:
    """List all GGUF models in the models directory."""
    models_dir = get_models_dir()
    models = []
    for f in models_dir.glob("*.gguf"):
        stat = f.stat()
        models.append(ModelInfo(
            filename=f.name,
            path=str(f),
            size_bytes=stat.st_size,
            size_display=_format_size(stat.st_size),
            quantization=_extract_quantization(f.name),
            base_name=_extract_base_name(f.name),
        ))
    models.sort(key=lambda m: m.filename)
    return models


def get_model_path(filename: str) -> str | None:
    """Get full path for a model by filename."""
    models_dir = get_models_dir()
    path = models_dir / filename
    if path.exists():
        return str(path)
    return None


async def download_model(repo: str, filename: str, progress_callback=None) -> str:
    """Download a model from HuggingFace Hub.

    Args:
        repo: HuggingFace repo id (e.g. "Qwen/Qwen2.5-7B-Instruct-GGUF")
        filename: Model filename (e.g. "qwen2.5-7b-instruct-q4_k_m.gguf")
        progress_callback: optional callable(downloaded_bytes, total_bytes)
            called during download to report progress.
    """
    import asyncio
    from huggingface_hub import hf_hub_download

    models_dir = get_models_dir()
    logger.info("Downloading %s from %s...", filename, repo)

    tqdm_cls = None
    if progress_callback:
        tqdm_cls = _make_progress_tqdm(progress_callback)

    path = await asyncio.to_thread(
        hf_hub_download,
        repo_id=repo,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
        tqdm_class=tqdm_cls,
    )

    logger.info("Model downloaded to: %s", path)
    return path


def _make_progress_tqdm(callback):
    """Create a custom tqdm-compatible class that calls a progress callback."""

    class ProgressTqdm:
        """Minimal tqdm-compatible wrapper that reports progress via callback."""

        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.n = kwargs.get("initial", 0)
            self.desc = kwargs.get("desc", "")
            self.unit = kwargs.get("unit", "it")

        def update(self, n=1):
            self.n += n
            try:
                callback(self.n, self.total)
            except Exception:
                pass

        def close(self):
            pass

        def set_description(self, desc):
            self.desc = desc

        def set_postfix(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return ProgressTqdm


def get_recommended_models() -> list[dict]:
    """Get list of recommended models with download status."""
    models_dir = get_models_dir()
    result = []
    for model in RECOMMENDED_MODELS:
        model_info = dict(model)
        model_info["downloaded"] = (models_dir / model["filename"]).exists()
        result.append(model_info)
    return result
