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
    {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "description": "Fast, great for coding and chat. Fits entirely in 16GB VRAM.",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size": "4.7 GB",
        "capabilities": ["coding", "chat", "analysis"],
    },
    {
        "name": "Qwen2.5-14B-Instruct (Q4_K_M)",
        "description": "Balanced power and speed. Fits in 16GB VRAM.",
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size": "8.9 GB",
        "capabilities": ["coding", "chat", "analysis", "creative"],
    },
    {
        "name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M)",
        "description": "Specialized for code generation and programming tasks.",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size": "4.7 GB",
        "capabilities": ["coding"],
    },
    {
        "name": "Llama-3.1-8B-Instruct (Q4_K_M)",
        "description": "Meta's flagship 8B model. Great general purpose with tool use support.",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size": "4.9 GB",
        "capabilities": ["chat", "analysis", "coding"],
    },
    {
        "name": "Mistral-Nemo-12B-Instruct (Q4_K_M)",
        "description": "Mistral's 12B model with strong function calling support.",
        "repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "size": "7.1 GB",
        "capabilities": ["chat", "coding", "analysis"],
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
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    models_dir = get_models_dir()
    logger.info("Downloading %s from %s...", filename, repo)

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )

    logger.info("Model downloaded to: %s", path)
    return path


def get_recommended_models() -> list[dict]:
    """Get list of recommended models with download status."""
    models_dir = get_models_dir()
    result = []
    for model in RECOMMENDED_MODELS:
        model_info = dict(model)
        model_info["downloaded"] = (models_dir / model["filename"]).exists()
        result.append(model_info)
    return result
