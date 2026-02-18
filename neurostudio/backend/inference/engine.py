"""
Inference Engine - manages llama.cpp server process and communicates via OpenAI-compatible API.
"""
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import httpx

from ..config import load_config, get_llama_server_path

logger = logging.getLogger("neurostudio.engine")


class InferenceEngine:
    """Manages a llama.cpp server process and provides inference via its API."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.current_model: str | None = None
        self.api_base: str = "http://127.0.0.1:8081"
        self._client: httpx.AsyncClient | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.api_base, timeout=300.0)
        return self._client

    async def start(self, model_path: str, **kwargs) -> bool:
        """Start llama.cpp server with the given model."""
        if self.is_running:
            if self.current_model == model_path:
                logger.info("Model already loaded: %s", model_path)
                return True
            await self.stop()

        server_path = get_llama_server_path()
        if not server_path:
            logger.error("llama-server binary not found. Run install.py first.")
            return False

        config = load_config()
        inf_config = config.get("inference", {})

        ctx_size = kwargs.get("context_size", inf_config.get("context_size", 8192))
        gpu_layers = kwargs.get("gpu_layers", inf_config.get("gpu_layers", -1))
        threads = kwargs.get("threads", inf_config.get("threads", 8))

        cmd = [
            server_path,
            "--model", model_path,
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(gpu_layers),
            "--threads", str(threads),
            "--host", "127.0.0.1",
            "--port", "8081",
            "--jinja",
        ]

        logger.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except FileNotFoundError:
            logger.error("llama-server binary not found at: %s", server_path)
            return False
        except Exception as e:
            logger.error("Failed to start llama-server: %s", e)
            return False

        # Wait for server to be ready
        if await self._wait_for_ready(timeout=120):
            self.current_model = model_path
            logger.info("llama-server started successfully with model: %s", Path(model_path).name)
            return True
        else:
            logger.error("llama-server failed to start within timeout")
            await self.stop()
            return False

    async def _wait_for_ready(self, timeout: int = 120) -> bool:
        """Wait for llama-server to respond to health checks."""
        start = time.time()
        client = await self.get_client()
        while time.time() - start < timeout:
            if not self.is_running:
                return False
            try:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        return True
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(1)
        return False

    async def stop(self):
        """Stop the llama.cpp server process."""
        if self.process:
            logger.info("Stopping llama-server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.current_model = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> AsyncIterator[dict] | dict:
        """Send a chat completion request to llama.cpp server."""
        if not self.is_running:
            raise RuntimeError("Inference engine is not running. Load a model first.")

        client = await self.get_client()
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        if stream:
            return self._stream_completion(client, payload)
        else:
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _stream_completion(self, client: httpx.AsyncClient, payload: dict) -> AsyncIterator[dict]:
        """Stream chat completion response."""
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    async def get_status(self) -> dict:
        """Get current engine status."""
        status = {
            "running": self.is_running,
            "model": Path(self.current_model).name if self.current_model else None,
            "model_path": self.current_model,
        }
        if self.is_running:
            try:
                client = await self.get_client()
                resp = await client.get("/health")
                status["health"] = resp.json()
            except Exception:
                status["health"] = {"status": "unreachable"}
        return status


# Singleton instance
engine = InferenceEngine()
