"""
NeuroForge - Main FastAPI application.
Serves the web UI and provides API endpoints for chat, model management,
conversation history, RAG, system monitor, templates, and file upload.
"""
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import load_config, save_config, get_models_dir, BASE_DIR
from .inference.engine import engine
from .inference.providers import provider_registry
from .inference.model_manager import (
    list_models, get_model_path, download_model, get_recommended_models
)
from .agent.loop import agent
from .agent.orchestrator import orchestrator
from .agent.roles import list_roles_dict, get_role, update_role_config
from .inference.router import (
    get_router_config, update_router_config, detect_task_type, get_task_scores
)
from .usage import usage_tracker
from .setup import get_engine_status, install_engine
from .storage import (
    save_conversation, load_conversation, list_conversations,
    delete_conversation, update_conversation_title,
    export_conversation_markdown,
)
from .monitor import get_system_snapshot, get_process_info
from .rag.engine import rag_engine
from .templates import get_all_templates, get_template, add_custom_template, delete_custom_template

logger = logging.getLogger("neurostudio")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("NeuroForge starting up...")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("NeuroForge shutting down...")
    await engine.stop()


app = FastAPI(title="NeuroForge", version="0.7.0", lifespan=lifespan)

# Mount static files
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# ──────────────────────── Pages ────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ──────────────────────── Model Management ────────────────────────

@app.get("/api/models")
async def api_list_models():
    """List all available local models."""
    models = list_models()
    return {"models": [m.to_dict() for m in models]}


@app.get("/api/models/recommended")
async def api_recommended_models():
    """List recommended models with download status."""
    return {"models": get_recommended_models()}


class ModelLoadRequest(BaseModel):
    filename: str
    gpu_layers: int = -1
    context_size: int = 8192
    threads: int = 8


@app.post("/api/models/load")
async def api_load_model(req: ModelLoadRequest):
    """Load a local GGUF model into the llama.cpp inference engine.
    Automatically switches active provider to 'local' since loading a GGUF
    model only makes sense with the local engine.
    """
    model_path = get_model_path(req.filename)
    if not model_path:
        raise HTTPException(404, f"Model not found: {req.filename}")

    # Check if llama-server binary exists before trying to start
    from .config import get_llama_server_path
    if not get_llama_server_path():
        raise HTTPException(
            503,
            "llama-server nie znaleziony. Uruchom install.py lub uzyj providera API (OpenAI, Ollama itp.)"
        )

    # Auto-switch to local provider when loading a local model
    if provider_registry._active_provider_id != "local":
        provider_registry.set_active("local")
        config = load_config()
        if "providers" not in config:
            config["providers"] = {}
        config["providers"]["active"] = "local"
        save_config(config)

    success = await engine.start(
        model_path,
        gpu_layers=req.gpu_layers,
        context_size=req.context_size,
        threads=req.threads,
    )
    if not success:
        raise HTTPException(500, "Nie udalo sie uruchomic silnika. Sprawdz logi.")

    return {"success": True, "model": req.filename, "provider": "local"}


@app.post("/api/models/unload")
async def api_unload_model():
    """Unload the current model."""
    await engine.stop()
    return {"success": True}


class ModelDownloadRequest(BaseModel):
    repo: str
    filename: str


@app.post("/api/models/download")
async def api_download_model(req: ModelDownloadRequest):
    """Download a model from HuggingFace with real-time progress via SSE."""
    import asyncio
    import time

    queue: asyncio.Queue = asyncio.Queue()
    start_time = time.time()
    last_report = {"time": start_time, "bytes": 0}

    def progress_callback(downloaded: int, total: int):
        now = time.time()
        elapsed = now - start_time
        speed = downloaded / elapsed if elapsed > 0 else 0
        eta = (total - downloaded) / speed if speed > 0 and total > 0 else 0
        pct = (downloaded / total * 100) if total > 0 else 0

        # Throttle to max 4 updates/sec
        if now - last_report["time"] < 0.25 and downloaded < total:
            return
        last_report["time"] = now
        last_report["bytes"] = downloaded

        try:
            queue.put_nowait({
                "type": "progress",
                "downloaded": downloaded,
                "total": total,
                "pct": round(pct, 1),
                "speed": round(speed / (1024 * 1024), 2),  # MB/s
                "eta": round(eta),
            })
        except asyncio.QueueFull:
            pass

    async def event_stream():
        download_task = asyncio.create_task(_run_download(req.repo, req.filename, progress_callback, queue))

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if download_task.done():
                    break
                continue

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") in ("done", "error"):
                break

        if not download_task.done():
            await download_task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _run_download(repo: str, filename: str, progress_callback, queue: asyncio.Queue):
    """Run model download and push final result to queue."""
    try:
        path = await download_model(repo, filename, progress_callback=progress_callback)
        await queue.put({"type": "done", "success": True, "path": path})
    except Exception as e:
        logger.error("Download failed: %s", e)
        await queue.put({"type": "error", "success": False, "message": str(e)})


# ──────────────────────── Engine Status ────────────────────────

@app.get("/api/status")
async def api_status():
    """Get current engine and system status."""
    status = await engine.get_status()
    models = list_models()
    config = load_config()
    return {
        "engine": status,
        "models_count": len(models),
        "config": {
            "inference": config.get("inference", {}),
            "router": config.get("router", {}),
        },
    }


# ──────────────────────── Configuration ────────────────────────

@app.get("/api/config")
async def api_get_config():
    return load_config()


class ConfigUpdateRequest(BaseModel):
    config: dict


@app.post("/api/config")
async def api_update_config(req: ConfigUpdateRequest):
    current = load_config()
    _deep_merge(current, req.config)
    save_config(current)
    return {"success": True}


def _deep_merge(base: dict, override: dict):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ──────────────────────── Providers ────────────────────────

@app.get("/api/providers")
async def api_list_providers():
    """List all available providers with their status."""
    return provider_registry.to_dict()


class ProviderActivateRequest(BaseModel):
    provider_id: str
    model: str | None = None


@app.post("/api/providers/activate")
async def api_activate_provider(req: ProviderActivateRequest):
    """Set the active provider and optionally select a model."""
    provider = provider_registry.get(req.provider_id)
    if not provider:
        raise HTTPException(404, f"Unknown provider: {req.provider_id}")

    if not provider_registry.set_active(req.provider_id):
        raise HTTPException(400, f"Provider not available: {req.provider_id}")

    if req.model:
        provider._active_model = req.model

    # Persist active provider choice
    config = load_config()
    if "providers" not in config:
        config["providers"] = {}
    config["providers"]["active"] = req.provider_id
    save_config(config)

    return {"success": True, "active": provider_registry.to_dict()}


class ProviderKeyRequest(BaseModel):
    provider_id: str
    api_key: str


@app.post("/api/providers/key")
async def api_set_provider_key(req: ProviderKeyRequest):
    """Set the API key for a provider."""
    provider = provider_registry.get(req.provider_id)
    if not provider:
        raise HTTPException(404, f"Unknown provider: {req.provider_id}")

    # Update provider config in memory
    provider.config["api_key"] = req.api_key

    # Persist to config.yaml
    config = load_config()
    if "providers" not in config:
        config["providers"] = {}
    if req.provider_id not in config["providers"]:
        config["providers"][req.provider_id] = {}
    config["providers"][req.provider_id]["api_key"] = req.api_key
    save_config(config)

    return {"success": True, "provider": provider.to_dict()}


@app.delete("/api/providers/key/{provider_id}")
async def api_delete_provider_key(provider_id: str):
    """Remove the API key for a provider."""
    provider = provider_registry.get(provider_id)
    if not provider:
        raise HTTPException(404, f"Unknown provider: {provider_id}")

    provider.config.pop("api_key", None)

    config = load_config()
    providers_cfg = config.get("providers", {})
    if provider_id in providers_cfg:
        providers_cfg[provider_id].pop("api_key", None)
        save_config(config)

    # If this was active, fallback to local
    if provider_registry._active_provider_id == provider_id:
        provider_registry.set_active("local")
        config["providers"]["active"] = "local"
        save_config(config)

    return {"success": True}


# ──────────────────────── Fallback & Smart Routing ────────────────────────

class FallbackChainRequest(BaseModel):
    chain: list[str]


@app.post("/api/providers/fallback")
async def api_set_fallback_chain(req: FallbackChainRequest):
    """Set the fallback chain - ordered list of provider IDs to try."""
    provider_registry.set_fallback_chain(req.chain)

    # Persist
    config = load_config()
    config.setdefault("providers", {})["fallback_chain"] = req.chain
    save_config(config)

    return {"success": True, "chain": provider_registry.fallback_chain}


class SmartRoutingRequest(BaseModel):
    enabled: bool


@app.post("/api/providers/smart-routing")
async def api_set_smart_routing(req: SmartRoutingRequest):
    """Enable or disable smart routing (simple→local, complex→cloud)."""
    provider_registry.set_smart_routing(req.enabled)

    config = load_config()
    config.setdefault("providers", {})["smart_routing"] = req.enabled
    save_config(config)

    return {"success": True, "smart_routing": provider_registry.smart_routing}


# ──────────────────────── Usage Tracking ────────────────────────

@app.get("/api/usage")
async def api_get_usage():
    """Get current usage statistics for all providers."""
    data = usage_tracker.to_dict()
    # Enrich with provider info
    for p_usage in data["providers"]:
        provider = provider_registry.get(p_usage["provider_id"])
        if provider:
            p_usage["provider_name"] = provider.provider_name
    data["active_provider"] = provider_registry._active_provider_id
    return data


@app.post("/api/usage/reset")
async def api_reset_usage():
    """Reset all usage statistics."""
    usage_tracker.reset()
    return {"success": True}


# ──────────────────────── Engine Setup ────────────────────────

@app.get("/api/setup/engine-status")
async def api_engine_status():
    """Check if llama-server binary is installed."""
    return get_engine_status()


@app.post("/api/setup/install-engine")
async def api_install_engine():
    """Download and install llama-server binary automatically."""
    result = await install_engine()
    if not result["success"]:
        raise HTTPException(500, result["message"])
    return result


# ──────────────────────── Router ────────────────────────

@app.get("/api/router")
async def api_get_router():
    """Get router configuration with available models."""
    return get_router_config()


class RouterUpdateRequest(BaseModel):
    enabled: bool | None = None
    assignments: dict | None = None


@app.post("/api/router")
async def api_update_router(req: RouterUpdateRequest):
    """Update router configuration (enable/disable, model assignments)."""
    return update_router_config(enabled=req.enabled, assignments=req.assignments)


class RouterTestRequest(BaseModel):
    message: str


@app.post("/api/router/test")
async def api_test_router(req: RouterTestRequest):
    """Test the router - classify a message and return scores."""
    task_type = detect_task_type(req.message)
    scores = get_task_scores(req.message)
    return {
        "message": req.message,
        "detected_type": task_type,
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


# ──────────────────────── Chat (WebSocket) ────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """WebSocket endpoint for real-time chat with the agent."""
    await ws.accept()
    session_id = str(uuid.uuid4())

    try:
        await ws.send_json({"type": "session", "session_id": session_id})

        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "message":
                user_msg = data.get("content", "")
                if not user_msg.strip():
                    continue

                temperature = data.get("temperature", 0.7)
                try:
                    temperature = max(0.0, min(2.0, float(temperature)))
                except (TypeError, ValueError):
                    temperature = 0.7

                async for event in agent.process_message(session_id, user_msg, temperature=temperature):
                    await ws.send_json(event)

                # Auto-save conversation after each exchange
                conversation = agent.conversations.get(session_id, [])
                if conversation:
                    save_conversation(session_id, conversation)

            elif msg_type == "clear":
                agent.clear_conversation(session_id)
                await ws.send_json({"type": "cleared"})

            elif msg_type == "set_session":
                new_id = data.get("session_id")
                if new_id:
                    # Load saved conversation into agent memory
                    saved = load_conversation(new_id)
                    if saved and saved.get("messages"):
                        agent.conversations[new_id] = saved["messages"]
                    session_id = new_id
                    await ws.send_json({"type": "session", "session_id": session_id})

    except WebSocketDisconnect:
        # Save on disconnect
        conversation = agent.conversations.get(session_id, [])
        if conversation:
            save_conversation(session_id, conversation)
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ──────────────────────── Chat (HTTP fallback) ────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """HTTP endpoint for chat (non-streaming, for simple clients)."""
    session_id = req.session_id or str(uuid.uuid4())
    events = []

    async for event in agent.process_message(session_id, req.message):
        events.append(event)

    text_parts = [e["content"] for e in events if e["type"] == "text"]
    tool_events = [e for e in events if e["type"] in ("tool_call", "tool_result")]

    return {
        "response": "".join(text_parts),
        "tool_events": tool_events,
        "session_id": session_id,
    }


# ──────────────────────── Conversation History ────────────────────────

@app.get("/api/conversations")
async def api_list_conversations():
    """List all saved conversations."""
    return {"conversations": list_conversations()}


@app.get("/api/conversations/{session_id}")
async def api_get_conversation(session_id: str):
    """Get a specific conversation."""
    data = load_conversation(session_id)
    if not data:
        raise HTTPException(404, "Conversation not found")
    return data


@app.delete("/api/conversations/{session_id}")
async def api_delete_conversation(session_id: str):
    """Delete a saved conversation."""
    agent.clear_conversation(session_id)
    delete_conversation(session_id)
    return {"success": True}


class ConversationRenameRequest(BaseModel):
    title: str


@app.patch("/api/conversations/{session_id}")
async def api_rename_conversation(session_id: str, req: ConversationRenameRequest):
    """Rename a conversation."""
    if not update_conversation_title(session_id, req.title):
        raise HTTPException(404, "Conversation not found")
    return {"success": True}


@app.get("/api/conversations/{session_id}/export")
async def api_export_conversation(session_id: str, format: str = "markdown"):
    """Export a conversation as downloadable file."""
    if format == "markdown":
        md = export_conversation_markdown(session_id)
        if not md:
            raise HTTPException(404, "Conversation not found")
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="conversation-{session_id[:8]}.md"'},
        )
    elif format == "json":
        data = load_conversation(session_id)
        if not data:
            raise HTTPException(404, "Conversation not found")
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="conversation-{session_id[:8]}.json"'},
        )
    else:
        raise HTTPException(400, f"Unknown format: {format}. Use 'markdown' or 'json'.")


@app.get("/api/sessions")
async def api_list_sessions():
    return {"sessions": agent.list_sessions()}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    agent.clear_conversation(session_id)
    return {"success": True}


# ──────────────────────── System Monitor ────────────────────────

@app.get("/api/monitor")
async def api_monitor():
    """Get system stats snapshot (CPU, RAM, disk, GPU)."""
    return get_system_snapshot()


@app.get("/api/monitor/processes")
async def api_processes():
    """Get top processes by resource usage."""
    return {"processes": get_process_info()}


@app.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket):
    """WebSocket for real-time system monitoring (pushes every 2s)."""
    await ws.accept()
    try:
        while True:
            snapshot = get_system_snapshot()
            await ws.send_json(snapshot)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Monitor WS error: %s", e)


# ──────────────────────── RAG / Documents ────────────────────────

@app.get("/api/documents")
async def api_list_documents():
    """List all indexed documents."""
    return {"documents": rag_engine.list_documents()}


@app.post("/api/documents/upload")
async def api_upload_document(file: UploadFile = File(...)):
    """Upload and index a document for RAG."""
    content = await file.read()

    # Try to decode text
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError:
            raise HTTPException(400, "Could not decode file. Only text files are supported.")

    if not text.strip():
        raise HTTPException(400, "File is empty")

    # Save upload
    upload_path = UPLOADS_DIR / file.filename
    upload_path.write_bytes(content)

    # Index in RAG
    doc = rag_engine.add_document(file.filename, text, {"size": len(content)})

    return {
        "success": True,
        "document": doc.to_dict(),
    }


@app.post("/api/documents/index-text")
async def api_index_text(name: str = Form(...), content: str = Form(...)):
    """Index raw text content for RAG."""
    doc = rag_engine.add_document(name, content)
    return {"success": True, "document": doc.to_dict()}


@app.delete("/api/documents/{doc_id}")
async def api_delete_document(doc_id: str):
    """Remove a document from the RAG index."""
    if not rag_engine.remove_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"success": True}


@app.get("/api/documents/search")
async def api_search_documents(q: str, limit: int = 5):
    """Search indexed documents."""
    results = rag_engine.search(q, top_k=limit)
    return {"results": results}


# ──────────────────────── Prompt Templates ────────────────────────

@app.get("/api/templates")
async def api_list_templates():
    """List all prompt templates."""
    return {"templates": get_all_templates()}


@app.get("/api/templates/{template_id}")
async def api_get_template(template_id: str):
    """Get a specific template."""
    t = get_template(template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    return t


class TemplateCreateRequest(BaseModel):
    name: str
    prompt: str
    icon: str = "&#9889;"
    category: str = "custom"
    variables: list[str] = []


@app.post("/api/templates")
async def api_create_template(req: TemplateCreateRequest):
    """Create a custom template."""
    t = add_custom_template(req.model_dump())
    return {"success": True, "template": t}


@app.delete("/api/templates/{template_id}")
async def api_delete_template(template_id: str):
    """Delete a custom template."""
    if not delete_custom_template(template_id):
        raise HTTPException(404, "Template not found or is a built-in template")
    return {"success": True}


# ──────────────────────── File Upload (Chat Attachments) ────────────────────────

@app.post("/api/upload")
async def api_upload_file(file: UploadFile = File(...)):
    """Upload a file for use in chat (stored in uploads dir)."""
    content = await file.read()

    # Limit file size to 50MB
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    upload_path = UPLOADS_DIR / file.filename
    upload_path.write_bytes(content)

    return {
        "success": True,
        "filename": file.filename,
        "path": str(upload_path),
        "size": len(content),
    }


@app.get("/api/uploads")
async def api_list_uploads():
    """List uploaded files."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in UPLOADS_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "path": str(f),
            })
    files.sort(key=lambda x: x["filename"])
    return {"files": files}


# ──────────────────────── Multi-Agent ────────────────────────

@app.get("/api/agents/roles")
async def api_list_roles():
    """List all available agent roles."""
    return {"roles": list_roles_dict()}


class RoleConfigRequest(BaseModel):
    role_id: str
    provider: str | None = None
    model: str | None = None


@app.post("/api/agents/roles/config")
async def api_update_role_config(req: RoleConfigRequest):
    """Update model/provider configuration for an agent role."""
    role = get_role(req.role_id)
    if not role:
        raise HTTPException(404, f"Unknown role: {req.role_id}")

    update_role_config(req.role_id, req.provider, req.model)

    # Persist to config
    config = load_config()
    config.setdefault("agent_roles", {})[req.role_id] = {
        "preferred_provider": req.provider,
        "preferred_model": req.model,
    }
    save_config(config)

    return {"success": True, "role": role.to_dict()}


class MultiAgentRequest(BaseModel):
    task: str
    session_id: str | None = None


@app.post("/api/agents/run")
async def api_run_multi_agent(req: MultiAgentRequest):
    """Run a multi-agent workflow (SSE stream for real-time events)."""
    session_id = req.session_id or str(uuid.uuid4())

    async def event_stream():
        async for event in orchestrator.run_workflow(req.task, session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/agents/workflows")
async def api_list_workflows():
    """List all active/completed workflows."""
    return {"workflows": orchestrator.list_workflows()}


@app.get("/api/agents/workflows/{workflow_id}")
async def api_get_workflow(workflow_id: str):
    """Get workflow details including subtasks, messages, artifacts."""
    ctx = orchestrator.get_workflow(workflow_id)
    if not ctx:
        raise HTTPException(404, "Workflow not found")
    return ctx.to_dict()


@app.websocket("/ws/agents")
async def ws_multi_agent(ws: WebSocket):
    """WebSocket endpoint for real-time multi-agent workflow execution."""
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "run":
                task = data.get("task", "")
                session_id = data.get("session_id", str(uuid.uuid4()))

                if not task.strip():
                    await ws.send_json({"type": "error", "message": "Puste zadanie"})
                    continue

                async for event in orchestrator.run_workflow(task, session_id):
                    await ws.send_json(event)

            elif msg_type == "list_workflows":
                await ws.send_json({
                    "type": "workflows",
                    "workflows": orchestrator.list_workflows(),
                })

    except WebSocketDisconnect:
        logger.info("Multi-agent WS disconnected")
    except Exception as e:
        logger.error("Multi-agent WS error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
