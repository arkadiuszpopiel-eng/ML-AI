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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import load_config, save_config, get_models_dir, BASE_DIR
from .inference.engine import engine
from .inference.model_manager import (
    list_models, get_model_path, download_model, get_recommended_models
)
from .agent.loop import agent
from .storage import (
    save_conversation, load_conversation, list_conversations,
    delete_conversation, update_conversation_title,
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


app = FastAPI(title="NeuroForge", version="0.2.0", lifespan=lifespan)

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
    """Load a model into the inference engine."""
    model_path = get_model_path(req.filename)
    if not model_path:
        raise HTTPException(404, f"Model not found: {req.filename}")

    success = await engine.start(
        model_path,
        gpu_layers=req.gpu_layers,
        context_size=req.context_size,
        threads=req.threads,
    )
    if not success:
        raise HTTPException(500, "Failed to start inference engine. Check logs.")

    return {"success": True, "model": req.filename}


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
    """Download a model from HuggingFace."""
    try:
        path = await download_model(req.repo, req.filename)
        return {"success": True, "path": path}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


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

                async for event in agent.process_message(session_id, user_msg):
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
