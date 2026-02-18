"""
Conversation persistence - save/load chat histories to disk.
"""
import json
import time
from pathlib import Path
from typing import Optional

from .config import BASE_DIR

HISTORY_DIR = BASE_DIR / "data" / "conversations"


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_conversation(session_id: str, messages: list[dict], title: Optional[str] = None) -> dict:
    """Save a conversation to disk. Returns metadata."""
    _ensure_dir()

    # Generate title from first user message if not provided
    if not title:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                title = content[:80].strip() or "Nowa rozmowa"
                break
        else:
            title = "Nowa rozmowa"

    meta = {
        "session_id": session_id,
        "title": title,
        "created_at": time.time(),
        "updated_at": time.time(),
        "message_count": len([m for m in messages if m.get("role") in ("user", "assistant")]),
    }

    data = {"meta": meta, "messages": messages}

    path = HISTORY_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return meta


def load_conversation(session_id: str) -> Optional[dict]:
    """Load a conversation from disk."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_conversations() -> list[dict]:
    """List all saved conversations (newest first)."""
    _ensure_dir()
    convos = []
    for path in HISTORY_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                convos.append(data.get("meta", {}))
        except (json.JSONDecodeError, KeyError):
            continue

    convos.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
    return convos


def delete_conversation(session_id: str) -> bool:
    """Delete a saved conversation."""
    path = HISTORY_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def update_conversation_title(session_id: str, title: str) -> bool:
    """Update the title of a saved conversation."""
    data = load_conversation(session_id)
    if not data:
        return False
    data["meta"]["title"] = title
    data["meta"]["updated_at"] = time.time()
    path = HISTORY_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return True
