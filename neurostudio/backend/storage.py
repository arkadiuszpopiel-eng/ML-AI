"""
Conversation persistence - save/load chat histories to disk.
Uses atomic writes (write to temp file + rename) to prevent data corruption.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .config import BASE_DIR

HISTORY_DIR = BASE_DIR / "data" / "conversations"


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: dict):
    """Write JSON atomically: write to temp file then rename to prevent corruption."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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

    # Preserve original created_at if conversation already exists
    path = HISTORY_DIR / f"{session_id}.json"
    created_at = time.time()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("meta", {}).get("created_at", created_at)
        except (json.JSONDecodeError, KeyError):
            pass

    meta = {
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": time.time(),
        "message_count": len([m for m in messages if m.get("role") in ("user", "assistant")]),
    }

    data = {"meta": meta, "messages": messages}
    _atomic_write_json(path, data)

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
    _atomic_write_json(path, data)
    return True


def export_conversation_markdown(session_id: str) -> Optional[str]:
    """Export a conversation as Markdown text.

    Format:
        # Title
        _Date_

        ---

        **User:**
        message

        **AI:**
        response

        > Tool: tool_name(args)
        > Result: ...
    """
    data = load_conversation(session_id)
    if not data:
        return None

    meta = data.get("meta", {})
    messages = data.get("messages", [])
    title = meta.get("title", "Konwersacja")

    from datetime import datetime
    created = meta.get("created_at")
    date_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M") if created else ""

    lines = [f"# {title}", ""]
    if date_str:
        lines.append(f"*{date_str}*")
        lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue
        elif role == "user":
            lines.append(f"**User:**")
            lines.append(content)
            lines.append("")
        elif role == "assistant":
            # Check for tool calls
            tool_calls = msg.get("tool_calls", [])
            if content:
                lines.append(f"**AI:**")
                lines.append(content)
                lines.append("")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    lines.append(f"> Tool: `{name}({args})`")
                lines.append("")
        elif role == "tool":
            # Tool result - show condensed
            tool_content = content[:200]
            if len(content) > 200:
                tool_content += "..."
            lines.append(f"> Result: `{tool_content}`")
            lines.append("")

    return "\n".join(lines)
