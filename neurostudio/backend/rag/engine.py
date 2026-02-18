"""
RAG Engine - lightweight document retrieval using TF-IDF.
No external vector DB needed - everything runs locally in memory.
"""
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ..config import BASE_DIR

DOCS_DIR = BASE_DIR / "data" / "documents"
INDEX_PATH = BASE_DIR / "data" / "rag_index.json"


def _ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanum, remove short tokens."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9ąćęłńóśźż]{2,}', text)
    return tokens


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency."""
    counter = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counter.items()}


class Document:
    """A single indexed document."""

    def __init__(self, doc_id: str, filename: str, content: str,
                 chunks: Optional[list[str]] = None, metadata: Optional[dict] = None):
        self.doc_id = doc_id
        self.filename = filename
        self.content = content
        self.chunks = chunks or []
        self.metadata = metadata or {}
        self.indexed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "content_length": len(self.content),
            "chunk_count": len(self.chunks),
            "metadata": self.metadata,
            "indexed_at": self.indexed_at,
        }


class RAGEngine:
    """TF-IDF based document retrieval engine."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: dict[str, Document] = {}
        self.chunk_index: list[dict] = []  # [{doc_id, chunk_idx, tokens, tf}]
        self.idf: dict[str, float] = {}
        self._load_index()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break

        if not chunks and text.strip():
            chunks = [text]

        return chunks

    def _rebuild_idf(self):
        """Rebuild inverse document frequency index."""
        doc_count = len(self.chunk_index)
        if doc_count == 0:
            self.idf = {}
            return

        df = Counter()
        for entry in self.chunk_index:
            unique_terms = set(entry["tokens"])
            for term in unique_terms:
                df[term] += 1

        self.idf = {
            term: math.log((doc_count + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

    def add_document(self, filename: str, content: str, metadata: Optional[dict] = None) -> Document:
        """Index a new document."""
        _ensure_dirs()

        doc_id = hashlib.md5(f"{filename}:{content[:200]}".encode()).hexdigest()[:12]
        chunks = self._chunk_text(content)

        doc = Document(doc_id, filename, content, chunks, metadata)
        self.documents[doc_id] = doc

        # Index chunks
        for i, chunk in enumerate(chunks):
            tokens = _tokenize(chunk)
            tf = _compute_tf(tokens)
            self.chunk_index.append({
                "doc_id": doc_id,
                "chunk_idx": i,
                "tokens": tokens,
                "tf": tf,
                "text": chunk,
            })

        self._rebuild_idf()

        # Save content to disk
        doc_path = DOCS_DIR / f"{doc_id}.txt"
        doc_path.write_text(content, encoding="utf-8")

        self._save_index()
        return doc

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self.chunk_index = [e for e in self.chunk_index if e["doc_id"] != doc_id]
        self._rebuild_idf()

        doc_path = DOCS_DIR / f"{doc_id}.txt"
        if doc_path.exists():
            doc_path.unlink()

        self._save_index()
        return True

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant chunks using TF-IDF similarity."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = _compute_tf(query_tokens)

        # Compute TF-IDF scores for query
        query_vec = {}
        for term, tf in query_tf.items():
            idf = self.idf.get(term, 0)
            query_vec[term] = tf * idf

        # Score each chunk
        results = []
        for entry in self.chunk_index:
            score = 0.0
            chunk_tf = entry["tf"]
            for term, q_weight in query_vec.items():
                if term in chunk_tf:
                    c_weight = chunk_tf[term] * self.idf.get(term, 0)
                    score += q_weight * c_weight

            if score > 0:
                doc = self.documents.get(entry["doc_id"])
                results.append({
                    "score": round(score, 4),
                    "text": entry["text"],
                    "doc_id": entry["doc_id"],
                    "chunk_idx": entry["chunk_idx"],
                    "filename": doc.filename if doc else "unknown",
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get_context_for_query(self, query: str, max_chars: int = 3000) -> str:
        """Get relevant context string for a query (used by agent)."""
        results = self.search(query, top_k=10)
        if not results:
            return ""

        context_parts = []
        total = 0
        for r in results:
            text = r["text"]
            if total + len(text) > max_chars:
                text = text[:max_chars - total]
            context_parts.append(f"[{r['filename']}]:\n{text}")
            total += len(text)
            if total >= max_chars:
                break

        return "\n\n---\n\n".join(context_parts)

    def list_documents(self) -> list[dict]:
        """List all indexed documents."""
        return [doc.to_dict() for doc in self.documents.values()]

    def _save_index(self):
        """Save index metadata to disk."""
        _ensure_dirs()
        data = {
            "documents": {
                doc_id: {
                    "filename": doc.filename,
                    "content_length": len(doc.content),
                    "chunk_count": len(doc.chunks),
                    "metadata": doc.metadata,
                    "indexed_at": doc.indexed_at,
                }
                for doc_id, doc in self.documents.items()
            }
        }
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        """Load index from disk on startup."""
        _ensure_dirs()
        if not INDEX_PATH.exists():
            return

        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            for doc_id, meta in data.get("documents", {}).items():
                doc_path = DOCS_DIR / f"{doc_id}.txt"
                if not doc_path.exists():
                    continue

                content = doc_path.read_text(encoding="utf-8")
                chunks = self._chunk_text(content)

                doc = Document(doc_id, meta["filename"], content, chunks, meta.get("metadata"))
                doc.indexed_at = meta.get("indexed_at", 0)
                self.documents[doc_id] = doc

                for i, chunk in enumerate(chunks):
                    tokens = _tokenize(chunk)
                    tf = _compute_tf(tokens)
                    self.chunk_index.append({
                        "doc_id": doc_id,
                        "chunk_idx": i,
                        "tokens": tokens,
                        "tf": tf,
                        "text": chunk,
                    })

            self._rebuild_idf()
        except (json.JSONDecodeError, KeyError):
            pass


# Singleton
rag_engine = RAGEngine()
