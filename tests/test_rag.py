"""Tests for RAG engine."""
import pytest

from neurostudio.backend.rag.engine import RAGEngine, _tokenize, _compute_tf, Document


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World! Python programming.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "programming" in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("I am a developer")
        assert "i" not in tokens  # too short (< 2 chars)
        assert "am" in tokens
        assert "developer" in tokens

    def test_handles_polish_characters(self):
        tokens = _tokenize("Zażółć gęślą jaźń")
        assert "gęślą" in tokens
        assert "jaźń" in tokens

    def test_empty_input(self):
        assert _tokenize("") == []


class TestTF:
    def test_compute_tf(self):
        tokens = ["python", "python", "java", "rust"]
        tf = _compute_tf(tokens)
        assert tf["python"] == pytest.approx(0.5)
        assert tf["java"] == pytest.approx(0.25)
        assert tf["rust"] == pytest.approx(0.25)

    def test_empty_tokens(self):
        assert _compute_tf([]) == {}


class TestRAGEngine:
    def test_add_document(self, tmp_path):
        engine = RAGEngine(chunk_size=10, chunk_overlap=2)
        doc = engine.add_document("test.txt", "To jest testowy dokument o programowaniu w Pythonie")

        assert isinstance(doc, Document)
        assert doc.filename == "test.txt"
        assert len(doc.chunks) >= 1

    def test_search(self, tmp_path):
        engine = RAGEngine(chunk_size=50, chunk_overlap=10)
        engine.add_document("python.txt", "Python jest jezykiem programowania. Python jest popularny.")
        engine.add_document("rust.txt", "Rust jest szybkim jezykiem systemowym.")

        results = engine.search("Python programowanie")
        assert len(results) > 0
        assert results[0]["filename"] == "python.txt"

    def test_search_empty_index(self, tmp_path):
        engine = RAGEngine()
        results = engine.search("anything")
        assert results == []

    def test_remove_document(self, tmp_path):
        engine = RAGEngine(chunk_size=50, chunk_overlap=10)
        doc = engine.add_document("remove_me.txt", "This document will be removed")
        doc_id = doc.doc_id

        assert engine.remove_document(doc_id) is True
        assert engine.remove_document(doc_id) is False  # already removed
        assert len(engine.search("removed")) == 0

    def test_list_documents(self, tmp_path):
        engine = RAGEngine()
        engine.add_document("doc1.txt", "Content one")
        engine.add_document("doc2.txt", "Content two")

        docs = engine.list_documents()
        assert len(docs) == 2
        filenames = {d["filename"] for d in docs}
        assert "doc1.txt" in filenames
        assert "doc2.txt" in filenames

    def test_get_context_for_query(self, tmp_path):
        engine = RAGEngine(chunk_size=50, chunk_overlap=10)
        engine.add_document("guide.txt", "Python to interpretowany jezyk programowania wysokiego poziomu")

        context = engine.get_context_for_query("Python programowanie")
        assert "Python" in context or "python" in context.lower()

    def test_get_context_empty(self, tmp_path):
        engine = RAGEngine()
        assert engine.get_context_for_query("anything") == ""

    def test_chunk_text(self, tmp_path):
        engine = RAGEngine(chunk_size=5, chunk_overlap=2)
        chunks = engine._chunk_text("a b c d e f g h i j k")
        assert len(chunks) >= 2
        # Each chunk should have at most 5 words
        for chunk in chunks:
            assert len(chunk.split()) <= 5
