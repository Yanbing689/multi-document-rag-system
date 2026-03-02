"""
tests/test_pipeline.py — Unit tests for the RAG pipeline.

Run with: pytest
Run with coverage: pytest --cov=rag --cov-report=term-missing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.config import Settings, get_settings
from rag.exceptions import DataNotFoundError, VectorStoreError
from rag.loader import split_documents


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_documents():
    """Create minimal Document objects for testing."""
    return [
        Document(
            page_content="Software engineers are in high demand globally. " * 30,
            metadata={"source": "report.pdf", "page": 1},
        ),
        Document(
            page_content="AI and machine learning are the top skills employers seek. " * 30,
            metadata={"source": "report.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def settings_override(tmp_path):
    """Override settings to use temp directories."""
    return Settings(
        data_dir=tmp_path / "data",
        chroma_db_path=tmp_path / "chroma_db",
        chunk_size=500,
        chunk_overlap=50,
    )


# ── Config Tests ──────────────────────────────────────────────

class TestSettings:
    def test_default_settings_load(self):
        """Settings should load with sensible defaults."""
        s = Settings()
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.retriever_k == 3

    def test_chunk_overlap_validation(self):
        """chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            Settings(chunk_size=100, chunk_overlap=200)

    def test_invalid_search_type(self):
        """Invalid search type should raise validation error."""
        with pytest.raises(ValueError, match="retriever_search_type"):
            Settings(retriever_search_type="invalid")

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("LLM_MODEL", "mistral")
        monkeypatch.setenv("RETRIEVER_K", "5")
        s = Settings()
        assert s.llm_model == "mistral"
        assert s.retriever_k == 5


# ── Loader Tests ──────────────────────────────────────────────

class TestLoader:
    def test_split_documents_creates_chunks(self, sample_documents):
        """Splitting should produce more chunks than input pages."""
        chunks = split_documents(sample_documents)
        assert len(chunks) > len(sample_documents)

    def test_split_preserves_metadata(self, sample_documents):
        """Every chunk should carry source metadata from original doc."""
        chunks = split_documents(sample_documents)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "report.pdf"

    def test_split_adds_chunk_index(self, sample_documents):
        """split_documents should add chunk_index to metadata."""
        chunks = split_documents(sample_documents)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_load_documents_missing_dir(self, tmp_path):
        """Should raise DataNotFoundError for missing directory."""
        from rag.loader import load_documents
        with pytest.raises(DataNotFoundError, match="not found"):
            load_documents(tmp_path / "nonexistent")

    def test_load_documents_empty_dir(self, tmp_path):
        """Should raise DataNotFoundError when no PDFs found."""
        from rag.loader import load_documents
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(DataNotFoundError, match="No PDF"):
            load_documents(empty_dir)

    def test_chunk_overlap_creates_sliding_window(self):
        """Overlapping chunks should share content at boundaries."""
        docs = [Document(page_content="word " * 500, metadata={"source": "test.pdf", "page": 1})]
        chunks = split_documents(docs)

        if len(chunks) >= 2:
            # Last words of chunk 0 should appear in start of chunk 1
            end_of_chunk_0 = chunks[0].page_content[-50:]
            start_of_chunk_1 = chunks[1].page_content[:200]
            assert any(word in start_of_chunk_1 for word in end_of_chunk_0.split())


# ── Vector Store Tests ────────────────────────────────────────

class TestVectorStore:
    def test_vector_store_not_exists(self, tmp_path, monkeypatch):
        """vector_store_exists() should return False for empty path."""
        from rag import vectorstore
        monkeypatch.setattr(
            vectorstore, "get_settings",
            lambda: Settings(chroma_db_path=tmp_path / "nonexistent")
        )
        assert vectorstore.vector_store_exists() is False

    def test_empty_chunks_raises(self):
        """create_vector_store with empty list should raise VectorStoreError."""
        from rag.vectorstore import create_vector_store
        with pytest.raises(VectorStoreError, match="empty"):
            create_vector_store([])


# ── RAG Chain Tests ───────────────────────────────────────────

class TestRagChain:
    def test_query_result_format(self):
        """QueryResult.format() should contain answer and source info."""
        from rag.rag_chain import QueryResult, Source
        result = QueryResult(
            question="What is demand for engineers?",
            answer="Demand is very high globally.",
            sources=[Source(file="report.pdf", page=1, chunk_index=0, preview="Software engineers")],
        )
        formatted = result.format()
        assert "Demand is very high" in formatted
        assert "report.pdf" in formatted
        assert "Page 1" in formatted
