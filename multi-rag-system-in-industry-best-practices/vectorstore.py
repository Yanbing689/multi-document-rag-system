"""
vectorstore.py — ChromaDB vector store management.

Production improvements:
- Singleton embedding function (loaded once, reused everywhere)
- MMR (Maximal Marginal Relevance) retrieval — reduces redundancy in results
- Health check to verify store is usable before querying
- Clear separation between create vs load paths
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import get_settings
from rag.exceptions import EmbeddingError, VectorStoreError
from rag.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Load and cache the embedding model (singleton).

    lru_cache(maxsize=1) ensures the model is loaded exactly once
    per process — loading it repeatedly would waste ~2-3s per call.
    """
    settings = get_settings()
    logger.info("loading_embeddings", model=settings.embedding_model)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Cosine similarity
        )
        logger.info("embeddings_ready", model=settings.embedding_model)
        return embeddings
    except Exception as e:
        raise EmbeddingError(f"Failed to load embedding model: {e}") from e


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Create a new ChromaDB vector store from document chunks.
    Persists to disk automatically.
    """
    settings = get_settings()

    if not chunks:
        raise VectorStoreError("Cannot create vector store from empty chunk list.")

    logger.info("creating_vector_store", chunks=len(chunks), path=str(settings.chroma_db_path))

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_function(),
            persist_directory=str(settings.chroma_db_path),
            collection_name=settings.collection_name,
        )
    except Exception as e:
        raise VectorStoreError(f"Failed to create vector store: {e}") from e

    count = vector_store._collection.count()
    logger.info("vector_store_created", documents=count, path=str(settings.chroma_db_path))
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    """
    settings = get_settings()

    logger.info("loading_vector_store", path=str(settings.chroma_db_path))
    try:
        vector_store = Chroma(
            persist_directory=str(settings.chroma_db_path),
            embedding_function=get_embedding_function(),
            collection_name=settings.collection_name,
        )
    except Exception as e:
        raise VectorStoreError(f"Failed to load vector store: {e}") from e

    # Health check — ensure store actually has documents
    count = vector_store._collection.count()
    if count == 0:
        raise VectorStoreError(
            "Vector store exists but is empty. Run with --rebuild to repopulate."
        )

    logger.info("vector_store_loaded", documents=count)
    return vector_store


def vector_store_exists() -> bool:
    """Return True if a persisted vector store exists on disk."""
    settings = get_settings()
    db_path = Path(settings.chroma_db_path)
    return db_path.exists() and any(db_path.iterdir())
