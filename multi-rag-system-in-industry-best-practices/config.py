"""
config.py — Type-safe, validated configuration using Pydantic Settings v2.

All settings are read from environment variables or .env file.
Pydantic validates types and raises clear errors on misconfiguration.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings — validated at startup.
    All values can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"), description="Folder containing PDFs")
    chroma_db_path: Path = Field(default=Path("chroma_db"), description="Vector DB path")

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model",
    )

    # ── LLM ─────────────────────────────────────────────────
    llm_model: str = Field(default="llama3.2", description="Ollama model name")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=120, description="LLM request timeout in seconds")

    # ── Vector Store ─────────────────────────────────────────
    collection_name: str = Field(default="rag_docs")
    retriever_k: int = Field(default=3, ge=1, le=20, description="Chunks retrieved per query")
    retriever_fetch_k: int = Field(default=20, description="Candidates before MMR reranking")
    retriever_search_type: str = Field(default="mmr", description="similarity | mmr")

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    # ── App ──────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    app_name: str = Field(default="Multi-Document RAG System")
    app_version: str = Field(default="1.0.0")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("retriever_search_type")
    @classmethod
    def valid_search_type(cls, v: str) -> str:
        allowed = {"similarity", "mmr"}
        if v not in allowed:
            raise ValueError(f"retriever_search_type must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    lru_cache ensures settings are loaded once and reused — 
    avoids re-reading .env on every function call.
    """
    return Settings()
