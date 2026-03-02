"""
exceptions.py — Domain-specific exceptions.

Using custom exceptions instead of bare Exception gives:
- Clear error messages for operators
- Easy catch-by-type in callers
- Better logging context
"""


class RAGError(Exception):
    """Base exception for all RAG system errors."""


class DataNotFoundError(RAGError):
    """Raised when the data directory or PDFs don't exist."""


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""


class LLMConnectionError(RAGError):
    """Raised when Ollama is unreachable or the model isn't available."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class ConfigurationError(RAGError):
    """Raised when the application is misconfigured."""
