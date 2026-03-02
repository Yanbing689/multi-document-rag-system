"""
loader.py — PDF loading and intelligent text splitting.

Production improvements over basic implementation:
- Recursive glob for nested folders
- Rich metadata on every chunk (source, page, chunk_index)
- Configurable separators for better split quality
- Graceful per-file error handling (one bad PDF won't crash the pipeline)
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_settings
from rag.exceptions import DataNotFoundError
from rag.logging import get_logger

logger = get_logger(__name__)


def load_documents(data_dir: Path | None = None) -> List[Document]:
    """
    Load all PDF files from the data directory.

    Args:
        data_dir: Override for the configured data directory.

    Returns:
        List of Document objects (one per PDF page).

    Raises:
        DataNotFoundError: If the directory or no PDFs are found.
    """
    settings = get_settings()
    folder = Path(data_dir or settings.data_dir)

    if not folder.exists():
        raise DataNotFoundError(
            f"Data directory not found: {folder.resolve()}\n"
            f"  Fix: mkdir -p {folder} && cp your-docs/*.pdf {folder}/"
        )

    # Support nested folders with recursive glob
    pdf_files = sorted(folder.rglob("*.pdf"))

    if not pdf_files:
        raise DataNotFoundError(
            f"No PDF files found in: {folder.resolve()}\n"
            f"  Fix: Add PDF files to the data/ directory and retry."
        )

    logger.info("loading_documents", folder=str(folder), pdf_count=len(pdf_files))

    documents: List[Document] = []
    failed: List[str] = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Enrich metadata with absolute source path
            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["file_path"] = str(pdf_path.resolve())

            documents.extend(pages)
            logger.info("pdf_loaded", file=pdf_path.name, pages=len(pages))

        except Exception as e:
            logger.warning("pdf_load_failed", file=pdf_path.name, error=str(e))
            failed.append(pdf_path.name)

    if not documents:
        raise DataNotFoundError("All PDF files failed to load. Check the logs above.")

    if failed:
        logger.warning("some_pdfs_failed", failed=failed, loaded=len(documents))

    logger.info(
        "documents_loaded",
        total_pages=len(documents),
        pdf_files=len(pdf_files),
        failed=len(failed),
    )
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for optimal retrieval.

    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraph breaks → newlines → sentences → words, in that order,
    preserving semantic coherence as much as possible.

    The chunk_overlap creates a sliding window so sentences at
    chunk boundaries appear in both neighboring chunks — preventing
    context loss during retrieval.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Ordered from most to least preferred split point
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "documents_split",
        input_pages=len(documents),
        output_chunks=len(chunks),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return chunks
