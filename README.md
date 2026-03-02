# Multi-Document RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for querying local PDF documents using a fully local LLM stack — no API keys, no cloud, no data leaving your machine.

**Stack:** LangChain · ChromaDB · HuggingFace Embeddings · Ollama (Llama3) · Typer · Rich · Pydantic Settings · Structlog

---

## Features

- **Local-first** — runs 100% offline after initial model download
- **MMR retrieval** — Maximal Marginal Relevance reduces redundant context
- **Source citations** — every answer cites document name + page number
- **Type-safe config** — Pydantic Settings v2 validates all settings at startup
- **Structured logging** — human-readable in terminal, JSON in production/CI
- **Rich CLI** — color-coded output, progress indicators, formatted tables
- **Docker support** — containerized for consistent deployment
- **Test suite** — pytest coverage for core pipeline components

---

## Project Structure

```
multi-document-rag-system/
├── src/rag/
│   ├── __init__.py        # Package definition
│   ├── cli.py             # Typer CLI — chat, query, build, info
│   ├── config.py          # Pydantic Settings v2 — type-safe config
│   ├── exceptions.py      # Domain-specific exceptions
│   ├── loader.py          # PDF loading + text splitting
│   ├── logging.py         # Structlog setup (JSON / Rich)
│   ├── rag_chain.py       # LCEL RAG chain + QueryResult
│   └── vectorstore.py     # ChromaDB create/load + embedding singleton
├── tests/
│   └── test_pipeline.py   # Pytest unit tests
├── data/                  # Drop your PDFs here (gitignored)
├── chroma_db/             # Auto-generated vector store (gitignored)
├── pyproject.toml         # Modern Python packaging + tool config
├── Makefile               # Developer shortcuts
├── Dockerfile             # Container build
├── .env.example           # Config template → copy to .env
├── .gitignore
└── .pre-commit-config.yaml
```

---

## Quick Start

### Prerequisites
- Python 3.11
- [Ollama](https://ollama.com) — download and install

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/multi-document-rag-system.git
cd multi-document-rag-system

# Create virtualenv and install everything
make setup
source rag_env/bin/activate
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env if you want to change models or settings
```

### 3. Add your PDFs

```bash
cp /path/to/your/documents/*.pdf data/
```

### 4. Start Ollama

```bash
# Terminal tab 1
ollama serve
ollama pull llama3.2   # first time only (~2GB)
```

### 5. Run

```bash
# Interactive chat
make chat
# or
python -m rag.cli chat

# Single query
python -m rag.cli query-cmd "What are the top tech jobs in 2025?"

# Force rebuild after adding new PDFs
python -m rag.cli build --rebuild

# Check system status
python -m rag.cli info
```

---

## Configuration

All settings are in `.env` (copied from `.env.example`). Pydantic validates every value at startup:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama model (`ollama list` to see options) |
| `LLM_TEMPERATURE` | `0` | 0=focused, 1=creative |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RETRIEVER_K` | `3` | Chunks retrieved per query |
| `RETRIEVER_SEARCH_TYPE` | `mmr` | `mmr` (diverse) or `similarity` (fast) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Sliding window overlap |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Development

```bash
# Run tests
make test

# Run tests with coverage report
make test-cov

# Lint + format
make lint
make format

# Install pre-commit hooks (runs on every git commit)
pre-commit install
```

---

## Docker

```bash
# Build image
make docker-build

# Run container (mounts local data/ and chroma_db/)
make docker-run
```

---

## How It Works

```
PDFs → Load pages → Split into overlapping chunks → Embed → ChromaDB
                                                              ↓
User question → Embed → MMR retrieval (top-K diverse chunks) → LLM → Answer + Citations
```

1. **Load** — PDFs are loaded page by page with `PyPDFLoader`
2. **Split** — Pages split with `RecursiveCharacterTextSplitter` (200-char sliding window)
3. **Embed** — Each chunk converted to a 384-dim vector via `all-MiniLM-L6-v2`
4. **Store** — Vectors persisted to ChromaDB on disk
5. **Retrieve** — MMR fetches 20 candidates, returns 3 most diverse
6. **Generate** — Local Llama3 answers using only retrieved context
7. **Cite** — Answer includes source document and page number
