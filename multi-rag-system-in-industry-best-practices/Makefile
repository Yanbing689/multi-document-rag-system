# ─────────────────────────────────────────────────────────────
# Makefile — Common commands for the Multi-Document RAG System
#
# Usage:
#   make setup       Install all dependencies
#   make chat        Start interactive Q&A
#   make build       Build vector store from PDFs
#   make test        Run tests
#   make lint        Lint and format code
#   make clean       Remove generated files
# ─────────────────────────────────────────────────────────────

.PHONY: setup install chat query build info test lint format clean docker-build docker-run help

PYTHON := python3.11
VENV := rag_env
BIN := $(VENV)/bin

# ── Setup ─────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -e ".[dev]"
	cp -n .env.example .env || true
	mkdir -p data
	@echo "✅ Setup complete. Activate with: source $(VENV)/bin/activate"

install:
	pip install -e ".[dev]"

# ── Run ───────────────────────────────────────────────────────
chat:
	python -m rag.cli chat

build:
	python -m rag.cli build

build-rebuild:
	python -m rag.cli build --rebuild

info:
	python -m rag.cli info

query:
	@read -p "Question: " q; python -m rag.cli query-cmd "$$q"

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest

test-cov:
	pytest --cov=rag --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t multi-rag:latest .

docker-run:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/chroma_db:/app/chroma_db \
		-v $(PWD)/.env:/app/.env \
		multi-rag:latest chat

# ── Cleanup ───────────────────────────────────────────────────
clean:
	rm -rf chroma_db/ __pycache__/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf $(VENV)/

help:
	@echo "Available commands:"
	@echo "  make setup        Install dependencies + create venv"
	@echo "  make chat         Start interactive Q&A session"
	@echo "  make build        Build vector store from PDFs"
	@echo "  make build-rebuild Force rebuild vector store"
	@echo "  make info         Show config and system status"
	@echo "  make test         Run test suite"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Auto-format code with ruff"
	@echo "  make clean        Remove generated files"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run in Docker container"
