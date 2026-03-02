# ─────────────────────────────────────────────────────────────
# Dockerfile — Multi-Document RAG System
#
# Build:  docker build -t multi-rag .
# Run:    docker run -it -v $(pwd)/data:/app/data multi-rag chat
# ─────────────────────────────────────────────────────────────

# Use slim Python 3.11 — smaller image, same compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching optimization)
# Reinstalling deps only when pyproject.toml changes, not on every code change
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Copy application source code
COPY src/ src/

# Create directories for mounted volumes
RUN mkdir -p data chroma_db

# Non-root user for security
RUN useradd --create-home --no-log-init appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command — runs interactive chat
ENTRYPOINT ["python", "-m", "rag.cli"]
CMD ["chat"]
