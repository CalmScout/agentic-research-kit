# -----------------------------------------------------------------------------
# Multi-Stage Dockerfile for Agentic Research Kit
# -----------------------------------------------------------------------------

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --no-install-project

# -----------------------------------------------------------------------------
# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/rag_storage /app/logs /app/data /root/.cache/huggingface && \
    chown -R appuser:appuser /app /root/.cache

# Set working directory
WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml ./

# Install application
COPY src ./src
COPY main.py ./

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Ensure Python uses the virtual environment
ENV PYTHONPATH="/app:/app/src:/app/.venv/lib/python3.11/site-packages"
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
