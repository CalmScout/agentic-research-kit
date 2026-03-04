# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Research Kit (ARK) is a multimodal agentic RAG system built with LangGraph. It implements a 3-agent sequential workflow for deep research and information synthesis with hybrid retrieval (Vector + BM25 + Knowledge Graph via LightRAG).

**Core Architecture:**
- **Agent 1 (Enhanced Retriever)**: Extracts entities, generates embeddings, performs hybrid retrieval, augments with web search
- **Agent 2 (Enhanced Response Generator)**: Reranks candidates, synthesizes evidence, generates response with citations
- **Agent 3 (Verification Node)**: Fact-checks response against sources, removes hallucinations (critique LLM)

The system uses:
- **LangGraph** for orchestration with ReAct loops (max 3 iterations for refinement)
- **LightRAG** for hybrid Vector + Knowledge Graph traversal
- **LanceDB** for high-performance storage (columnar KV + vectors)
- **Qwen3-VL-2B** (local) for embeddings/vision + **DeepSeek** (API) for reasoning
- **Arize Phoenix** for distributed tracing and observability
- **Loguru** for structured application logging

## Development Commands

**CRITICAL: Always use `uv` instead of `pip` for all Python operations.**

```bash
# Dependency management
uv sync                    # Install all dependencies
uv sync --dev              # Install with dev dependencies

# Code quality
make lint                  # Ruff linter
make format                # Black formatting
make format-check          # Check formatting
make type-check            # Mypy type checking

# Testing
make test                  # Run pytest
make test-cov              # Run with coverage report
uv run pytest tests/       # Run specific tests
uv run pytest tests/test_utils/test_logging.py  # Single file

# Docker
make docker-build          # Build Docker image
make docker-up             # Start services
make docker-down           # Stop services

# Application
uv run ark serve           # Start FastAPI server
uv run ark query "Your query here" --mode hybrid
uv run ark ingest-dir ./data --pattern "*.pdf"
```

## Configuration Management

ARK uses a two-tier configuration system:

1. **`.env.defaults`** - Shareable defaults (committed to git)
2. **`.env`** - Personal overrides (gitignored, contains secrets)

Priority: `.env` > `.env.defaults` > code defaults

**Required in `.env`:**
- `DEEPSEEK_API_KEY` - Get from https://platform.deepseek.com/

**Key configuration options:**
- `PHOENIX_ENABLED=true` - Enable distributed tracing
- `LOG_LEVEL=DEBUG` - Set logging severity (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT=text` - Set log format (text, json)
- `USE_GPU=true` - Enable CUDA for local models
- `DEEPSEEK_MODE=api` - Use DeepSeek API (falls back to local if configured)
- `RAG_WORKING_DIR=./rag_storage` - LightRAG storage location

## Logging & Observability

### 1. Logging (Loguru)
ARK uses a centralized logging utility in `src/utils/logger.py`.
- **Convention**: ALWAYS use `from src.utils.logger import logger`.
- **Automatic Setup**: `setup_logging()` is called automatically on workflow import.
- **Output**: Console (JSON/Text) + Files (`logs/ark_YYYY-MM-DD.log`).

### 2. Observability (Phoenix)
Distributed tracing is centralized in `src/utils/observability.py`.
- **Automatic Setup**: `setup_observability()` is called automatically on workflow import.
- **UI**: View traces at `http://localhost:6006`.

## Key Architecture Patterns

### 1. LangGraph Workflow
The workflow is defined in [src/agents/workflow.py](src/agents/workflow.py):
- State flows through agents via `BaseAgentState` TypedDict
- ReAct loop allows verification to trigger refinement (max 3 iterations)
- Observability and Logging are initialized at the module level.

### 2. Agent State Management
[BaseAgentState](src/agents/base_state.py) contains all fields that flow between agents.

### 3. Tool Registry
Dynamic tool loading via [ToolRegistry](src/agents/tools/registry.py).

### 4. Memory Store
[MemoryStore](src/agents/memory/store.py) provides persistent research context.
- **Initialization**: Requires workspace Path: `MemoryStore(Path("./workspace"))`.

## Storage Architecture
**LanceDB** (columnar storage) is used for:
1. **LightRAG Knowledge Graph** - `./rag_storage/lancedb_store/`
2. **Research Memory** - `./workspace/memory/lancedb`

## Important Implementation Notes
1. **Thread Isolation**: LightRAG uses `isolated_lightrag.py` to prevent async conflicts.
2. **GPU Memory**: 8-bit quantization enabled for 12GB VRAM compatibility.
3. **Async Patterns**: All I/O operations are async.
4. **Error Handling**: Tenacity retry logic for API calls.
5. **Citation Integrity**: Verification node enforces 100% source attribution.
