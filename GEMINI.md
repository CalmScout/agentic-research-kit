# Agentic Research Kit (ARK) - Context & Guidelines

Welcome to the **Agentic Research Kit (ARK)**, a multimodal agentic RAG system designed for deep research and information synthesis. This document provides essential context and instructions for AI agents (like Gemini) to effectively navigate and contribute to this project.

## 🚀 Project Overview

ARK utilizes a **3-agent LangGraph workflow** to orchestrate knowledge graph traversal, high-performance vector search, and real-time web augmentation.

### Core Architecture
The system follows a sequential 3-agent loop to minimize hallucination and maximize evidence grounding:

1.  **Enhanced Retriever (Agent 1):** Analyzes query modality, extracts entities, and executes hybrid retrieval (Vector + BM25 + LightRAG Knowledge Graph).
2.  **Enhanced Response Generator (Agent 2):** Reranks candidates, synthesizes evidence into structured summaries, and generates draft responses with source attribution.
3.  **Verification Node (Agent 3):** Fact-checks the response against retrieved sources, ensuring 100% citation integrity and removing hallucinations.

### Technology Stack
- **Orchestration:** LangGraph (State-aware sequential workflow)
- **RAG Engine:** LightRAG (Hybrid Vector + Knowledge Graph)
- **Storage:** LanceDB (High-performance columnar storage for KV and Vectors)
- **Models:** Qwen3-VL-2B (Cross-modal embeddings), Qwen2.5 (Local reasoning/extraction), DeepSeek/OpenAI (Primary reasoning)
- **Observability:** Arize Phoenix (OTLP tracing and span analysis)
- **Logging:** Loguru (Structured logging with file rotation and JSON/Text support)
- **Evaluation:** RAGAS (Faithfulness, Relevancy, and Context metrics)

## 🛠 Building, Running, and Testing

The project uses `uv` for dependency management and a `Makefile` for common tasks.

### Key Commands
- **Install Dependencies:** `make install` (runs `uv sync`)
- **Run Tests:** `make test` or `make test-cov` (includes coverage)
- **Linting & Formatting:** `make lint`, `make format`, `make type-check`
- **Full CI Suite:** `make ci` (runs all checks + docker-build)

### Operation
- **Ingestion:** `uv run ark ingest-dir ./data/research_papers --pattern "*.pdf"`
- **Research Query:** `uv run ark query "Your research question" --mode hybrid`
- **Start API Server:** `uv run ark serve` (FastAPI)
- **Start Telegram Gateway:** `uv run ark gateway`
- **Observability UI:** `PHOENIX_ENABLED=true uv run python -m phoenix.server.main serve` (available at http://localhost:6006)

## 📂 Directory Structure

- `src/agents/`: LangGraph workflow (`workflow.py`), agent logic, tools, and memory.
- `src/data_ingestion/`: Universal pipeline for PDF, DOCX, HTML, and CSV.
- `src/evaluation/`: RAGAS and retrieval metric suites.
- `src/api/`: FastAPI REST implementation.
- `src/utils/`: Centralized utilities for config, logging, and observability.
- `docs/`: Technical deep-dives (Architecture, LightRAG, Phoenix, Testing).
- `workspace/`: Long-term research memory (LanceDB).
- `rag_storage/`: Knowledge Graph storage for LightRAG.

## 📝 Development Conventions

- **Code Style:** Black for formatting, Ruff for linting.
- **Type Safety:** Mypy is used for type checking.
- **Testing:** Pytest is the primary testing framework. Aim for high test coverage (handled by `make test-cov`).
- **Logging:** 
    - Use `Loguru` for all logging. 
    - **Convention:** `from src.utils.logger import logger`. 
    - Avoid direct `from loguru import logger` or standard `import logging`.
- **Observability:** 
    - Centralized in `src/utils/observability.py`. 
    - Automatically initialized when `src.agents.workflow` is imported.
    - Ensure `PHOENIX_ENABLED=true` in `.env` for tracing.
- **State Management:** Agent state is managed via `BaseAgentState` in `src/agents/base_state.py`.
- **Async First:** Most agent and tool operations are asynchronous (`async`/`await`).

## 🗺 Roadmap (Context for Changes)

ARK is currently in **Phase 4 (Cognitive Intelligence)**:
- **Upcoming Phases:**
  - **Phase 4:** Iterative ReAct reasoning loops and subagent delegation.
  - **Phase 5:** High-Fidelity Research Memory (LanceDB).
  - **Phase 6:** Message Bus integration and interactive clarification.
  - **Phase 7:** Temporal Autonomy (Cron services).

Refer to `docs/ROADMAP.md` and `nanobot_adaptation_roadmap.md` for detailed transition goals.
