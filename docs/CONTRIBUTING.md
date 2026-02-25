# Contributing to Agentic Research Kit (ARK)

Thank you for your interest in contributing! This document provides guidelines for contributing to the Agentic Research Kit (ARK) project.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended for local model execution)
- Arize Phoenix (for observability)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/CalmScout/agentic-research-kit.git
cd agentic-research-kit
```

2. **Install dependencies:**
```bash
uv sync
```

3. **Configure environment:**
```bash
cp .env.defaults .env
# Edit .env and add your API keys (DEEPSEEK_API_KEY, BRAVE_API_KEY)
```

### Running Tests

```bash
# Run all tests (automatically ignores unit_local)
uv run python -m pytest tests/

# Run with coverage
uv run python -m pytest --cov=src tests/

# Run specific test file
uv run python -m pytest tests/test_agents/test_enhanced_retriever.py
```

### Code Quality

This project uses the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

```bash
# Use Makefile for convenience
make format    # Format code
make lint      # Run linter
make test      # Run tests
```

## Project Structure

```text
agentic-research-kit/
├── src/                    # Source code
│   ├── agents/            # Multi-agent system (LangGraph)
│   ├── data_ingestion/    # Universal pipeline
│   ├── utils/             # GPU model wrappers & config
│   ├── evaluation/        # RAGAS evaluation
│   └── api/               # FastAPI REST API
├── tests/                 # Test suite
├── docs/                  # Documentation
├── rag_storage/           # LanceDB storage
└── data/                  # Research datasets
```

## Architecture Overview

ARK implements a **3-agent LangGraph workflow**:

1. **Enhanced Retriever** (Agent 1) - Query analysis, entity extraction, and hybrid retrieval (Vector + BM25 + KG).
2. **Enhanced Response Generator** (Agent 2) - Reranking, evidence synthesis, and citation-rich draft generation.
3. **Verification Node** (Agent 3) - Expert critique and fact-checking against retrieved sources.

See **[Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)** for more details.

## Making Changes

### Commit Messages

Follow conventional commit format:

```
feat(retriever): add support for semantic reranking
fix(storage): resolve LanceDB connection leak
docs(api): update endpoint documentation
test(workflow): add observability tests
```

### Pull Request Process

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Ensure tests and linting pass (`make test && make lint`).
5. Update documentation if necessary.
6. Open a Pull Request.

## Testing Guidelines

### Unit Tests
- Use `AsyncMock` for LLM and external API calls.
- Avoid resource-intensive tests in the main suite (move to `tests/unit_local/`).

### Test Coverage
- Maintain >80% coverage on core agent and workflow logic.

## Documentation

- Use Google-style docstrings.
- Update technical guides in `docs/` when introducing major architectural changes.

---

**Last Updated**: 2026-02-25
