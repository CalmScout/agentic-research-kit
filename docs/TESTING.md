# Testing Guide

## Overview

This document provides comprehensive documentation for the test suite of the Agentic Research Kit (ARK). The test suite ensures reliability, correctness, and maintainability of the 3-agent LangGraph workflow and high-performance storage backends.

## Test Statistics (Current)

- **Total Tests**: 53
- **Passing**: 53 (100%)
- **Test Framework**: `pytest` with `pytest-asyncio`
- **Coverage Target**: 70%+ on critical components (Current: >90% on core workflow)

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and mocks
├── test_agents/
│   ├── __init__.py
│   ├── test_adaptation.py          # nanobot feature adaptation tests
│   ├── test_enhanced_retriever.py  # Agent 1: Analysis & Retrieval
│   ├── test_enhanced_response_generator.py # Agent 2: Synthesis & Generation
│   ├── test_verification.py        # Agent 3: Fact-checking & Hallucination removal
│   ├── test_workflow_observability.py # Phoenix/OpenTelemetry integration
│   └── test_mcp.py                 # MCP client and tool discovery
├── test_workflow.py                 # Integration tests for complete pipeline
├── test_lancedb_storage.py          # LanceDB backend unit tests
├── test_api/
│   ├── __init__.py
│   └── test_endpoints.py           # FastAPI REST API tests
└── test_evaluation.py              # Evaluation metrics (RAGAS) tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests (automatically ignores unit_local to avoid crashing)
uv run python -m pytest tests/

# Run with coverage report
uv run python -m pytest --cov=src tests/

# Run specific test suite
uv run python -m pytest tests/test_agents/

# Run specific test file
uv run python -m pytest tests/test_agents/test_enhanced_retriever.py
```

### Resource Intensive Tests

Tests that load large local models (GPU required) are located in `tests/unit_local/` and are **ignored by default** in `pyproject.toml` to prevent machine unresponsiveness during standard development cycles.

To run them manually:
```bash
uv run python -m pytest tests/unit_local/
```

---

## Mocking Strategy

ARK uses a "Mock at Boundaries" strategy:
1. **LLMs**: All LLM calls (local and API) are mocked using `AsyncMock` to ensure tests are fast and cost-effective.
2. **Embeddings**: Vector generation is mocked with fixed-dimensional arrays.
3. **LanceDB**: Storage operations are tested against temporary directories or mocked connection objects.

---

## Best Practices

1. **Async Testing**: Always use `@pytest.mark.asyncio` for asynchronous functions.
2. **Isolation**: Each test should create its own temporary workspace or clean up existing state.
3. **Traceability**: New features must include tests that verify Phoenix/OpenTelemetry tracing spans are emitted correctly.

---

**Last Updated**: 2026-02-25 (Updated for 3-agent workflow and LanceDB migration)
