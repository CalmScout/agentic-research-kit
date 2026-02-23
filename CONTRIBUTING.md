# Contributing to MultiModal Agentic RAG

Thank you for your interest in contributing! This document provides guidelines for contributing to the MultiModal Agentic RAG project.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended) or CPU
- Docker & Docker Compose (for deployment)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/multimodal-agentic-rag.git
cd multimodal-agentic-rag
```

2. **Install dependencies:**
```bash
uv sync --dev
```

3. **Configure environment:**
```bash
cp .env.defaults .env
# Edit .env and add your DEEPSEEK_API_KEY (optional)
```

### Running Tests

```bash
# Run all tests
uv run pytest -v

# Run tests with coverage
uv run pytest -v --cov=src --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/test_agents/test_query_analyzer.py
```

### Code Quality

This project uses the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

```bash
# Format code
uv run black src/

# Check formatting
uv run black --check src/

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/ --ignore-missing-imports
```

Or use the Makefile:

```bash
make format    # Format code
make lint      # Run linter
make test      # Run tests
```

## Project Structure

```
multimodal-agentic-rag/
├── src/                    # Source code
│   ├── agents/            # Multi-agent system
│   ├── data_ingestion/    # Data processing
│   ├── utils/             # Utilities and config
│   ├── evaluation/        # Evaluation metrics
│   └── api/               # FastAPI REST API
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
├── rag_storage/           # LightRAG storage (generated)
├── data/                  # Dataset files
└── main.py               # CLI entry point
```

## Architecture Overview

The system implements a **2-agent LangGraph workflow**:

1. **Enhanced Retriever** (Agent 1) - Query preprocessing, entity extraction, and hybrid retrieval (Vector + BM25 + KG)
2. **Enhanced Response Generator** (Agent 2) - Reranking, evidence synthesis, and final response generation with citations

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed architecture documentation.

## Making Changes

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature branches
- `bugfix/*` - Bug fix branches

### Commit Messages

Follow conventional commit format:

```
feat: add custom agent support
fix: resolve LightRAG async conflict
docs: update README with badges
test: add integration tests for workflow
refactor: simplify model selector logic
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test && make lint`)
5. Commit your changes (`git commit -m 'feat: amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted with black
- [ ] No linting errors
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow conventional format

## Adding New Features

### Adding a New Agent

1. Create agent function in `src/agents/`
2. Follow the pattern:
```python
async def my_custom_agent(state: AgentState) -> AgentState:
    """Custom agent that does X."""
    # Your logic here
    return updated_state
```

3. Add to workflow in `src/agents/workflow.py`
4. Add tests in `tests/test_agents/`
5. Update documentation

### Adding New Models

1. Add model configuration in `src/utils/config.py`
2. Create wrapper class in appropriate module
3. Update model selector logic
4. Add tests
5. Update documentation

### Adding New Evaluation Metrics

1. Implement metric in `src/evaluation/`
2. Add to evaluation suite
3. Document in TECHNICAL_REPORT.md
4. Add example usage

## Testing Guidelines

### Unit Tests

- Test individual functions and classes
- Use mocks for external dependencies (API calls, models)
- Follow naming: `test_<function_name>`

### Integration Tests

- Test agent workflows end-to-end
- Test with real data (small subset)
- Follow naming: `test_integration_<feature>`

### Test Coverage

- Aim for 70%+ coverage on critical components
- Use `pytest-cov` for coverage reports

## Documentation

### Code Documentation

- Use docstrings for all functions and classes
- Follow Google style guide:
```python
def query_rag(query: str, top_k: int = 50) -> List[Dict]:
    """Query the RAG system and retrieve relevant documents.

    Args:
        query: The query string
        top_k: Number of documents to retrieve

    Returns:
        List of retrieved documents with metadata
    """
```

### README Updates

When adding features:
1. Update "Features" section if applicable
2. Add usage examples
3. Update architecture diagram if needed

### ADR Updates

For significant architectural changes:
1. Create new ADR in DESIGN.md
2. Follow ADR template (Decision, Why, Trade-off, Consequences)
3. Update TECHNICAL_REPORT.md

## Questions or Issues?

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Documentation**: See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) and [docs/](docs/)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build great software together.

## License

By contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0.
