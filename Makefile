.PHONY: help install test lint format clean serve docs ingest query

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install dependencies
	uv sync

test:  ## Run tests
	uv run pytest -v

test-cov:  ## Run tests with coverage
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting (ruff)
	uv run ruff check src/

format:  ## Format code with black
	uv run black src/

format-check:  ## Check code formatting
	uv run black --check src/

type-check:  ## Run type checker (mypy)
	uv run mypy src/ --ignore-missing-imports

clean:  ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache

serve:  ## Start API server
	uv run ark serve

ingest:  ## Ingest sample PDF
	uv run ark ingest-dir ./data/pdf --pattern "transformers.pdf" --max-items 1

ingest-full:  ## Ingest all PDFs
	uv run ark ingest-dir ./data/pdf --pattern "*.pdf"

query:  ## Run a query
	@read -p "Enter your query: " query; \
	uv run ark query "$$query"

example-simple:  ## Run simple query example
	uv run python examples/simple_query.py

example-batch:  ## Run batch ingestion example
	uv run python examples/batch_ingest.py --max-items 10

example-custom:  ## Run custom agent example
	uv run python examples/custom_agent.py

docs:  ## Serve documentation
	@echo "Documentation available at:"
	@echo "  - README.md"
	@echo "  - docs/ARK_NANOBOT_ARCHITECTURE.md"
	@echo "  - docs/ROADMAP.md"
	@echo "  - docs/ARCHITECTURE_DIAGRAMS.md"

docker-build:  ## Build Docker image
	docker-compose build

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f api

.DEFAULT_GOAL := help
