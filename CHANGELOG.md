# Changelog

All notable changes to the MultiModal Agentic RAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SOTA multimodal embeddings using Qwen3-VL-Embedding-2B (MMEB-V2 #1, 77.9 score)
- Knowledge graph-enhanced retrieval via LightRAG entity extraction
- 2-agent LangGraph workflow (Retriever → Generator)
- Hybrid local/API strategy with automatic fallback (98.5% cost savings vs GPT-4)
- Phoenix observability integration with OpenTelemetry tracing
- RAGAS evaluation framework (faithfulness, context precision/recall, answer relevancy)
- FastAPI REST API with auto-generated Swagger documentation
- CLI with query, ingest, evaluate, and serve commands
- Docker deployment with docker-compose
- Comprehensive documentation (README, Technical Report, Architecture Diagrams)
- 3 example scripts demonstrating system usage and extensibility
- GitHub Actions CI workflow with testing, linting, and Docker builds
- Makefile with convenience commands for common operations
- Contributing guidelines for developers

### Changed
- Removed Ollama dependency, migrated to pure HuggingFace architecture
- Unified vision model (Qwen2.5-VL-3B) reduces VRAM usage from ~16GB to ~10GB
- Simplified architecture to 2 agents for better performance
- Simplified documentation by archiving 14 non-essential files

### Fixed
- VRAM optimization via unified model instance (saves ~6GB)
- Keyword search fallback for LightRAG async conflicts

### Known Issues
- LightRAG hybrid retrieval currently using keyword search fallback due to async context manager conflicts with LangGraph
  - Root cause identified: Event loop conflicts
  - Multiple solution paths documented in [docs/LIGHTRAG_INTEGRATION.md](docs/LIGHTRAG_INTEGRATION.md)
  - System remains fully functional with keyword-based retrieval

## [0.1.0] - 2026-02-23

### Initial Release
- Multi-modal RAG system for claim matching and verification
- 2485 claims ingested with multimodal embeddings (text + images)
- 2-agent LangGraph workflow operational
- Knowledge graph with 95 entities and 1260 relationships
- CLI, REST API, and Docker deployment
- Phoenix observability integrated
- RAGAS evaluation metrics documented

---

## Version Summary

- **0.1.0** - Initial release (<3 days development)
- **Unreleased** - Ongoing enhancements and improvements

## Links

- **Documentation**: [README.md](README.md), [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- **Architecture**: [docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md)
- **Integration Status**: [docs/LIGHTRAG_INTEGRATION.md](docs/LIGHTRAG_INTEGRATION.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
