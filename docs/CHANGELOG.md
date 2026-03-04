# Changelog

All notable changes to the Agentic Research Kit (ARK) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Centralized Logging & Observability**: Unified all logging into a single `Loguru` infrastructure with automated file rotation, retention, and dual-sink output (JSON/Text).
- **Global Observability Hub**: Centralized Phoenix and OpenTelemetry setup into `src/utils/observability.py` with automatic initialization on workflow import.
- **Unified Configuration**: Integrated all logging and tracing settings into the Pydantic `Settings` class for consistent environment-based control.
- **3-Agent Architecture**: Introduced a dedicated **Verification Node** (Agent 3) for expert critique and hallucination removal.
- **LanceDB Integration**: Migrated storage backends (KV, Vector, Doc Status) to LanceDB for high-performance binary storage and scalability.
- **Thread-Isolated Retrieval**: Successfully resolved LightRAG async conflicts via a dedicated background event loop, enabling full local hybrid retrieval.
- **Semantic Memory**: Implemented LanceDB-backed persistent research findings memory.
- **Enhanced Observability**: Integrated Loguru for structured logging and added OpenTelemetry tracing to ToolRegistry and Embeddings.
- **MCP Tool Support**: Added capability to dynamically load and execute tools from Model Context Protocol servers.
- **Web Research Tools**: Ported proactive Brave Search and Web Fetch tools from the nanobot framework.
- **Telegram Gateway**: Added asynchronous communication channel for remote research queries.
- **RAGAS Evaluation Pipeline**: Established a "Golden Dataset" and automated evaluator for retrieval and generation quality.

### Changed
- **Project Rebranding**: Renamed from "MultiModal Agentic RAG" to **Agentic Research Kit (ARK)**.
- **Documentation Refactor**: Consistently moved technical guides to the `docs/` directory and updated all architecture diagrams.
- **Test Suite Modernization**: Achieved >90% coverage on core workflow components and resolved all deprecation/runtime warnings (Phase 3.5 Complete).
- **Roadmap Shift**: Transitioned focus to **Phase 4: Cognitive Intelligence**, prioritizing iterative reasoning loops (ReAct) and subagent delegation.

### Fixed
- **VRAM Optimization**: Unified model loading for Qwen3-VL and Qwen2.5 saves ~6GB VRAM.
- **Async Safety**: Fixed "object dict can't be used in await" and "coroutine never awaited" warnings in tests.
- **LanceDB Deprecations**: Replaced `table_names()` with `list_tables()`.

## [0.1.0] - 2026-02-23

### Initial Release
- Multi-modal RAG system for research and verification.
- 2-agent LangGraph workflow operational.
- Unified 2048D multimodal embeddings (Qwen3-VL).
- Knowledge graph integration (LightRAG).
- CLI, REST API, and Docker deployment.
- Arize Phoenix observability integration.

---

## Version Summary

- **0.1.0** - Initial release.
- **Unreleased** - Transition to 3-agent architecture and LanceDB.

## Links

- **Documentation**: [README.md](../README.md)
- **Architecture**: [docs/ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)
- **Testing**: [docs/TESTING.md](TESTING.md)
- **Observability**: [docs/PHOENIX.md](PHOENIX.md)
