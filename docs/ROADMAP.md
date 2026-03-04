# ARK Roadmap

## Overview
This document outlines the evaluation of the Agentic Research Kit (ARK) architecture, the status of features adapted from the `nanobot` project, and the roadmap for future enhancements.

## 1. Architectural Evaluation
ARK is designed as a high-performance, specialized RAG system for deep research.

### Strengths
- **LangGraph Orchestration**: Provides robust, state-aware agent coordination.
- **Phoenix Observability**: Deep tracing of retrieval and generation pipelines.
- **Two-Layer Memory**: Sophisticated separation of long-term findings and query logs.
- **Multimodal Support**: Native handling of text and image data.

### Opportunities
- **Performance**: The current JSON-based storage backend (JsonKVStorage) is a major bottleneck for large datasets and complex queries.
- **Reliability**: The linear pipeline lacks an explicit "Critique & Verify" step to cross-check hallucinations against cited sources.
- **Flexibility**: Tool and provider management are currently more "hardcoded" than in the nanobot framework.
- **Proactivity**: The system is reactive; it lacks nanobot's background tasks and scheduling.
- **Event-Driven Architecture**: ARK relies on synchronous request-response cycles, whereas `nanobot` utilizes an asynchronous `MessageBus` to decouple communication channels.

## 2. Nanobot Adaptation Status

| Feature | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Tool Registry** | Dynamic management and execution of tools. | **Adapted** | Located in `src/agents/tools/registry.py`. |
| **Two-Layer Memory** | `RESEARCH_MEMORY.md` and `QUERY_HISTORY.md`. | **Adapted** | Located in `src/agents/memory/store.py`. |
| **MCP Support** | Standardized protocol for external tool servers. | **Adapted** | Implemented in `src/agents/tools/mcp.py` and integrated into Retriever. |
| **Provider Registry** | Declarative LLM provider management. | **Adapted** | Refactored `ModelSelector` using `src/agents/providers.py`. |
| **Memory Consolidation** | LLM-based distillation of history into findings. | **Adapted** | Implemented `consolidate_session` in `MemoryStore`. |
| **Web Research Tools** | Integrated web search and fetching capabilities. | **Adapted** | Ported `WebSearchTool` and `WebFetchTool` to `src/agents/tools/web.py`. |
| **Channel Gateway** | Multi-channel chat interface. | **Adapted** | `ChannelManager` and **Telegram** implementation complete. |
| **Message Bus** | Decoupled event queue for async handling. | **Pending** | Core agent logic is still synchronous/linear. |
| **Cron/Scheduling** | Ability for agent to schedule future tasks. | **Pending** | `CronTool` and `CronService` not yet ported. |
| **Subagents** | Fractal delegation of tasks. | **Pending** | `SpawnTool` and `SubagentManager` not yet ported. |

## 3. Implementation Roadmap

### Phase 1: Extensibility (Complete)
- [x] **Implement MCP Client**: Enable ARK to connect to stdio and HTTP-based MCP servers.
- [x] **Dynamic Tool Loading**: Allow tools discovered via MCP to be registered in the `ToolRegistry`.
- [x] **Configuration Integration**: Added `mcp_servers` to global settings.

### Phase 2: Performance & Integrity (Complete)
- [x] **High-Performance Storage Backend**:
    - [x] Migrated from JSON-based storage to **LanceDB**.
    - [x] Implemented `LanceDBKVStorage`, `LanceDBDocStatusStorage`, and `LanceDBVectorDBStorage`.
    - [x] This resolves the "slowness" issue by using binary storage and optimized indexing for vector and text search.
- [x] **Verification Node (Critique)**:
    - [x] Implemented a "Critique" node in the LangGraph workflow that validates generated claims against retrieved evidence before showing the final answer.
- [x] **Semantic Memory Store**:
    - [x] Migrate `RESEARCH_MEMORY.md` to the local vector store to support retrieval from thousands of past findings without context window limits.

### Phase 3: Research Scope Expansion (Complete)
- [x] **Integrated Web Search**: 
    - Ported `WebSearchTool` (Brave Search) and `WebFetchTool` from nanobot.
- [x] **Proactive Research**:
    - Implemented `ResearchTaskManager` for background task execution.
    - Added `EntityDeepDiveTool` to trigger asynchronous research into detected entities.

### Phase 3.5: Hardening & Evaluation (Complete)
- [x] **Test Suite Repair & Modernization**:
    - Fixed broken unit tests and resolved deprecation/runtime warnings.
    - Achieved >90% coverage on the core 3-agent LangGraph workflow components.
- [x] **Observability Deepening**:
    - Integrated **Loguru** across all core agents and storage components.
    - Added **OpenTelemetry/Arize Phoenix** tracing to ToolRegistry and Embedding generation for deep visibility.
- [x] **RAGAS Evaluation Pipeline**:
    - Established a "Golden Dataset" in `data/research_golden_dataset.csv`.
    - Implemented `RAGASEvaluator` and `run_ragas_eval.py` script.

### Phase 4: Cognitive Intelligence (Complete)
- [x] **Iterative Reasoning Loop (ReAct)**:
    - Transitioned `workflow.py` from a linear pipeline (Retrieve -> Generate -> Verify) to a dynamic loop (Reason -> Act -> Observe -> Repeat).
    - Enabled the `VerificationNode` to trigger "Refinement Cycles" if evidence is insufficient or hallucinations are detected.
    - Implemented "Research Gap" detection in `EnhancedRetriever` to guide subsequent retrieval steps.
- [ ] **Subagent Delegation**:
    - Port `SubagentManager` and `SpawnTool` from nanobot for ephemeral "Deep Dive" agents to handle specialized sub-tasks. (Pending integration)

### Phase 5: High-Fidelity Research Memory (In Progress)
- [x] **Semantic Research Store (LanceDB)**:
    - [x] Fully migrated `RESEARCH_MEMORY.md` findings to a vector-enabled LanceDB table in `src/agents/memory/store.py`.
    - [x] Implemented "Long-term RAG" to allow retrieval from past research findings across different sessions.
- [ ] **Memory Hardening**:
    - [ ] Add semantic deduplication to prevent redundant findings from clogging the memory.
    - [ ] Implement automated session consolidation triggered by `ResearchTaskManager`.

### Phase 6: Interface & Asynchronicity (Planned)
- [ ] **Message Bus Integration**:
    - Refactor `ChannelManager` to use an async `MessageBus` (Inbound/Outbound queues) to decouple communication channels from the Agent processor.
- [ ] **Async Event Bus**:
    - Refactor the core loop to be asynchronous, emitting "thinking" events and partial updates via the `MessageBus`.
- [ ] **Interactive Clarification**:
    - Enable the agent to pause execution and ask the user for clarification via the `ChannelGateway` if a query is ambiguous (Human-in-the-Loop).
- [ ] **Slack/Discord Gateway**: 
    - Port Slack and Discord channels from nanobot for broader collaboration.

### Phase 7: Temporal Autonomy (Planned)
- [ ] **Cron Service**:
    - Port `CronService` and `CronTool` for scheduled research tasks.
- [ ] **Self-Wake**:
    - Enable the system to wake up on schedule and push notifications to Telegram/Slack without user initiation.
