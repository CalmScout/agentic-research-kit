# ARK Roadmap

## Overview
This document outlines the evaluation of the Agentic Research Kit (ARK) architecture, the status of features adapted from the `nanobot` project, and the roadmap for future enhancements, specifically designed to merge LangGraph's predictability with Nanobot's event-driven autonomy.

## 1. Architectural Evaluation
ARK is designed as a high-performance, specialized RAG system for deep research. It is built to support local, single-user operation on a **12GB VRAM GPU** while retaining the flexibility to scale to multi-tenant cloud deployments.

### Strengths
- **LangGraph Orchestration**: Provides robust, state-aware, and highly predictable agent coordination. This deterministic execution is vital for debugging and long-term reliability.
- **Phoenix Observability**: Deep tracing of retrieval and generation pipelines.
- **Two-Layer Memory**: Sophisticated separation of long-term findings and query logs via LanceDB.
- **Local Privacy**: Capable of running specialized quantized models entirely locally.

### Opportunities & Weaknesses
- **Synchronous Bottleneck**: The current LangGraph pipeline blocks execution, preventing real-time event streaming ("Thinking...") to external channels without an async bus.
- **Hardware Contention**: Running parallel agents locally easily triggers OOM errors on a 12GB GPU. The system requires hardware-aware execution (sequential locally, parallel in the cloud).
- **Hardcoded Flexibility**: Core prompts are locked in Python files, lacking a dynamic, declarative skills system.
- **Proactivity**: The system is entirely reactive and lacks temporal autonomy for background research tasks.

## 2. Nanobot Adaptation Strategy

We will retain **LangGraph** as the core decision engine and augment it with `nanobot` patterns:

| Feature | Adaptation Strategy | Status |
| :--- | :--- | :--- |
| **Tool Registry** | Standardized declarative loading of internal tools. | **Adapted** |
| **Two-Layer Memory** | LanceDB backed long-term semantic memory. | **Adapted** |
| **MCP Support** | Standardized protocol for external tool servers. | **Adapted** |
| **Web Research** | Ported web search/fetch tools. | **Adapted** |
| **Async Message Bus** | LangGraph executes asynchronously, listening to inbound queues and emitting state updates (`.astream()`) to outbound queues. | **Pending (Phase 7)** |
| **Fractal Subagents** | Implemented natively as **LangGraph Subgraphs** (`Send` API). Configured to execute sequentially on local hardware and in parallel on cloud setups. | **Pending (Phase 7)** |
| **Temporal Autonomy** | Independent `CronService` and `HeartbeatService` background tasks that inject `[WAKE]` events into the Message Bus. | **Pending (Phase 8)** |
| **Skills System** | A `Skill Injector` LangGraph node dynamically loads `SKILL.md` constraints into the execution state before generation. | **Pending (Phase 9)** |

## 3. Implementation Roadmap

### Phase 1 to Phase 6: Core RAG & Hardening (Complete)
- [x] MCP Support & Dynamic Tool Loading
- [x] LanceDB High-Performance Storage Backend
- [x] Verification Node (Critique & Hallucination Cleanup)
- [x] Web Research Integration
- [x] Observability (Loguru, Phoenix) & RAGAS Evaluation
- [x] Iterative Reasoning Loop (ReAct)
- [x] Semantic Research Store (LanceDB Memory)

### Phase 7: Interface, Asynchronicity & Subgraphs (In Progress)
- [x] **Multi-Channel Gateway**: Telegram implementation complete.
- [ ] **Async Message Bus Integration**: Decouple the LangGraph executor from communication channels via `InboundMessage` and `OutboundMessage` queues. Allow `.astream()` updates to flow seamlessly to users.
- [ ] **Hardware-Aware Subgraphs**: Implement specialized research delegation using LangGraph Subgraphs.
    - *Local Mode (12GB VRAM)*: Sequential execution to prevent OOM.
    - *Cloud Mode*: Parallel execution for high-speed fan-out research.

### Phase 8: Temporal Autonomy & Scalability (Planned)
- [ ] **Cron & Heartbeat Services**: External async daemons that monitor schedules and system state, pushing triggers onto the Message Bus to proactively execute RAG workflows.
- [ ] **Tenant Isolation**: Add `user_id` routing tags in the Message Bus and LanceDB to support multi-user deployments.

### Phase 9: Advanced Agentic Capabilities (Planned)
- [ ] **Markdown-Driven Skills System**: Implement a dynamic `SkillsLoader` as the first node in the LangGraph, injecting domain-specific protocols (`SKILL.md`) directly into the agent's context.
- [ ] **Session Management**: Advanced multi-session lifecycles handling paused/resumed LangGraph state checkpoints.
