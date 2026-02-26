# ARK & nanobot Architecture Comparison

This document provides a comparative analysis of the architectures for ARK (Agentic Research Kit) and nanobot, illustrating the project's transition from a sequential LangGraph workflow to an event-driven agent system.

---

## 1. ARK (Agentic Research Kit) Architecture (Current State)

ARK is currently implemented as a sequential multi-agent workflow using **LangGraph**. It has been enhanced with core patterns from `nanobot` (like the Tool Registry and Two-Layer Memory) and uses **LanceDB** for high-performance storage.

### ARK Mermaid Diagram

```mermaid
graph TD
    subgraph "Interfaces"
        CLI[CLI Query]
        API[FastAPI Server]
        TG[Telegram Gateway]
    end

    subgraph "ARK Core Workflow (LangGraph)"
        START((START)) --> ER[Enhanced Retriever Agent]
        ER --> ERG[Enhanced Response Generator Agent]
        ERG --> VN[Verification Node Agent]
        VN --> END((END))
    end

    subgraph "Agent 1: Enhanced Retriever"
        ER_TR[Tool Registry]
        ER_EXT[Entity Extractor]
        ER_RAG[Hybrid RAG Tool]
        ER_WEB[Web Search/Fetch Tools]
        ER_MCP[MCP Tools]
        
        ER --> ER_TR
        ER_TR --> ER_EXT & ER_RAG & ER_WEB & ER_MCP
    end

    subgraph "Agent 2: Enhanced Response Generator"
        ERG_TR[Tool Registry]
        ERG_RER[Reranker Tool]
        ERG_AGG[Evidence Aggregator]
        ERG_GEN[Response Generator]
        
        ERG --> ERG_TR
        ERG_TR --> ERG_RER & ERG_AGG & ERG_GEN
    end

    subgraph "Agent 3: Verification Node"
        VN_FC[Fact Checker]
        VN_CH[Hallucination Cleanup]
        
        VN --> VN_FC --> VN_CH
    end

    subgraph "Memory & Storage"
        LDB[(LanceDB Store)]
        KG[(NetworkX KG)]
        MS_RH[RESEARCH_MEMORY.md]
        MS_QH[QUERY_HISTORY.md]
        PX[Phoenix Observability]
    end

    CLI & API & TG --> START
    
    ER_RAG <--> LDB & KG
    ER_WEB <--> WEB[World Wide Web]
    
    ERG_GEN <--> LDB
    VN_FC <--> LDB
    
    ER -.-> PX
    ERG -.-> PX
    VN -.-> PX
```

### ARK Key Characteristics
*   **3-Agent Sequential Workflow**: Orchestrates three specialized agents (Retriever, Generator, Verifier) via LangGraph.
*   **Verification Node**: Explicit fact-checking step to prevent hallucinations and ensure citation integrity.
*   **LanceDB Storage**: High-performance binary storage for documents, entities, and vectors.
*   **Tool Registry**: Agents use a dynamic `ToolRegistry` to execute specialized tasks.
*   **Two-Layer Memory**: Persistent context via `RESEARCH_MEMORY.md` (distilled findings) and `QUERY_HISTORY.md` (audit logs).
*   **Hybrid Retrieval**: Combined Vector + BM25 + Knowledge Graph retrieval.

---

## 2. nanobot Architecture (Target State)

nanobot represents the **event-driven, asynchronous** architecture that ARK is transitioning towards. It decouples the agent core from communication channels using a Message Bus.

### nanobot Mermaid Diagram

```mermaid
graph LR
    subgraph "Multi-Channel Inbound"
        C_TG[Telegram]
        C_DS[Discord]
        C_SL[Slack]
        C_CLI[CLI]
    end

    subgraph "Message Bus"
        MB_IN{Inbound Queue}
        MB_OUT{Outbound Queue}
    end

    subgraph "Agent Loop Core Engine"
        AL[Agent Loop]
        TR[Tool Registry]
        CTX[Context Builder]
        SM[Session Manager]
        SAM[Subagent Manager]
    end

    subgraph "Tools"
        T_SH[Shell/Exec]
        T_FS[Filesystem]
        T_WB[Web Search/Fetch]
        T_MCP[MCP Tools]
        T_SP[Spawn Subagent]
    end

    subgraph "Brain"
        LLM[LLM Provider]
        MEM[(Semantic Memory)]
    end

    %% Flow
    C_TG & C_DS & C_SL & C_CLI --> MB_IN
    MB_IN --> AL
    
    AL <--> CTX
    AL <--> SM
    AL <--> TR
    AL <--> LLM
    AL --> SAM
    
    TR --> T_SH & T_FS & T_WB & T_MCP & T_SP
    
    SM <--> MEM
    SAM -.-> MB_IN
    
    AL --> MB_OUT
    MB_OUT --> C_TG & C_DS & C_SL & C_CLI
```

### nanobot Key Characteristics
*   **Event-Driven**: Built around a `MessageBus` that decouples channels from the agent logic.
*   **Asynchronous Processing**: Handles multiple concurrent sessions and background tasks.
*   **Fractal Delegation**: Capability to spawn specialized subagents via `SpawnTool`.
*   **Rich Toolset**: General purpose tools including filesystem, shell, and full MCP support.
*   **Semantic Memory**: Moving from flat-files to vector-based memory consolidation.

---

## 3. Key Differences & Evolution

| Feature | ARK (Current) | nanobot (Future Target) |
| :--- | :--- | :--- |
| **Execution Model** | Sequential (3-Agent Pipeline) | Event-Driven (Reactive Loop) |
| **Orchestration** | LangGraph (3 Nodes) | Agent Loop + Message Bus |
| **Communication** | Direct (Request-Response) | Pub/Sub via Message Bus |
| **Task Handling** | Predefined Steps | Dynamic Tool Execution |
| **Delegation** | Fixed Pipeline | Subagent Spawning (Fractal) |
| **Memory Store** | LanceDB (Columnar) | Vector Store (Semantic RAG) |

The project has successfully adapted the **Tool Registry**, **Two-Layer Memory**, **MCP Support**, and **Channel Gateways** (Telegram) from nanobot. The next major phases involve **Phase 4: Cognitive Intelligence** (Iterative ReAct loops and Subagent Manager) and **Phase 6: Interface & Asynchronicity** (Message Bus for true asynchronicity).

---

**Last Updated**: 2026-02-26 (Updated to align with Phase 4/6 roadmap focus)
