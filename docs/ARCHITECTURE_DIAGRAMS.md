# Architecture Diagrams - Agentic Research Kit (ARK)

This document provides visual representations of the system architecture using Mermaid diagrams.

> **Note:** For diagrams covering the new Hybrid Event-Driven Architecture (Message Bus, Worker, and Skills), please refer to [ARK_NANOBOT_ARCHITECTURE.md](ARK_NANOBOT_ARCHITECTURE.md).

---

## Table of Contents

1. [High-Level System Architecture](#diagram-1-high-level-system-architecture)
2. [Research Query Flow](#diagram-2-research-query-flow)
3. [Universal Data Ingestion Pipeline](#diagram-3-universal-data-ingestion-pipeline)
4. [Evaluation & Observability](#diagram-4-evaluation--observability)

---

## Diagram 1: High-Level System Architecture

This diagram shows the complete system architecture with all layers and their interactions.

```mermaid
flowchart TB
    subgraph "User Layer"
        CLI[CLI Client<br/>ark query]
        API[REST API<br/>FastAPI /query endpoint]
        Gateway[Gateway<br/>Telegram Bot]
    end

    subgraph "3-Agent Orchestration Layer"
        LangGraph[LangGraph Workflow<br/>3-Agent ReAct Loop]
        A1[Agent 1: Enhanced Retriever<br/>Tool Registry]
        A2[Agent 2: Enhanced Response Generator<br/>Tool Registry]
        A3[Agent 3: Verification Node<br/>Critique Agent]
    end

    subgraph "Tool Layer"
        T1[EntityExtractorTool]
        T2[HybridRetrieverTool]
        T3[WebSearch/FetchTools]
        T4[RerankerTool]
    end

    subgraph "Model Layer"
        Embeddings[Qwen3-VL-Embedding-2B<br/>2048-dim Multimodal]
        LLM_Unified[Qwen3.5-4B<br/>Unified Vision/Text GPU]
        LLM_Fast[Qwen2.5-1.5B<br/>Fast Extraction GPU]
        LLM_API[DeepSeek / OpenAI API<br/>Reasoning LLM]
    end

    subgraph "Data Layer"
        LanceDB[(LanceDB<br/>High-Perf Storage)]
        KG[Knowledge Graph<br/>LightRAG]
        Memory[Semantic Memory<br/>LanceDB Store]
    end

    subgraph "Observability Layer"
        Phoenix[Phoenix Tracing<br/>Arize AI<br/>localhost:6006]
        Loguru[Loguru Logging<br/>Application Logs]
        RAGAS[RAGAS Evaluation<br/>Faithfulness<br/>Answer Relevancy]
    end

    %% User Layer to Orchestration
    CLI --> LangGraph
    API --> LangGraph
    Gateway --> LangGraph

    %% Orchestration Flow
    LangGraph --> A1
    A1 --> A2
    A2 --> A3
    A3 -- "Refine / Loop" --> A1
    A3 -- "Verified" --> END((END))
    
    LangGraph --> CLI
    LangGraph --> API

    %% Agents to Tools
    A1 --> T1
    A1 --> T2
    A1 --> T3
    A2 --> T4

    %% Tools to Models
    T1 --> LLM_Fast
    T4 --> LLM_Unified
    A2 --> LLM_API
    A3 --> LLM_Unified

    %% Tools to Data
    T2 --> LanceDB
    T2 --> KG

    %% Memory Integration
    A1 --> Memory
    A2 --> Memory

    %% Observability
    A1 --> Phoenix
    A2 --> Phoenix
    A3 --> Phoenix
    A1 --> Loguru
    A2 --> Loguru
    A3 --> Loguru

    %% Styling
    style LangGraph fill:#4A90E2,stroke:#2C5282,stroke-width:3px
    style Phoenix fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px
    style Loguru fill:#FFA726,stroke:#F57C00,stroke-width:2px
    style RAGAS fill:#50E3C2,stroke:#2E8B57,stroke-width:2px
    style LanceDB fill:#FFE0B2,stroke:#FF9800,stroke-width:2px
    style KG fill:#C8E6C9,stroke:#4CAF50,stroke-width:2px
    style Memory fill:#BA68C8,stroke:#7B1FA2,stroke-width:2px
    style A1 fill:#64B5F6,stroke:#1976D2,stroke-width:2px
    style A2 fill:#64B5F6,stroke:#1976D2,stroke-width:2px
    style A3 fill:#64B5F6,stroke:#1976D2,stroke-width:2px
```

### Key Components Explained

**User Layer**:
- **CLI**: Command-line interface (`ark`) for ingestion, queries, and evaluation.
- **API**: REST API for programmatic access and web integration.
- **Gateway**: Asynchronous communication channels (e.g., Telegram).

**3-Agent Orchestration**:
- **LangGraph**: Orchestrates the 3-agent ReAct loop with state management.
- **Agent 1 (Enhanced Retriever)**: Query analysis, entity extraction, and multi-source retrieval. Supports refinement based on Agent 3 feedback.
- **Agent 2 (Enhanced Response Generator)**: Reranking, evidence synthesis, and citation-rich draft generation.
- **Agent 3 (Verification Node)**: Fact-checking draft responses against sources. Can trigger a loop back to Agent 1 if gaps are found.

**Tool Layer**:
- **EntityExtractorTool**: Extracts key entities using local lightweight LLMs (Qwen2.5-1.5B).
- **HybridRetrieverTool**: Performs Vector + BM25 + KG retrieval via thread-isolated LightRAG backed by LanceDB.
- **WebSearch/Fetch**: Augments local knowledge with real-time web data (Brave Search).
- **RerankerTool**: Improves precision by re-ordering retrieved documents using Unified Qwen3.5.

**Model Layer**:
- **Embeddings**: Qwen3-VL-Embedding-2B for unified text/image vector space.
- **Unified Qwen3.5**: A single natively multimodal engine (4B parameters) for vision, reasoning, and fact-checking.
- **Fast Qwen2.5**: A lightweight 1.5B model optimized for rapid entity extraction.
- **API LLMs**: DeepSeek-R1 or GPT-4 for high-quality reasoning and synthesis.

---

## Diagram 2: Research Query Flow

This sequence diagram shows how a research query flows through the system.

```mermaid
sequenceDiagram
    participant U as User
    participant LG as LangGraph
    participant A1 as Enhanced Retriever
    participant Tools as Tool Registry
    participant A2 as Enhanced Response Generator
    participant A3 as Verification Agent
    participant Mem as Memory Store

    U->>LG: query_with_agents(query)
    LG->>Mem: Load research context (LanceDB)
    Mem-->>LG: context

    loop ReAct Refinement (Max 3 iterations)
        LG->>A1: Execute (query, context, feedback)
        activate A1
        A1->>Tools: execute("entity_extractor")
        Tools-->>A1: entities
        A1->>Tools: execute("hybrid_retriever")
        Tools-->>A1: local_docs (LanceDB)
        alt insufficient results
            A1->>Tools: execute("web_search")
            Tools-->>A1: web_docs
        end
        A1-->>LG: all_retrieved_docs
        deactivate A1

        LG->>A2: Execute (query, docs)
        activate A2
        A2->>Tools: execute("reranker")
        Tools-->>A2: top_docs
        A2->>A2: Synthesize evidence
        A2->>A2: Generate draft response
        A2-->>LG: draft_response, sources
        deactivate A2

        LG->>A3: Execute (query, sources, draft)
        activate A3
        A3->>A3: Fact-check response against sources
        alt gaps detected
            A3-->>LG: status="refine", feedback="missing context for X"
        else verified
            A3-->>LG: status="verified", verified_response
        end
        deactivate A3
        
        break if verified or max_iterations
        end
    end

    LG->>Mem: Log query to history (LanceDB)
    LG-->>U: result
```

---

## Diagram 3: Universal Data Ingestion Pipeline

This diagram shows how raw data from various sources is processed into structured knowledge.

```mermaid
flowchart LR
    subgraph "Input Sources"
        PDF[PDF Documents]
        DOCX[Word Docs]
        HTML[Web Pages]
        CSV[CSV Datasets]
    end

    subgraph "Universal Pipeline"
        Loader[Format-Specific Loaders<br/>PyMuPDF, python-docx, BS4]
        Parser[Generic Parser<br/>Text Cleaning & Metadata]
        Chunker[Semantic Chunker<br/>LightRAG Internal]
    end

    subgraph "Knowledge Extraction"
        LLM[Local extraction LLM<br/>Qwen2.5-1.5B]
        Embed[Multimodal Embedder<br/>Qwen3-VL-Embedding-2B]
    end

    subgraph "Persistent Storage"
        LanceDB[(LanceDB<br/>Vector & Doc Store)]
        KG[(NetworkX<br/>Knowledge Graph)]
    end

    %% Connections
    PDF & DOCX & HTML & CSV --> Loader
    Loader --> Parser
    Parser --> Chunker
    
    Chunker --> LLM
    Chunker --> Embed
    
    LLM --> KG
    Embed --> LanceDB
    Parser --> LanceDB

    %% Styling
    style LanceDB fill:#FFE0B2,stroke:#FF9800,stroke-width:2px
    style KG fill:#C8E6C9,stroke:#4CAF50,stroke-width:2px
```

---

## Diagram 4: Evaluation & Observability

This diagram shows the dual-stack observability and evaluation framework.

```mermaid
flowchart TB
    subgraph "Observability Stack"
        PX[Phoenix Tracing<br/>OTLP/OpenInference]
        LG[LangGraph Traces]
        LLM[LLM Call Details]
        Spans[Agent Spans]
        
        Log[Loguru Logging<br/>Structured/File]
    end

    subgraph "Evaluation (RAGAS)"
        F[Faithfulness]
        AR[Answer Relevancy]
        CP[Context Precision]
        CR[Context Recall]
    end

    subgraph "Workflow"
        W[ARK Research Workflow]
    end

    %% Connections
    W --> LG & LLM & Spans
    LG & LLM & Spans --> PX
    W --> Log
    W --> F & AR & CP & CR
```

---

**Last Updated**: 2026-03-11 (Updated with reference to the new Hybrid Event-Driven Architecture)
