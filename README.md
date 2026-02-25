<p align="center">
  <img src="docs/assets/logo.png" width="400" alt="Agentic Research Kit Logo">
</p>

# Agentic Research Kit (ARK)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://github.com/langchain-ai/langgraph)

Agentic Research Kit (ARK) is a multimodal agentic RAG system for deep research and information synthesis. It utilizes a 3-agent [LangGraph](https://github.com/langchain-ai/langgraph) workflow to orchestrate knowledge graph traversal, high-performance vector search, and real-time web augmentation to produce grounded, citation-rich analysis from heterogeneous document sets.

## 🏗 Project Structure

```text
├── src/
│   ├── agents/             # LangGraph workflow and agent logic
│   │   ├── channels/       # Communication gateways (Telegram, etc.)
│   │   ├── memory/         # Persistent research context management (LanceDB)
│   │   ├── tools/          # Dynamic tool registry and RAG tools
│   │   ├── lancedb_storage.py # High-performance storage backends
│   │   └── workflow.py     # 3-agent orchestration logic
│   ├── api/                # FastAPI REST implementation
│   ├── data_ingestion/     # Universal pipeline (PDF, DOCX, HTML, CSV)
│   ├── evaluation/         # RAGAS and retrieval metric suites
│   └── utils/              # GPU model wrappers (Qwen3-VL, Qwen2.5)
├── docs/                   # Technical deep-dives and architecture diagrams
├── examples/               # Implementation patterns and use cases
├── rag_storage/            # Knowledge Graph storage
└── workspace/              # Long-term research memory (LanceDB)
```

## 🧠 Core Architecture

ARK implements a sequential 3-agent loop designed to minimize hallucination and maximize evidence grounding:

1.  **Enhanced Retriever (Agent 1)**:
    *   Analyzes query modality (text/image).
    *   Extracts entities using local lightweight LLMs (Qwen2.5-1.5B).
    *   Executes hybrid retrieval: Vector + BM25 + Knowledge Graph ([LightRAG](https://github.com/HKUDS/LightRAG)).
    *   Augments local context with Brave Search if internal knowledge is insufficient.

2.  **Enhanced Response Generator (Agent 2)**:
    *   Reranks candidates using cross-modal scoring.
    *   Synthesizes evidence into a structured summary.
    *   Generates draft response with strict source attribution.

3.  **Verification Node (Agent 3)**:
    *   Expert critique node that fact-checks the response against retrieved sources.
    *   Detects and removes hallucinations before the final answer is returned.
    *   Ensures 100% citation integrity.

## 🛠 Technology Stack

| Component | Technology | Implementation Details |
|-----------|-----------|------------------------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | State-aware 3-agent sequential workflow |
| **RAG Engine** | [LightRAG](https://github.com/HKUDS/LightRAG) | Hybrid Vector + Knowledge Graph traversal |
| **Storage Backend**| [LanceDB](https://lancedb.com/) | High-performance columnar storage for KV and Vectors |
| **Embeddings** | [Qwen3-VL-2B](https://huggingface.co/Qwen) | Unified 2048D cross-modal vector space |
| **Inference (Local)** | [Qwen2.5](https://huggingface.co/Qwen) | Entity extraction and fact-checking |
| **Inference (API)** | [DeepSeek](https://www.deepseek.com/) / [OpenAI](https://openai.com/) | Primary reasoning and synthesis |
| **Observability** | [Arize Phoenix](https://phoenix.arize.com/) | Distributed OTLP tracing and span analysis |
| **Evaluation** | [RAGAS](https://github.com/explodinggradients/ragas) | Faithfulness, Relevancy, and Context metrics |

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU (Recommended 12GB+ VRAM for full local mode)

### Setup
```bash
# Clone and install dependencies
git clone https://github.com/CalmScout/agentic-research-kit
cd agentic-research-kit
uv sync

# Configure environment
cp .env.defaults .env
# Edit .env to add API keys (DEEPSEEK_API_KEY, BRAVE_API_KEY)
```

### Operation

**1. Ingestion**
Ingest documents into the local knowledge graph:
```bash
uv run ark ingest-dir ./data/research_papers --pattern "*.pdf"
```

**2. Research Query**
Run a multimodal or text-only query via CLI:
```bash
uv run ark query "How does Multi-Head Attention scale?" --mode hybrid
```

**3. REST API**
Start the FastAPI server:
```bash
uv run ark serve
```

**4. Gateway (Telegram)**
Start the asynchronous communication gateway:
```bash
uv run ark gateway
```

**5. Observability**
Launch the Phoenix trace explorer:
```bash
PHOENIX_ENABLED=true uv run python -m phoenix.server.main serve
```

## 📅 Roadmap

ARK is transitioning from a synchronous RAG pipeline to an event-driven agent architecture:

*   ✅ **Phase 1: Extensibility**: MCP Client implementation and dynamic tool loading.
*   ✅ **Phase 2: Performance & Integrity**: LanceDB migration and Verification Node (Critique).
*   ✅ **Phase 3: Research Scope Expansion**: Integrated Web Search and Proactive Research.
*   ⬜ **Phase 4: Interface & Asynchronicity**: Message Bus integration and interactive clarification.
*   ⬜ **Phase 5: Cognitive Architecture**: Iterative ReAct reasoning loops and async event bus.
*   ⬜ **Phase 6: Temporal Autonomy**: Cron services and self-waking research tasks.

## 📝 License
Apache License 2.0
