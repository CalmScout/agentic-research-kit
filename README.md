<p align="center">
  <img src="docs/assets/logo.png" width="400" alt="Agentic Research Kit Logo">
</p>

# Agentic Research Kit (ARK)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://github.com/langchain-ai/langgraph)

Agentic Research Kit (ARK) is a multimodal, agentic RAG system engineered to solve the core limitations of standard LLM applications: **hallucinations, superficial answers, and lack of verifiability.** 

Designed as a rigorous research synthesis tool, ARK shifts the paradigm from simple "prompt-and-respond" to a self-correcting, multi-agent workflow. By orchestrating a 3-agent [LangGraph](https://github.com/langchain-ai/langgraph) loop, it combines knowledge graph traversal, high-performance vector search, and real-time web augmentation. The system leverages a unified local multimodal architecture to ensure that every output is logically sound, empirically grounded, and enriched with strict, verifiable citations.

## 🎯 The Problem & The Solution

**The Problem:** Standard RAG and LLM systems fail in deep research contexts. They often blindly trust top-k vector similarities, synthesize conflicting information poorly, and hallucinate facts when context is sparse. They lack the cognitive architecture to pause, reflect, and verify their own claims.

**The Solution:** ARK introduces a cyclic, critically-evaluating architecture:
*   **Adaptive Context:** It doesn't just search; it formulates a research plan, extracts semantic entities, and queries a knowledge graph to understand relationship context, not just vector distance.
*   **Self-Correction:** It employs a dedicated Verification Agent that actively critiques generated responses against the raw retrieved data, forcing the system to rewrite and drop unsupported claims before the user ever sees the output.
*   **Resource Efficiency:** It achieves this high-level cognitive intelligence entirely locally on consumer hardware (12GB VRAM) by consolidating tasks into a single natively multimodal `Qwen3.5` engine utilizing 8-bit quantization.

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
│   └── utils/              # GPU model wrappers (Unified Qwen3.5 architecture)
├── docs/                   # Technical deep-dives and architecture diagrams
├── examples/               # Implementation patterns and use cases
├── rag_storage/            # Knowledge Graph storage
└── workspace/              # Long-term research memory (LanceDB)
```

## 🧠 Core Architecture

ARK implements a sequential 3-agent loop designed to minimize hallucination and maximize evidence grounding. This workflow operates using a natively unified multimodal engine (`Qwen3.5-4B`), which allows a single model instance to seamlessly analyze both text and visual inputs within a strict 12GB local VRAM footprint.

1.  **Enhanced Retriever (Agent 1): The Researcher**
    *   Analyzes query modality (text/image) using the unified reasoning engine.
    *   Extracts entities from the query to formulate dynamic search plans.
    *   Executes hybrid retrieval: Vector + BM25 + Knowledge Graph ([LightRAG](https://github.com/HKUDS/LightRAG)).
    *   Proactively augments local context with Brave Search if internal knowledge is deemed insufficient.

2.  **Enhanced Response Generator (Agent 2): The Synthesizer**
    *   Reranks candidates using cross-modal scoring to prioritize highest-signal data.
    *   Synthesizes evidence into a structured, highly cohesive summary.
    *   Generates draft responses with strict, inline source attribution.

3.  **Verification Node (Agent 3): The Critic**
    *   Expert critique node that fact-checks the response against retrieved sources via iterative ReAct loops.
    *   Detects and removes hallucinations, demanding rewrites if claims lack explicit grounding.
    *   Ensures 100% citation integrity.

## 🛠 Technology Stack

| Component | Technology | Implementation Details |
|-----------|-----------|------------------------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | State-aware 3-agent sequential workflow with ReAct logic |
| **RAG Engine** | [LightRAG](https://github.com/HKUDS/LightRAG) | Hybrid Vector + Knowledge Graph traversal |
| **Storage Backend**| [LanceDB](https://lancedb.com/) | High-performance columnar storage for KV and Vectors |
| **Embeddings** | [Qwen3-VL-2B](https://huggingface.co/Qwen) | Unified 2048D cross-modal vector space |
| **Inference (Local)** | [Qwen3.5-4B](https://huggingface.co/Qwen) | Unified dense model for entity extraction, vision, and fact-checking |
| **Inference (API)** | [DeepSeek](https://www.deepseek.com/) / [OpenAI](https://openai.com/) | Configurable cloud fallback for primary reasoning |
| **Observability** | [Arize Phoenix](https://phoenix.arize.com/) | Distributed OTLP tracing and span analysis |
| **Evaluation** | [RAGAS](https://github.com/explodinggradients/ragas) | Faithfulness, Relevancy, and Context metrics |

## 🧪 Development & CI/CD

Before pushing changes, you can run the full CI/CD check suite locally:

```bash
# Run all checks (Lint, Format, Type, Test)
make lint format-check type-check test

# Or individually:
make lint          # Ruff linter
make format-check  # Black formatting check
make format        # Apply Black formatting
make type-check    # Mypy type checking
make test          # Pytest suite
```

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
*   ✅ **Phase 3.5: Hardening & Evaluation**: RAGAS evaluation pipeline and 90% test coverage.
*   ✅ **Phase 4: Cognitive Intelligence**: Iterative ReAct reasoning loops and unified `Qwen3.5` local model consolidation.
*   🔄 **Phase 5: High-Fidelity Research Memory**: Semantic research store in LanceDB (In Progress).
*   ⬜ **Phase 6: Interface & Asynchronicity**: Message Bus integration and interactive clarification.
*   ⬜ **Phase 7: Temporal Autonomy**: Cron services and self-waking research tasks.

## 📚 Documentation

Detailed technical guides are available in the `docs/` directory:

- **[Architecture Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)**: Visual deep-dive into the 3-agent research flow.
- **[Qwen3.5 Research & Roadmap](docs/QWEN35_ARCHITECTURE_RESEARCH.md)**: Detailed reasoning behind the unified model consolidation and future vLLM scaling plans.
- **[LightRAG Integration](docs/LIGHTRAG_INTEGRATION.md)**: Details on hybrid retrieval and LanceDB storage.
- **[Logging Infrastructure](docs/LOGGING.md)**: Details on structured logging and log management.
- **[Observability Setup](docs/PHOENIX.md)**: Guide to using Arize Phoenix for trace analysis.
- **[Testing Guide](docs/TESTING.md)**: Instructions for running the 50+ test suite and coverage reports.
- **[RAGAS Usage](docs/RAGAS_USAGE.md)**: Automated evaluation of retrieval and generation quality.

## 📝 License
Apache License 2.0

