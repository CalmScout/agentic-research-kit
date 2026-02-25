# Phoenix Observability Setup Guide

## Overview

Phoenix (Arize AI) provides **open-source observability** for the Agentic Research Kit (ARK). It automatically traces all agent executions, tool calls, embeddings, and generations in real-time.

**Key Benefits**:
- 🔍 **Auto-instrumentation**: Traces LangGraph and LangChain automatically.
- 📊 **Visual Trace Explorer**: Inspect every step of the 3-agent research workflow.
- 🐛 **Deep Debugging**: Identify hallucinations or retrieval failures by inspecting exact tool inputs/outputs.
- 📈 **Performance Insights**: Monitor latency and token usage across local and API models.

---

## Quick Start

### 1. Start Phoenix Server

Phoenix UI will be available at: **http://localhost:6006**

```bash
# Start Phoenix in the background
uv run python -m phoenix.server.main serve &
```

### 2. Enable Tracing

Set the `PHOENIX_ENABLED` environment variable before running queries or starting the API.

```bash
# Via CLI
PHOENIX_ENABLED=true uv run ark query "How does Multi-Head Attention scale?"

# Via API
PHOENIX_ENABLED=true uv run ark serve
```

---

## Tracing Architecture

ARK implements deep tracing across several layers:

### 1. Workflow Tracing
Every research query generates a top-level trace showing the sequential handoff between:
- **Agent 1 (Enhanced Retriever)**: Entity extraction and hybrid retrieval.
- **Agent 2 (Enhanced Response Generator)**: Evidence synthesis and draft generation.
- **Agent 3 (Verification Node)**: Fact-checking and cleanup.

### 2. Tool Tracing
The `ToolRegistry` is instrumented to capture:
- **`hybrid_retriever`**: Exact query parameters and number of documents found.
- **`web_search`**: External API calls to Brave Search.
- **`reranker`**: Re-ordering logic and scores.

### 3. Model Tracing
All model interactions are captured:
- **Local Embeddings**: Qwen3-VL vector generation spans.
- **Local LLMs**: Qwen2.5 extraction and verification prompts.
- **API LLMs**: DeepSeek/OpenAI reasoning and synthesis outputs.

---

## Using Phoenix for Evaluation

ARK integrates Phoenix trace IDs with **RAGAS evaluation**. When tracing is enabled, each query result includes a `phoenix_trace_id` which allows you to:
1. Run an automated RAGAS eval (e.g., faithfulness).
2. Find the exact trace in the Phoenix UI for any query that received a low score.
3. Inspect the retrieved context to understand why the agent halluncinated.

---

## Advanced Configuration

### Custom Project Name
The default project name is `agentic-research-kit`. You can modify this in `src/agents/workflow.py`.

### Collector Endpoint
By default, traces are sent to `http://localhost:6006/v1/traces`. You can override this using:
```bash
export PHOENIX_COLLECTOR_ENDPOINT="http://your-remote-server:6006/v1/traces"
```

---

**Status**: ✅ Phoenix observability fully integrated into the 3-agent pipeline.

**Last Updated**: 2026-02-25
