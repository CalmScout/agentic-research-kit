# Thread-Based LightRAG Isolation

## Overview

LightRAG hybrid retrieval is enabled through **thread-based isolation**, avoiding async context conflicts with LangGraph's event loop while maintaining zero API costs and full local execution.

## Problem Solved

**The Conflict**:
```
LangGraph workflow.ainvoke() (LangGraph's event loop)
  └─> LightRAG.aquery()
      └─> Internal Workers (initialized in main event loop)
          └─> CONFLICT: Two async systems managing same event loop, 
                        cross-loop object access leading to timeouts or 'Event loop is closed'
```

**The Solution**:
- **Isolated Thread**: Run LightRAG in a separate thread with its own dedicated event loop.
- **Lazy Thread-Local Initialization**: Initialize the LightRAG instance *inside* the isolated thread using a factory function. This ensures all internal workers and async objects are bound to the same event loop.
- **Structured Data Retrieval**: Use `aquery_data()` instead of `aquery()` to get raw, structured results (chunks, entities) instead of a pre-formatted LLM response string.
- **Synchronous Interface**: Provide a simple synchronous wrapper (`aquery_sync`) for easy integration with any workflow.

## Architecture

```
LangGraph Event Loop (Main Thread)
    ↓
HybridRetrieverTool.execute() (await)
    ↓
IsolatedLightRAG.aquery_sync() (calls factory on first use in thread)
    ↓
ThreadPoolExecutor → Worker Thread
    ↓
Thread-Local Event Loop & LightRAG Instance (Isolated)
    ↓
LightRAG.aquery_data() (Hybrid: Vector + BM25 + KG)
    ↓
Return structured Dict via thread boundary
```

## Key Components

### 1. IsolatedLightRAG (`src/agents/isolated_lightrag.py`)

The core isolation layer that manages thread-local resources.

**Key Features**:
- `rag_factory`: A callable that creates the LightRAG instance. It's executed within the worker thread.
- `_get_thread_resources()`: Manages thread-local storage for both the `asyncio` loop and the `LightRAG` instance.
- `aquery_sync()`: Handles the switch between `aquery_data` (for context) and `aquery` (for answers).

### 2. HybridRetrieverTool (`src/agents/tools/rag_tools/hybrid_retriever.py`)

The high-level tool used by the LangGraph agents.

**Features**:
- Lazy initialization of the isolation layer.
- Automatic fallback to keyword search (`simple_retriever`) if LightRAG fails.
- Parses `aquery_data` response into a consistent document format.

### 3. DirectLightRAGRetriever (`src/agents/direct_lightrag_retriever.py`)

Provides the configuration and factory for LightRAG using local models.

**Configuration**:
- **Embeddings**: `Qwen/Qwen3-VL-Embedding-2B` (local, 2048D)
- **LLM (Extraction)**: `Qwen/Qwen2.5-1.5B-Instruct` (local)
- **Storage**: **LanceDB** (High-performance KV & Vector) + NetworkX (Graph)

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Query Latency** | 2-8s | Depends on KG complexity |
| **VRAM Usage** | ~4-6GB | Shared with main embedding model |
| **API Costs** | $0 | Fully local models |
| **Reliability** | High | Thread isolation prevents loop crashes |

## Troubleshooting

### Issue: "LightRAG operation timed out"
**Symptoms**: `TimeoutError` after 60s.
**Cause**: Heavy GPU load or extremely large knowledge graph.
**Solution**: Increase `timeout` in `IsolatedLightRAG` initialization.

### Issue: "Event loop is closed"
**Symptoms**: Occurs if the process is shutting down or if thread-local storage is corrupted.
**Solution**: The current implementation re-creates the loop if it's closed. Restart the process if persistent.

---

**Last Updated**: 2026-02-25 (Updated for LanceDB integration)
**Status**: ✅ Production Ready (Isolated hybrid retrieval active)
