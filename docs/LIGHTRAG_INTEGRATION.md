# LightRAG Integration: Current State & Architecture

## 📊 Executive Summary

This project **uses LightRAG** for document ingestion and storage with **hybrid retrieval capabilities** combining:
- ✅ **Vector similarity search** (semantic matching via Qwen3-VL-Embedding-2B)
- ✅ **BM25 keyword search** (lexical matching)
- ✅ **Knowledge graph traversal** (entity-based retrieval)

The storage backend has been migrated from JSON-based files to **LanceDB** for high performance and scalability.

**Current Status**:
- 🟢 **Ingestion**: Fully operational (backed by LanceDB).
- 🟢 **Retrieval**: **Full local hybrid retrieval** fully functional via thread-based isolation (**zero API costs**, no OpenAI key required).

---

## 🏗️ Architecture Overview

### 3-Agent LangGraph Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 1: Enhanced Retriever                             │
├─────────────────────────────────────────────────────────┤
│ • Detect query type (text/multimodal)                   │
│ • Extract entities (people, orgs, locations)            │
│ • Generate query embedding using Qwen3-VL               │
│ • Hybrid Retrieval (Vector + BM25 + KG)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 2: Enhanced Response Generator                    │
├─────────────────────────────────────────────────────────┤
│ • Rerank top 50 → top 10 documents                      │
│ • Synthesize evidence summary                           │
│ • Generate citation-rich draft response                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 3: Verification Node                              │
├─────────────────────────────────────────────────────────┤
│ • Fact-check response against retrieved sources         │
│ • Correct hallucinations and remove ungrounded claims   │
│ • Ensure 100% citation integrity                        │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Data Storage Architecture (LanceDB)

### LightRAG Storage Components

The system uses **LanceDB** as the primary storage engine, located in `rag_storage/lancedb_store/`.

| Component | Implementation | Description |
|-----------|----------------|-------------|
| **KV Storage** | `LanceDBKVStorage` | Stores full documents and extracted entities in binary format. |
| **Vector DB** | `LanceDBVectorDBStorage` | High-performance vector search for 2048D embeddings. |
| **Doc Status** | `LanceDBDocStatusStorage` | Tracks ingestion progress and file metadata. |
| **Graph DB** | `NetworkXStorage` | Knowledge graph representation (Nodes/Edges). |

---

## 🔍 Hybrid Retrieval Mechanisms

### 1. Vector Similarity Search (LanceDB)

**Model**: Qwen/Qwen3-VL-Embedding-2B (local HuggingFace model)

**Process**:
- Query is embedded into a 2048D vector.
- LanceDB performs an optimized cosine similarity search against stored chunk embeddings.
- Results are filtered by a configurable threshold.

**Benefits**:
- ✅ **Scalable**: Handles millions of vectors with sub-millisecond latency.
- ✅ **Multimodal**: Native support for image-to-text and text-to-image retrieval.

### 2. BM25 Keyword Search

**Algorithm**: Integrated into retrieval logic for lexical matching.

**Benefits**:
- ✅ Exact term matching for technical jargon and IDs.
- ✅ Complements semantic search where precision is required.

### 3. Knowledge Graph Traversal

**Storage**: NetworkX graph with entities and relationships.

**Benefits**:
- ✅ **Multi-hop reasoning**: Discovers indirect relationships between entities.
- ✅ **Context Expansion**: Pulls in relevant metadata from related nodes.

---

## 🔒 Thread-Isolated Hybrid Retrieval (Zero API Costs)

### The Challenge
LightRAG's async operations often conflict with LangGraph's event loop. Additionally, local GPU model inference must be managed to avoid blocking the main thread.

### The Solution: `IsolatedLightRAG`
We use a thread-based isolation layer that allows LightRAG to run its own event loop and local models in a separate thread.

**Key Components**:
1. **`DirectLightRAGRetriever`**: Configures LightRAG with local Qwen3-VL embedding and extraction functions.
2. **`IsolatedLightRAG`**: Manages a dedicated `asyncio` event loop for LightRAG operations in a background thread.
3. **`HybridRetrieverTool`**: Provides a clean, synchronous interface for the agent to trigger complex async retrieval.

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `src/agents/lancedb_storage.py` | LanceDB backend implementation |
| `src/agents/isolated_lightrag.py` | Thread isolation logic |
| `src/agents/direct_lightrag_retriever.py` | Local model configuration |
| `src/agents/tools/rag_tools/hybrid_retriever.py` | Unified retrieval tool |
| `src/utils/vision_embedding.py` | Local model implementations |

---

**Last Updated**: 2026-02-25 (Updated for LanceDB migration and 3-agent architecture)
