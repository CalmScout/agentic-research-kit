# LightRAG Integration: Current State & Architecture

## 📊 Executive Summary

This project **uses LightRAG** for document ingestion and storage with **hybrid retrieval capabilities** combining:
- ✅ **Vector similarity search** (semantic matching via Qwen3-VL-Embedding-2B)
- ✅ **BM25 keyword search** (lexical matching)
- ✅ **Knowledge graph traversal** (entity-based retrieval)

**Current Status**:
- 🟢 **Ingestion**: Fully operational (2485 documents processed)
- 🟢 **Retrieval**: **Full local hybrid retrieval** fully functional via thread-based isolation (**zero API costs**, no OpenAI key required).

---

## 🏗️ Architecture Overview

### 2-Agent LangGraph Pipeline

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
│ • Generate final response with citations                │
│ • Add confidence scoring                                │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Data Storage Architecture

### LightRAG Storage Components

```
./rag_storage/
├── vdb_chunks.json              # Vector embeddings (2048D)
│   └── NanoVectorDBStorage
│       ├── 2485 chunk embeddings
│       └── Cosine similarity search
│
├── graph_chunk_entity_relation.graphml  # Knowledge graph
│   └── NetworkXStorage
│       ├── 12 entities
│       ├── 6 relationships
│       └── Multi-hop traversal
│
├── kv_store_full_docs.json       # Document store
│   └── JsonKVStorage
│       └── 2485 complete documents
│
├── kv_store_text_chunks.json     # Text chunks
│   └── JsonKVStorage
│       └── Document segments for retrieval
│
├── full_entities.json            # Entity store
│   └── JsonKVStorage
│       └── Extracted entities (people, orgs, locations)
│
└── full_relations.json           # Relationship store
    └── JsonKVStorage
        └── Entity relationships
```

### Storage Backend Configuration

| Storage Type | Backend | Purpose | Status |
|--------------|---------|---------|--------|
| **KV Store** | JsonKVStorage | Key-value data (docs, entities) | ✅ Active |
| **Vector DB** | NanoVectorDBStorage | Embeddings (2048D) | ✅ Active |
| **Graph DB** | NetworkXStorage | Knowledge graph | ✅ Active |
| **Doc Status** | JsonDocStatusStorage | Processing status | ✅ Active |

---

## 🔍 Hybrid Retrieval Mechanisms

### 1. Vector Similarity Search

**Model**: Qwen/Qwen3-VL-Embedding-2B (local HuggingFace model)

**Process**:
```python
# Query embedding
query_embedding = Qwen3VLEmbedding.embed_text("climate change")
# → Returns: 2048-dimensional vector

# Similarity search
similar_docs = cosine_similarity(query_embedding, stored_embeddings)
# → Finds: Semantically similar documents
```

**Benefits**:
- ✅ Understands meaning, not just keywords
- ✅ Finds related concepts (e.g., "global warming" → "climate change")
- ✅ Multimodal support (text + images in same vector space)

---

### 2. BM25 Keyword Search

**Algorithm**: Built into LightRAG (TF-IDF style scoring)

**Process**:
```python
# Token matching
score = IDF(term) × (TF(term in doc) × (k1 + 1)) / (TF + k1)

# Ranks documents by keyword overlap
results = rank_by_bm25(query_terms, documents)
```

**Benefits**:
- ✅ Exact term matching
- ✅ Fast retrieval
- ✅ Good for specific queries (names, IDs, technical terms)

---

### 3. Knowledge Graph Traversal

**Storage**: NetworkX graph with entities and relationships

**Process**:
```python
# Entity extraction
entities = ["Federal Reserve", "interest rates", "inflation"]

# Graph traversal
path = graph.find_path("Federal Reserve", "stock market")
# → Federal Reserve → interest rates → inflation → stock market

# Multi-hop retrieval
docs = get_documents_along_path(path)
```

**Benefits**:
- ✅ Multi-hop reasoning
- ✅ Discovers indirect relationships
- ✅ Entity-based context expansion

---

## 🔒 Thread-Isolated Hybrid Retrieval (Zero API Costs)

### The Challenge
LightRAG's async operations often conflict with LangGraph's event loop, leading to `NoneType` errors and context manager failures. Additionally, semantic retrieval typically requires an external API key (OpenAI/DeepSeek).

### The Solution: `IsolatedLightRAG`
We implemented a thread-based isolation layer that allows LightRAG to run its own event loop and local models in a separate thread.

**Key Components**:
1. **`DirectLightRAGRetriever`**: Configures LightRAG with local Qwen3-VL embedding and LLM functions.
2. **`IsolatedLightRAG`**: Manages a `ThreadPoolExecutor` and a dedicated `asyncio` event loop for LightRAG operations.
3. **`HybridRetrieverTool`**: A tool-registry compatible wrapper that provides a simple sync interface for complex async hybrid retrieval.

**Benefits**:
- ✅ **100% Local**: No data leaves your machine.
- ✅ **Zero Cost**: No OpenAI/DeepSeek embedding API costs.
- ✅ **Full Power**: Uses Vector + BM25 + Knowledge Graph simultaneously.
- ✅ **Async Safe**: Works perfectly within LangGraph workflows.

---

## 🖼️ Multimodal Image Processing

### Cross-Modal Retrieval
All embeddings (text and image) are in the same 2048D vector space using Qwen3-VL-Embedding-2B.

| Query Type | Embedding Method | Can Retrieve |
|------------|------------------|--------------|
| **Text** | `embed_text(query)` | Text docs + Image captions |
| **Image** | `embed_image(image)` | Similar images + Related text |
| **Multimodal** | `embed_multimodal(text, image)` | Cross-modal results |

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `src/agents/isolated_lightrag.py` | Thread isolation logic |
| `src/agents/direct_lightrag_retriever.py` | Local model configuration |
| `src/agents/tools/rag_tools/hybrid_retriever.py` | Unified retrieval tool |
| `src/utils/vision_embedding.py` | Local model implementations |
| `rag_storage/` | Persistent LightRAG data |

---

## 🚀 Current Implementation Status

### ✅ Fully Functional (Local)
- **Hybrid Retrieval**: Vector + BM25 + Knowledge Graph (Local)
- **Multimodal Embeddings**: Qwen3-VL-Embedding-2B (Local)
- **Entity Extraction**: Qwen2.5-1.5B (Local)
- **Thread Isolation**: Async context safety

### 🔧 Configuration
The system automatically uses local hybrid retrieval by default. No OpenAI API key is required.

**Last Updated**: 2026-02-23 (Updated to reflect local thread-isolation success)
