# Qwen3.5 Capability Research and Architecture Evaluation

This document provides a rigorous, independent evaluation of the Qwen3.5 model family and an architectural assessment of the Agentic Research Kit (ARK) for scaling from local 12GB VRAM setups to HPC/Cloud environments and external APIs like DeepSeek.

## 1. Qwen3.5 Capabilities vs. Current Local Models

### The Current Local Setup
The codebase utilizes singletons in `src/utils/vision_embedding.py` to manage several models within a 12GB VRAM constraint:
*   **Unified Reasoning Engine (Vision/Text):** `Qwen/Qwen3.5-4B` (8-bit quantized)
*   **Fast Entity Extraction:** `Qwen/Qwen2.5-1.5B-Instruct`
*   **Embeddings:** `Qwen/Qwen3-VL-Embedding-2B` (~4GB, unquantized)
*   **Reranker:** `Qwen/Qwen3-VL-Reranker-2B` (~2GB)

### Qwen3.5 Analysis
Contrary to previous generation models where Vision (VL) and Text were separated, the **Qwen3.5 series** (ranging from 0.8B to 397B parameters) features a natively unified Image-Text-to-Text architecture.

*   **Embedding and Reranker Status:** Hugging Face currently only offers the `Qwen3-VL` and `Qwen3` series for embeddings and rerankers. **There are no specific "Qwen3.5-VL-Embedding" or Reranker models yet.** The existing Qwen3-VL models remain state-of-the-art for these specific tasks.
*   **Unified Reasoning Engine:** Because Qwen3.5 instruct models inherently process both text and images, the system has been consolidated to use `Qwen3.5-4B` for vision reasoning, fact-checking (Agent 3), and complex reranking (Agent 2).

### 12GB VRAM Optimization (Implemented)
We have consolidated the reasoning models. By replacing both the separate vision model and the text reasoning engine with a single **`Qwen/Qwen3.5-4B`**, we leverage a much stronger reasoning engine while staying strictly within the 12GB limit.

**VRAM Math (8-bit Quantization):**
*   `Qwen3-VL-Embedding-2B`: ~4.0 GB (unquantized)
*   `Qwen3-VL-Reranker-2B`: ~2.0 GB
*   `Qwen3.5-4B` (8-bit): ~4.5 GB
*   **Total VRAM:** ~10.5 GB (Leaving 1.5GB overhead for context windows and LangGraph state).

---

## 2. Architecture Evaluation: Local vs. Cloud / HPC / API

### Current Architectural Bottlenecks
The current implementation in `src/utils/vision_embedding.py` tightly couples the HuggingFace `transformers` library to the application layer. Models are loaded as PyTorch singletons within the same Python process. 

While `src/agents/model_selector.py` handles external APIs (like DeepSeek and OpenAI) via LangChain's `ChatOpenAI` client, local models are handled via a custom `Qwen2LangChainWrapper`. This approach has drawbacks when scaling:
1.  **Poor Hardware Utilization:** PyTorch's native `generate()` function does not utilize continuous batching or PagedAttention.
2.  **Memory Fragmentation:** Keeping model weights and RAG application logic in the same VRAM space can lead to OOM errors during long context retrieval.

### Proposed Target Architecture (Phase 2 Focus)
To get the absolute maximum out of available hardware—whether that is a local 12GB GPU, a multi-GPU HPC cluster, or external APIs like DeepSeek—the system should **decouple model inference from application logic**.

1.  **Local Inference Server (vLLM):**
    Instead of loading HuggingFace models directly in Python via `transformers`, run a highly optimized inference engine like **vLLM** alongside the ARK application. 
    *   vLLM provides an OpenAI-compatible API endpoint (e.g., `http://localhost:8000/v1`).
    *   This unlocks continuous batching and PagedAttention, doubling or tripling hardware utilization.

2.  **Unified Client Interface:**
    Refactor `src/agents/model_selector.py` so that **all** interactions go through `ChatOpenAI`. 
    *   For **DeepSeek**: Use DeepSeek API keys and endpoints.
    *   For **Local Qwen3.5**: Point the `ChatOpenAI` client to the local vLLM endpoint.

3.  **Dedicated Embedding Microservices:**
    For the remaining models (`Qwen3-VL-Embedding-2B`), utilize a dedicated embedding microservice like Text Embeddings Inference (TEI). This completely removes heavy PyTorch dependencies from the LangGraph agents.

### Conclusion
By adopting an **API-first architecture** internally, ARK can seamlessly transition from a local 12GB laptop to a cloud deployment with DeepSeek or a dedicated HPC cluster running massive Qwen3.5-35B MoE models, all without changing a single line of LangGraph or RAG agent code.

---

## 3. Implementation Plan

### Phase 1: Model Consolidation (Complete)
**Goal:** Prove Qwen3.5-4B can handle both Vision and Text tasks within the 12GB VRAM limit.

1. [x] **Create `UnifiedQwen35` Class:** 
   Implemented in `src/utils/vision_embedding.py` using `AutoModelForCausalLM` (Qwen3.5 architecture).
2. [x] **Update Singletons & Env:** 
   Updated singleton getters in `vision_embedding.py` to return the unified class.
3. [x] **8-bit Quantization:** 
   Integrated `BitsAndBytesConfig` (8-bit) to keep the 4B model at ~4.5GB VRAM.
4. [x] **Validation:** 
   Confirmed via RAGAS evaluation and Pytest suites that the model works within memory constraints.

### Phase 2: Inference Decoupling (Immediate Focus)
**Goal:** Prepare for Cloud/HPC scaling and external APIs by removing heavy PyTorch management from the application.

1. [ ] **Spin up Local vLLM Server:**
   Provide documentation for running an OpenAI-compatible inference server locally (e.g., vLLM).
2. [ ] **Refactor `model_selector.py`:**
   Modify local LLM routing to use a standard LangChain `ChatOpenAI` client pointed at the local vLLM endpoint instead of `Qwen2LangChainWrapper`.
3. [ ] **Strip out Transformers:**
   Systematically remove `Qwen2TextLLM`, `UnifiedQwen35`, and custom wrappers once the API transition is stable.

---

**Last Updated**: 2026-03-04 (Updated to reflect Phase 1 completion and Qwen3.5-4B integration)