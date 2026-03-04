# Qwen3.5 Capability Research and Architecture Evaluation

This document provides a rigorous, independent evaluation of the Qwen3.5 model family and an architectural assessment of the Agentic Research Kit (ARK) for scaling from local 12GB VRAM setups to HPC/Cloud environments and external APIs like DeepSeek.

## 1. Qwen3.5 Capabilities vs. Current Local Models

### The Current Local Setup
Currently, the codebase relies on singletons in `src/utils/vision_embedding.py` to manage several models within a 12GB VRAM constraint:
*   **Vision/Text:** `Qwen/Qwen3-VL-2B-Instruct` or `Qwen/Qwen2.5-VL-3B-Instruct`
*   **Text Entity Extraction:** `Qwen/Qwen2.5-1.5B-Instruct` or `Phi-3.5-mini-instruct`
*   **Embeddings:** `Qwen/Qwen3-VL-Embedding-2B` (~4GB, unquantized)
*   **Reranker:** `Qwen/Qwen3-VL-Reranker-2B` (~2GB)

### Qwen3.5 Analysis
Contrary to previous generation models where Vision (VL) and Text were separated, the **Qwen3.5 series** (ranging from 0.8B to 397B parameters) features a natively unified Image-Text-to-Text architecture.

*   **Embedding and Reranker Status:** Hugging Face currently only offers the `Qwen3-VL` and `Qwen3` series for embeddings and rerankers. **There are no specific "Qwen3.5-VL-Embedding" or Reranker models yet.** The existing Qwen3-VL models remain state-of-the-art for these specific tasks.
*   **Unified Reasoning Engine:** Because Qwen3.5 instruct models inherently process both text and images, you do not need separate models for vision reasoning and text entity extraction. 

### Recommendation for 12GB VRAM Upgrade
Consolidate the reasoning models. By replacing both the vision model and the text entity extractor with a single **`Qwen/Qwen3.5-4B-Instruct`**, you can leverage a much stronger reasoning engine while staying within the 12GB limit.

**VRAM Math (8-bit Quantization):**
*   `Qwen3-VL-Embedding-2B`: ~4.0 GB (unquantized)
*   `Qwen3-VL-Reranker-2B`: ~2.0 GB
*   `Qwen3.5-4B-Instruct` (8-bit): ~4.5 GB
*   **Total VRAM:** ~10.5 GB (Leaving 1.5GB overhead for context windows and LangGraph overhead).

---

## 2. Architecture Evaluation: Local vs. Cloud / HPC / API

### Current Architectural Bottlenecks
The current implementation in `src/utils/vision_embedding.py` tightly couples the HuggingFace `transformers` library to the application layer. Models are loaded as PyTorch singletons within the same Python process. 

While `src/agents/model_selector.py` elegantly handles external APIs (like DeepSeek and OpenAI) via LangChain's `ChatOpenAI` client, local models are handled via a custom `Qwen2LangChainWrapper`. This approach has significant drawbacks when scaling:
1.  **Poor Hardware Utilization:** PyTorch's native `generate()` function does not utilize continuous batching, PagedAttention, or optimized CUDA kernels natively, leading to low throughput on HPC setups.
2.  **Memory Fragmentation:** Keeping model weights and RAG application logic in the same VRAM memory space often leads to Out-Of-Memory (OOM) errors during long context retrieval.

### Proposed Target Architecture
To get the absolute maximum out of available hardware—whether that is a local 12GB GPU, a multi-GPU HPC cluster, or external APIs like DeepSeek—the system should **decouple model inference from application logic**.

1.  **Local Inference Server (vLLM or llama.cpp):**
    Instead of loading HuggingFace models directly in Python via `transformers`, run a highly optimized inference engine like **vLLM** (or `llama.cpp` for local quantized models) alongside the ARK application. 
    *   vLLM provides an OpenAI-compatible API endpoint (e.g., `http://localhost:8000/v1`).
    *   This instantly unlocks continuous batching and PagedAttention, doubling or tripling hardware utilization on cloud/HPC.

2.  **Unified Client Interface:**
    Remove `Qwen2LangChainWrapper` and the complex singleton management in `vision_embedding.py`. 
    Configure `src/agents/model_selector.py` so that **all** interactions go through `ChatOpenAI`. 
    *   For **DeepSeek**: Use DeepSeek API keys and endpoints.
    *   For **Local Qwen3.5**: Point the `ChatOpenAI` client to the local vLLM endpoint.
    
3.  **Dedicated Embedding & Reranker Microservices:**
    For the remaining models (`Qwen3-VL-Embedding-2B` and `Qwen3-VL-Reranker-2B`), utilize a dedicated embedding microservice like Text Embeddings Inference (TEI) or Infinity. This completely removes heavy PyTorch dependencies from the LangGraph agents.

### Conclusion
By adopting an **API-first architecture** internally, ARK can seamlessly transition from a local 12GB laptop to a cloud deployment with DeepSeek or a dedicated HPC cluster running massive Qwen3.5-35B MoE models, all without changing a single line of LangGraph or RAG agent code.

---

## 3. Implementation Plan

This is a pragmatic, phased approach to separate the "Model Risk" (Qwen3.5 capability) from the "Architecture Risk" (vLLM decoupling).

### Phase 1: Model Consolidation (Tightly Coupled)
**Goal:** Prove Qwen3.5-4B can handle both Vision and Text tasks within the 12GB VRAM limit using the existing architecture.

1. **Create `UnifiedQwen35` Class:** 
   In `src/utils/vision_embedding.py`, create a new class using `AutoModelForImageTextToText`. Because Qwen3.5 natively handles both modalities, this single class will expose two methods:
   - `analyze_image(image, prompt)`
   - `generate_text(prompt)`
2. **Update Singletons & Env:** 
   Update `.env.defaults` to point `VISION_MODEL` and `ENTITY_MODEL` to `Qwen/Qwen3.5-4B-Instruct`. Update the singleton getters in `vision_embedding.py` to return the new unified class if requested.
3. **8-bit Quantization:** 
   Ensure `BitsAndBytesConfig` (8-bit) is applied to keep the 4B model at ~4.5GB VRAM.
4. **Validation:** 
   Run RAGAS evaluation and Pytest suites to confirm the model works and memory holds steady under 10.5GB total VRAM.

### Phase 2: Inference Decoupling (API-First Architecture)
**Goal:** Prepare for Cloud/HPC scaling and external APIs by removing heavy PyTorch management from the application.

1. **Spin up Local vLLM/llama.cpp Server:**
   Run an OpenAI-compatible inference server locally (e.g., `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B-Instruct --quantization bitsandbytes`).
2. **Refactor `model_selector.py`:**
   Modify your local LLM routing to stop instantiating `Qwen2LangChainWrapper`. Instead, configure it to return a standard LangChain `ChatOpenAI` client pointed at `http://localhost:8000/v1`.
3. **Strip out Transformers:**
   Once stable, systematically delete `Qwen2TextLLM`, `UnifiedQwen35`, and the custom LangChain wrappers. The only HuggingFace code remaining in the application should be for your Embedding and Reranker models (until you eventually move those to a microservice like TEI).