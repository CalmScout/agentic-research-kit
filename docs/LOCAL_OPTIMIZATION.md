# Local GPU Optimization Guide (12GB VRAM)

The Agentic Research Kit (ARK) is optimized to run a full 3-agent research loop on consumer hardware with 12GB of VRAM. This document explains how to get the best performance out of your local setup.

## 1. Expected Performance Profile
On a single 12GB GPU (e.g., RTX 3060/4070):
- **Startup**: ~60 seconds (LanceDB initialization).
- **Per Iteration**: ~2.5 minutes (Retrieval + Reasoning + Verification).
- **Total Research**: 3 to 8 minutes depending on the complexity and number of ReAct iterations.

## 2. Key Optimizations Implemented
- **API-First Design**: Decoupled inference via **vLLM** and **TEI** to prevent Python memory fragmentation.
- **Thinking Process Stripper**: Automatically cleans reasoning blocks to prevent JSON parsing errors in the RAG engine.
- **Serial Processing**: Forced `llm_model_max_async=1` to prevent GPU context switching and OOMs.
- **Paranoid Filtering**: Blacklists logical connectors (e.g., "however", "usually") from triggering wasteful background research tasks.

## 3. How to Speed Up Queries

### Use the API Server instead of CLI
The CLI re-initializes the entire database and model connections on every query. Using the API server keeps the system "warm."
```bash
# Start the server
uv run ark serve

# Use a client or curl to query
curl -X POST http://localhost:8000/query -d '{"query": "Your question", "mode": "local"}'
```

### Adjust Verification Strictness
If the system is looping too many times (3 iterations), you can reduce the strictness in `src/agents/verification.py` or lower the `MAX_ITERATIONS` in `src/agents/workflow.py`.

### Use "Naive" Mode for Quick Answers
If you don't need the Knowledge Graph traversal, use `--mode naive` to perform simple vector search, which skips the complex LightRAG entity extraction phase.

## 4. Troubleshooting Slowness
- **Check Docker Logs**: `docker logs ark-vllm` to ensure the model isn't hitting KV cache limits.
- **Monitor VRAM**: Use `nvidia-smi`. You should see vLLM taking ~4.5GB and TEI taking ~4GB. If usage hits 11.5GB+, the system will slow down due to memory swapping.
- **JSON Errors**: If you see `No JSON-like structure found`, the model is rambling. The `ThinkingProcessStripper` and `json_repair` handle this, but it still adds latency.
