# vLLM Integration & API-First Design

This document details the architectural transition of the Agentic Research Kit (ARK) to an API-first local inference model, specifically how **vLLM** has been integrated to serve the Qwen3.5-4B reasoning engine, and provides instructions for verifying its correctness.

## 1. Analysis of vLLM Integration

### Previous Architecture (Tight Coupling)
Previously, ARK loaded HuggingFace models (like `Qwen2TextLLM` or custom unified wrappers) directly within the Python application process using the `transformers` library. This tightly coupled the application logic with heavy model inference, leading to:
- Suboptimal hardware utilization (lack of continuous batching or PagedAttention).
- Memory fragmentation when running complex LangGraph workflows alongside model weights in VRAM.

### The API-First vLLM Design
To scale from local 12GB VRAM setups to HPC or external cloud APIs without altering application code, ARK decoupled model inference by integrating **vLLM**.

#### How it works:
1. **Dockerized Inference Server:**
   The `vllm` service is defined in `docker-compose.yml`. It uses the official `vllm/vllm-openai:v0.17.0` image to run the `Qwen/Qwen3.5-4B` model as a standalone process.
   - It exposes an OpenAI-compatible REST API on port `8001` (`http://localhost:8001/v1`).
   - It utilizes arguments like `--gpu-memory-utilization 0.7` and `--max-model-len 2048` to optimize VRAM and ensure compatibility with local hardware alongside TEI (Text Embeddings Inference) containers.

2. **Application Routing via LangChain (`src/agents/model_selector.py`):**
   Instead of custom wrapper classes, the application now uses a standard LangChain `ChatOpenAI` client pointing to the local vLLM endpoint:
   ```python
   self._local_llm = ChatOpenAI(
       model="Qwen/Qwen3.5-4B",
       base_url="http://localhost:8001/v1",
       api_key="EMPTY",  # vLLM does not require a real key by default
       temperature=0.0,
       timeout=120.0,
   )
   ```
   Before returning the client, the `ModelSelector` performs a fast-fail liveness check by pinging `http://localhost:8001/v1/models`. If the vLLM server is unreachable, the system automatically falls back to other configured providers (like DeepSeek or OpenAI).

#### Benefits:
- **Zero Application Overhead:** ARK no longer manages PyTorch tensors directly, significantly reducing memory spikes.
- **Native Efficiency:** vLLM natively parses `<thought>` blocks and handles continuous batching, doubling hardware utilization.
- **Uniformity:** Local inference now perfectly mimics external API interactions, meaning the system switches between local Qwen3.5, remote DeepSeek, or OpenAI seamlessly.

---

## 2. Verifying vLLM Correctness

To ensure the vLLM inference server is properly integrated and operational within ARK, perform the following verifications:

### Step 1: Verify the Docker Container is Running
Ensure that the vLLM service has started successfully and is not trapped in an initialization loop or OOM failure.
```bash
docker ps | grep ark-vllm
# To view the logs and ensure the Uvicorn server started on port 8000 (mapped to 8001 on host):
docker logs ark-vllm
```

### Step 2: API Liveness Check
Since vLLM exposes an OpenAI-compatible endpoint, query the `/v1/models` route to confirm it correctly loaded `Qwen/Qwen3.5-4B`.
```bash
curl -s http://localhost:8001/v1/models | jq
```
*Expected Output:* A JSON object listing the `Qwen/Qwen3.5-4B` model.

### Step 3: Test Inference via cURL
Manually trigger an inference request to verify the model responds accurately without hanging.
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
      "model": "Qwen/Qwen3.5-4B",
      "messages": [{"role": "user", "content": "Explain the concept of RAG briefly."}],
      "temperature": 0.0,
      "max_tokens": 100
  }'
```
*Expected Output:* A structured JSON response containing the model's completion.

### Step 4: Verify ARK Integration (CLI)
Finally, test that the ARK application correctly routes to the local vLLM model through the `ModelSelector`.
Ensure your `.env` is configured to use the local provider (`LLM_PROVIDER=local`) or specify it via the CLI:
```bash
uv run ark query "What are the latest advancements in quantum computing?" --mode local
```
**Check the Logs:**
Look for the following entries in your console or application logs (`logs/ark.log`):
1. `Initializing local Qwen3.5-4B model via vLLM...`
2. `✓ Local Qwen3.5-4B model initialized via vLLM`

If these logs appear and a valid research response is generated, the vLLM integration is operating correctly.
