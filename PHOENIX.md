# Phoenix Observability Setup Guide

## Overview

Phoenix (Arize AI) provides **open-source observability** for your multi-agent RAG system. It automatically traces all agent executions, retrievals, embeddings, and generations in real-time.

**Key Benefits**:
- 🔍 **Auto-instrumentation**: No code changes needed - traces LangGraph automatically
- 📊 **Visual Trace Explorer**: See all agents, embeddings, retrievals, generations in one view
- 🐛 **Debug Failures**: Identify exactly which agent failed and why
- 📈 **Performance Insights**: See latency per agent, token usage, costs
- 🎯 **Interview-Ready**: Impressive visualization of agent orchestration

---

## Quick Start

### 1. Install Dependencies

```bash
# Dependencies already added to pyproject.toml
uv sync
```

This installs:
- `arize-phoenix>=4.0.0` - Phoenix tracing server and client
- `openinference-instrumentation-langchain>=0.1.0` - LangChain/LangGraph instrumentation
- `opentelemetry-sdk>=1.22.0` - OpenTelemetry SDK
- `opentelemetry-api>=1.22.0` - OpenTelemetry API

### 2. Start Phoenix UI

**Option A: Background (Recommended for Development)**
```bash
# Start Phoenix in background
python -m phoenix.server.main serve &
# Or use nohup for persistent background
nohup python -m phoenix.server.main serve > phoenix.log 2>&1 &
```

**Option B: Foreground (For Testing)**
```bash
python -m phoenix.server.main serve
# Visit http://localhost:6006
```

Phoenix UI will be available at: **http://localhost:6006**

### 3. Enable Tracing

**Option A: Environment Variable (Recommended)**
```bash
export PHOENIX_ENABLED=true
uv run ark query "Is climate change caused by humans?"
```

**Option B: Inline**
```bash
PHOENIX_ENABLED=true uv run ark query "Is climate change caused by humans?"
```

**Option C: Add to `.env`**
```bash
echo "PHOENIX_ENABLED=true" >> .env
```

### 4. View Traces

1. Open http://localhost:6006
2. Click on "Traces" in the left sidebar
3. See your multi-agent workflow traced!

---

## What You'll See

### Trace View

Each query creates a **trace** showing:

```
Query: "Is climate change caused by humans?"
  ├─ Agent 1: Enhanced Retriever
  │   ├─ Entity Extraction (LLM call)
  │   ├─ Embedding Generation (Qwen3-VL model)
  │   └─ Hybrid Retrieval (LightRAG)
  └─ Agent 2: Enhanced Response Generator
      ├─ Reranking (Qwen3-VL-Reranker)
      ├─ Evidence Synthesis
      ├─ Response Generation (DeepSeek API)
      └─ Confidence Scoring
```

### Span Details

Click on any span to see:
- **Input**: Query, entities, embeddings
- **Output**: Retrieved docs, summaries, final response
- **Metadata**: Model used, latency, tokens, costs
- **Attributes**: Agent names, timestamps, error info

---

## Advanced Configuration

### Custom Collector Endpoint

By default, Phoenix sends traces to `http://localhost:6006/v1/traces`.

To use a remote Phoenix instance:

```bash
export PHOENIX_COLLECTOR_ENDPOINT="https://your-phoenix-instance.com/v1/traces"
export PHOENIX_ENABLED=true
uv run claim-rag query "Your query"
```

### Project Name

Traces are organized by **project name** (default: "multimodal-rag").

To change it, edit [src/agents/workflow.py](src/agents/workflow.py:58):

```python
phoenix_register(
    project_name="my-custom-project",  # Change this
    endpoint=collector_endpoint,
)
```

### Sampling Rate

To reduce tracing overhead, sample only a percentage of queries:

```python
from opentelemetry.sdk.trace import sampling

phoenix_register(
    project_name="multimodal-rag",
    endpoint=collector_endpoint,
    # Sample 50% of traces
    sampling_rate=0.5,
)
```

---

## Production Deployment

### Docker Compose

Add Phoenix to your [docker-compose.yml](docker-compose.yml):

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
    environment:
      - PHOENIX_PROJECT_NAME=multimodal-rag
    volumes:
      - ./phoenix_data:/tmp/phoenix
    restart: unless-stopped

  api:
    # ... your existing API service
    environment:
      - PHOENIX_ENABLED=true
      - PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces
    depends_on:
      - phoenix
```

Start with:
```bash
docker-compose up -d
```

### Remote Phoenix Server

For production, deploy Phoenix on a separate server:

1. **Deploy Phoenix** (e.g., on AWS EC2, GCP, Azure)
2. **Configure firewall** to allow port 6006 (or use reverse proxy)
3. **Set endpoint** in your application:
   ```bash
   export PHOENIX_COLLECTOR_ENDPOINT="https://phoenix.your-domain.com/v1/traces"
   ```

---

## Using Phoenix for Debugging

### Example 1: Retrieval Failed

**Symptom**: Agent 1 (Enhanced Retriever) returns 0 documents

**In Phoenix**:
1. Find trace in UI
2. Click "Agent 1: Enhanced Retriever" span
3. Check "Output" - see `retrieved_docs = []`
4. Check "Attributes" - see retrieval method used
5. **Debug**: Issue with LightRAG query or vector DB

**Solution**: Check [src/agents/enhanced_retriever.py](src/agents/enhanced_retriever.py)

### Example 2: Low Confidence

**Symptom**: Response confidence < 50%

**In Phoenix**:
1. Find trace
2. Click "Agent 2: Enhanced Response Generator" span
3. Check "Input" - see evidence_summary quality
4. Check "Agent 1" span - see retrieval quality
5. **Debug**: Poor retrieval or reranking

**Solution**: Improve retrieval in Agent 1 or reranking in Agent 2

### Example 3: High Latency

**Symptom**: Query takes 10+ seconds

**In Phoenix**:
1. Check trace timeline - see which agent is slow
2. Click slow agent span - see latency breakdown
3. Check "Attributes" - see model inference time vs. overhead
4. **Debug**: Slow LLM or inefficient code

**Solution**: Optimize slow agent or use faster model

---

## Demo Script for Interviews

### Setup (Before Demo)

```bash
# Terminal 1: Start Phoenix
python -m phoenix.server.main serve

# Terminal 2: Run queries
export PHOENIX_ENABLED=true
```

### Demo Flow

**Step 1: Show Phoenix UI**
```
"I use Phoenix for observability to trace all agent executions.
Let me show you the UI at http://localhost:6006"
```

**Step 2: Run Query**
```bash
uv run ark query "Is climate change caused by humans?"
```

**Step 3: Show Live Trace**
```
"Now let's view the trace in Phoenix.

See how the agents are orchestrated:
1. Agent 1 (Enhanced Retriever) extracted entities and retrieved 29 relevant documents
2. Agent 2 (Enhanced Response Generator) reranked results and generated a response with 85% confidence"
```

**Step 4: Explain Benefits**
```
"This helps me:
- Debug failures: See exactly which agent failed and why
- Optimize performance: Identify slow agents
- Track costs: Monitor token usage and API calls
- Ensure quality: Verify each agent's output"
```

**Step 5: Show Advanced Features**
```
"I can also:
- Compare traces across queries
- Filter by agent name, latency, error status
- Export traces for analysis
- Set alerts on failure rates"
```

---

## Troubleshooting

### Phoenix Not Starting

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 6006
lsof -i :6006
# Kill it
kill -9 <PID>
# Restart Phoenix
python -m phoenix.server.main serve
```

### No Traces Appearing

**Check 1**: Phoenix enabled?
```bash
echo $PHOENIX_ENABLED  # Should be "true"
```

**Check 2**: Dependencies installed?
```bash
uv pip list | grep phoenix
# Should see: arize-phoenix, openinference-instrumentation-langchain
```

**Check 3**: Correct endpoint?
```bash
echo $PHOENIX_COLLECTOR_ENDPOINT
# Should be: http://localhost:6006/v1/traces
```

**Check 4**: Phoenix server running?
```bash
curl http://localhost:6006/health
# Should return: {"status":"ok"}
```

### ImportError

**Error**: `No module named 'phoenix'`

**Solution**:
```bash
uv sync
# Or install manually:
uv add arize-phoenix openinference-instrumentation-langchain
```

---

## Architecture

### How Phoenix Works

```
Your Multi-Agent App
    ↓ (sends traces via OTLP)
Phoenix Collector (localhost:6006)
    ↓ (stores traces)
Phoenix UI
    ↓ (displays)
Your Browser
```

### OpenTelemetry Integration

Phoenix uses **OpenTelemetry (OTEL)** for vendor-agnostic tracing:

- **Protocol**: OTLP (OpenTelemetry Protocol)
- **Format**: OpenInference (standard for LLM traces)
- **Benefits**: Future-proof, not locked into Phoenix

You can export traces to other OTEL-compatible tools:
- Grafana
- Datadog
- Jaeger
- Lightstep

---

## Resources

### Official Documentation
- [Phoenix LangGraph Integration](https://arize.com/docs/phoenix/integrations/python/langgraph/langgraph-tracing)
- [Phoenix GitHub Repository](https://github.com/Arize-ai/phoenix)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

### Tutorials
- [Tracing LangGraph Agents](https://mohdzain.com/blogs/tracing-agentic-llm-workflows-arize-phoenix-langgraph)
- [Debugging Agent Loops](https://medium.com/@ap3617180/debugging-agent-loops-bridging-the-observability-gap-with-arize-phoenix-de78cb093496)
- [Building and Tracing Calculus Agent](https://danielegbo.medium.com/building-and-tracing-calculus-code-agent-with-langgraph-and-arize-phoenix-007146e11751)

### Videos
- [Phoenix Observability Demo](https://www.youtube.com/watch?v=your-video-link)

---

## FAQ

**Q: Does Phoenix slow down my system?**
A: Minimal overhead (<50ms per query). Use sampling to reduce further.

**Q: Can I use Phoenix without internet?**
A: Yes! Phoenix is self-hosted and works offline.

**Q: How much disk space does Phoenix use?**
A: ~1MB per 1000 traces. Configure retention in Phoenix settings.

**Q: Can I export traces?**
A: Yes! Export as JSON, CSV, or send to other OTEL tools.

**Q: Is Phoenix free?**
A: Yes! 100% open-source and free. No usage limits.

---

## Next Steps

1. ✅ Install dependencies (done in [pyproject.toml](pyproject.toml))
2. ✅ Add instrumentation (done in [workflow.py](src/agents/workflow.py))
3. ⏳ Start Phoenix UI: `python -m phoenix.server.main serve`
4. ⏳ Enable tracing: `export PHOENIX_ENABLED=true`
5. ⏳ Run query and view traces!
6. ⏳ Use traces for debugging and optimization

---

**Status**: ✅ Phoenix observability integrated!

**Commands**:
```bash
# Start Phoenix
python -m phoenix.server.main serve

# Run query with tracing
PHOENIX_ENABLED=true uv run ark query "Is climate change real?"

# View traces at
open http://localhost:6006
