# AirPI v2 — Stateful LLM Inference Server for Raspberry Pi 5

> Ollama-compatible inference server with persistent memory, knowledge graph, web UI, and CLI for a Raspberry Pi 5 with 8 GB RAM and NVMe storage.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-backed-444444)](https://github.com/ggerganov/llama.cpp)
[![FastAPI](https://img.shields.io/badge/FastAPI-00A393?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-99%2F99-28a745)](./tests/)
[![Ollama API](https://img.shields.io/badge/Ollama-API_compatible-0B5FFF)](https://github.com/ollama/ollama)

---

## What's New in v2

AirPI v2 adds **stateful inference** to the Pi 5:

| Component | Capability |
|-----------|-----------|
| **Memory System** | Persistent fact storage with deduplication and categorization |
| **Knowledge Graph** | Visual concept relationships extracted from conversations |
| **Web UI** | Markdown-rendering chat, system prompt editor, export, real-time stats |
| **CLI** | Full-featured command-line interface with multiline input and editor support |
| **Session Cache** | KV-cache reuse within and across requests (70–90% prefill savings) |
| **Speculative Decoding** | Optional draft-target acceleration |

### v2 vs v1

| Feature | v1 | v2 |
|---------|----|----|
| API server | ✅ | ✅ |
| Ollama compatibility | ✅ | ✅ |
| Session KV-cache | ✅ | ✅ |
| Web UI | ❌ | ✅ Chat + Graph |
| Memory persistence | ❌ | ✅ Searchable facts |
| CLI | ❌ | ✅ Full interactive |
| System prompt editor | ❌ | ✅ Persistent |

---

## Quick Start

### Prerequisites

- Raspberry Pi 5 with 8 GB RAM and NVMe SSD
- Python 3.11+
- systemd (for service mode)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/AirPI.git
cd AirPI

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Build & Install

```bash
# Build llama-cpp-python for ARM64 (required)
CMAKE_ARGS="-DLLAMA_NATIVE=on -DLLAMA_BLAS=OFF" \
  pip install llama-cpp-python --no-binary llama-cpp-python

# Install AirPI + dependencies
pip install -e .
```

### 3. Download Models

```bash
# Create models directory
sudo mkdir -p /data/models && sudo chown $USER /data/models

# Download Qwen2.5-Coder (recommended for Pi 5)
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --local-dir /data/models

huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir /data/models
```

### 4. Run

#### Local Development

```bash
uvicorn server:app --host 127.0.0.1 --port 11435
```

#### systemd Service

```bash
sudo cp systemd/airpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now airpi
sudo systemctl status airpi
```

### 5. Access AirPI

- **Web UI**: http://localhost:11435/ui/
- **Graph Visualizer**: http://localhost:11435/ui/graph.html
- **API Health**: http://localhost:11435/health

---

## Features

### 💬 Web Chat Interface

- **Markdown rendering** with syntax-highlighted code blocks (copy button on hover)
- **System prompt editor** — change model behavior per session (persisted in localStorage)
- **Chat export** — download conversation as `.md` or `.json`
- **Live stats bar** — model name, queue depth, cache hit rate, tokens/sec
- **No external CDN** — all libraries vendor'd locally

### 🧠 Persistent Memory

- **Automatic extraction** from chat responses
- **Deduplication** on content hash
- **Categorization** — fact, preference, correction, project, system, todo
- **Confidence scoring** (0–100, learned from recency and context)
- **Top-50 limiting** — memory.md stays handleable even after hundreds of facts
- **Fallback strategies** — graceful handling of malformed LLM extraction

### 📊 Knowledge Graph

- **Concept nodes** — extracted entities with category, source, age, confidence
- **Edges** — co-occurrence relationships between concepts
- **Interactive search** — filter and dim unmatched nodes
- **Backlinks panel** — click any node to see which facts reference it
- **SVG export** — save the graph for documentation
- **Live updates** ready (groundwork in Phase 3)

### ⌨️ CLI

Full interactive command-line interface:

```bash
# Start interactive chat
airpi chat

# Send system prompt
airpi chat --system "Du bist ein Python-Experte"

# Load from file
airpi chat --system-file /path/to/prompt.txt

# Machine-readable output
airpi chat --json

# Watch status with live polling
airpi status --watch

# Plain text for pipes
airpi status --plain

# Multiline input with backslash continuation
prompt> explain\
... tensorflow \
... architecture

# Open $EDITOR for long input
prompt> .edit
```

**Exit codes:**
- `0` — success
- `1` — server error
- `2` — timeout
- `3` — model error

### 🔄 Session-Based KV-Cache

Pass `session_id` to reuse KV-cache across requests:

```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "prompt": "Continue the story: Once upon a time...",
    "session_id": "story-session-001"
  }'
```

Cache hit rate appears in `/health` and stats bar.

---

## API Reference

### REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/live` | GET | Liveness check |
| `/ready` | GET | Readiness (models + queue) |
| `/health` | GET | Runtime state, uptime, cache stats, recovery info |
| `/metrics` | GET | Prometheus format metrics |
| `/api/tags` | GET | Ollama-compatible model list |
| `/api/generate` | POST | Ollama-compatible inference (streaming) |
| `/api/chat` | POST | Non-streaming chat with system prompt support |
| `/memory/store` | POST | Manually store a fact |
| `/memory/delete` | POST | Delete facts by keyword |
| `/memory` | GET | Retrieve current memory + all active entries |
| `/graph/data` | GET | Knowledge graph nodes and edges |
| `/graph` | GET | Redirect to web graph UI |
| `/ui/` | GET | Serve web UI (index.html) |
| `/ui/graph.html` | GET | Serve graph visualizer |

### Chat Request Example

```bash
curl -X POST http://localhost:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [
      {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
      {"role": "user", "content": "Erkläre mir rekursion."}
    ],
    "stream": false,
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AIRPI_MODELS_DIR` | `/data/models` | GGUF model directory |
| `AIRPI_DEFAULT_MODEL` | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Default model |
| `AIRPI_FAST_MODEL` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Low-latency fast lane |
| `AIRPI_LARGE_MODEL` | `qwen2.5-coder-7b-q4_k_m.gguf` | Large prompt handling |
| `AIRPI_N_THREADS` | `3` | CPU decode threads |
| `AIRPI_N_CTX_SMALL` | `2048` | Small model context window |
| `AIRPI_N_CTX_LARGE` | `2048` | Large model context window |
| `AIRPI_MAX_QUEUE` | `10` | Max queued requests |
| `AIRPI_KEEP_ALIVE_TIMEOUT` | `900` | Model idle timeout (seconds) |
| `AIRPI_HOST` | `127.0.0.1` | Bind address |
| `AIRPI_PORT` | `11435` | HTTP port |
| `AIRPI_LOG_LEVEL` | `info` | Log verbosity (debug, info, warning, error) |
| `MEMORY_DB_PATH` | `./memory.db` | SQLite memory database location |
| `MEMORY_MD_MAX_ENTRIES` | `50` | Max entries in memory.md |

See `.env.example` for all options.

---

## Performance

### Benchmarks (Qwen2.5-Coder family)

| Model | Size | RAM | tok/sec | First-token |
|-------|------|-----|---------|------------|
| 0.5B | Q4_K_M | 0.4 GB | 25–30 | 80 ms |
| 1.5B | Q4_K_M | 1.2 GB | 12–15 | 120 ms |
| 7B | Q4_K_M | 4.1 GB | 3–5 | 250 ms |
| 14B | Q4_K_M | 8.1 GB | 1–2* | 500 ms |

\* With mmap paging to NVMe; full RAM resident is faster but requires larger device.

### Cache Hit Impact

- **Cache miss** (first request in session): full prefill overhead
- **Cache hit** (subsequent requests): 70–90% reduction in prefill tokens
- Example: 2000-token prompt, cache hit saves ~1600 decode iterations

---

## Architecture

```
┌─────────────────────────────────────┐
│        User Facing (Web + CLI)      │
│  Chat UI • Graph • CLI              │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   FastAPI Server (server.py)        │
│  /api/generate • /api/chat          │
│  /memory • /graph • /ui             │
└───────────┬─────────────────────────┘
            │
     ┌──────┼──────────┐
     │      │          │
┌────▼──┐ ┌─▼──────┐ ┌─▼────────────┐
│ Model │ │ Memory │ │ Knowledge    │
│Manager│ │ System │ │ Graph        │
└──┬────┘ └────────┘ └──────────────┘
   │
┌──▼──────────────────────────────────┐
│   llama.cpp (C++ inference core)    │
│   KV-cache • Session management     │
└────────────────────────────────────┘
```

---

## Development

### Running Tests

```bash
pytest tests/ -v              # All tests
pytest tests/test_server.py   # Unit tests
pytest tests/test_integration.py -s  # Integration (requires live server)
```

### Project Structure

```
AirPI/
├── server.py                 # FastAPI app, endpoints
├── model_manager.py          # Model loading & cache
├── config.py                 # Configuration
├── cli/
│   └── main.py              # CLI implementation
├── memory/
│   ├── manager.py           # Persistence, deduplication
│   └── graph.py             # Concept extraction
├── ui/
│   ├── index.html           # Chat interface
│   ├── graph.html           # Graph visualizer
│   └── vendor/              # marked.js, DOMPurify (no CDN)
├── tests/
│   ├── test_server.py
│   ├── test_cli.py
│   ├── test_memory.py
│   └── test_integration.py
├── systemd/
│   └── airpi.service        # systemd unit
└── docs/
    ├── API_CONTRACT.md
    └── OPERATIONS.md
```

---

## Troubleshooting

### Server won't start

```bash
# Check systemd journal
sudo journalctl -u airpi --since "5 minutes ago"

# Verify models exist
ls -lh /data/models/*.gguf

# Test local run with debug logging
AIRPI_LOG_LEVEL=debug uvicorn server:app --host 127.0.0.1 --port 11435
```

### Slow inference

- Check `/health` for cache hit rate
- Monitor queue depth (`queue_depth` in health)
- Profile with `scripts/airpi_bench.py`
- Consider: model too large for available RAM, competing processes

### Memory issues

- Reduce `AIRPI_N_CTX_SMALL` / `AIRPI_N_CTX_LARGE`
- Enable `AIRPI_MMAP=true` to page to NVMe
- Use smaller model (0.5B or 1.5B instead of 7B)

---

## Integration

### PI Guardian Router

AirPI is API-compatible with Ollama. Set:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11435
```

PI Guardian will route requests to AirPI automatically. No adapter layer needed.

### Home Assistant

Connect HA to AirPI via the `Ollama` integration. Update the base URL to `http://192.168.x.x:11435` (substitute your Pi's local IP).

---

## Contributing

Contributions welcome. Please:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Add tests for new functionality
3. Run `pytest` to verify all tests pass
4. Create a pull request with a clear description

---

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

## Roadmap (Phase 3 — Whimsy Features)

Planned for v2.1:

- **Ambient Mode** — Home Assistant webhooks trigger quick summaries
- **Dream Mode** — Nightly reflection & deduplication
- **Memory Decay** — Older concepts fade visually in the graph
- **Persona System** — Vordefined prompt presets (coder, teacher, coach, etc.)
- **Voice Interface** — Push-to-talk web UI + TTS
- **Hybrid Router** — Route to external APIs (Claude, OpenAI) when needed
- **Live Synapse Formation** — Graph updates in real-time during chat
- **Model Sparring** — Compare responses across models side-by-side

---

## Status

- **Phase 1 ✅** — Backend contract, memory schema, graph foundation (93 tests)
- **Phase 2 ✅** — Chat UI, Memory v1.5, Graph UI, CLI (99 tests)
- **Phase 3 🚧** — Whimsy features (in planning)
- **Phase 4 📋** — Performance gates & optimization

---

## Contact

Questions? Open an issue on GitHub or reach out to the maintainers.

**AirPI** — Intelligent inference on the edge. 🚀
