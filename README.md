# AirPI: Local AI Control Plane

> Local AI Control Plane for private and small-scale infrastructure. Ollama-compatible edge inference server with policy enforcement, memory control, and audit logging for Raspberry Pi 5 and similar devices.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-backed-444444)](https://github.com/ggerganov/llama.cpp)
[![FastAPI](https://img.shields.io/badge/FastAPI-00A393?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-99%2F99-28a745)](./tests/)
[![Ollama API](https://img.shields.io/badge/Ollama-API_compatible-0B5FFF)](https://github.com/ollama/ollama)

---

## Overview

AirPI is an infrastructural component that securely accepts, evaluates, locally processes, and logs AI requests. It provides a structured boundary between your local network services and AI inference models. It is designed to ensure that AI operates in a transparent, auditable, and locally-controlled manner.

### Core Capabilities

| Component | Capability |
|-----------|-----------|
| **Local Inference** | Secure, on-premise execution using llama.cpp and KV-cache reuse |
| **Policy and Routing** | Evaluates requests and enforces processing boundaries |
| **Memory Control** | Verifiable and strictly controllable fact storage with deduplication |
| **Audit Logging** | Transparent records of model routing and processing |
| **API and Integrations** | Ollama-compatible API for Home Assistant and PI Guardian |
| **Management Interfaces** | Web-UI and CLI tailored for infrastructure diagnostics and control |

---

## Quick Start

### Prerequisites

- Raspberry Pi 5 with 8 GB RAM and NVMe SSD (or equivalent hardware)
- Python 3.11+
- systemd (for service operation)

### 1. Clone and Setup Environment

```bash
git clone https://github.com/CyberG3niusIT/AirPI.git
cd AirPI
python -m venv venv
source venv/bin/activate
```

### 2. Build & Install

```bash
# Build llama-cpp-python for ARM64 (required)
CMAKE_ARGS="-DLLAMA_NATIVE=on -DLLAMA_BLAS=OFF" \
  pip install llama-cpp-python --no-binary llama-cpp-python

# Install AirPI + dependencies (enables CLI)
pip install -e .
```

### 3. Download Models

Create the default model directory and download a GGUF model:

```bash
sudo mkdir -p /data/models
sudo chown $USER:$USER /data/models

# Example: Qwen2.5-Coder 1.5B (Fast, low-latency lane)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --local-dir /data/models

# Example: Qwen2.5-Coder 7B (For complex tasks)
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

### Management UI

- **Diagnostic Interface:** Review system behavior, model responses, and adjust system prompts.
- **Local Analytics:** Live statistics displaying current model load, queue depth, cache hit rate, and inference speed.
- **Dependency-free:** All resources are served locally to ensure offline capability.

### Memory Control

- **Rule-based Extraction:** Structured capture of configuration and facts.
- **Categorization:** Classifies entities as fact, preference, correction, project, system, or todo.
- **Bounded Storage:** Enforces strict limits on memory entries to maintain performance and control.
- **Auditable Deletion:** Full capability to wipe stored facts based on keywords or categories.

### Knowledge Graph and Audit

- **Concept Auditing:** Visual mapping of extracted entities, showing source, age, and relationship.
- **Transparency:** Backlinks panel enables administrators to verify the origin of learned facts.
- **Exportable Records:** Graph exports support documentation of system knowledge.

### Command Line Interface

Full administrative command-line interface for headless management:

```bash
# Start interactive management session
airpi chat

# Apply a specific system policy
airpi chat --system "Enforce strict coding standard compliance"

# Machine-readable output for scripts
airpi chat --json

# Monitor system health
airpi status --watch
airpi status --plain

# Multiline input support
prompt> explain\
... tensorflow \
... architecture
```

### Session-Based KV-Cache

Pass `session_id` to reuse KV-cache across requests, reducing redundant processing:

```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "prompt": "Continue the analysis...",
    "session_id": "audit-session-001"
  }'
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/live` | GET | Liveness check |
| `/ready` | GET | Readiness status (models and queue) |
| `/health` | GET | Runtime state, cache statistics, and resource info |
| `/metrics` | GET | Prometheus format metrics for monitoring |
| `/api/tags` | GET | Ollama-compatible model listing |
| `/api/generate` | POST | Ollama-compatible inference (streaming) |
| `/api/chat` | POST | Non-streaming chat with system prompt configuration |
| `/memory/store` | POST | Manually store a state or fact |
| `/memory/delete` | POST | Delete facts by keyword to enforce data policies |
| `/memory` | GET | Retrieve current memory allocations |
| `/graph/data` | GET | Extract nodes and edges for external auditing |
| `/ui/` | GET | Serve the administrative web interface |
| `/ui/graph.html` | GET | Serve the graph visualizer |

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AIRPI_MODELS_DIR` | `/data/models` | GGUF model directory |
| `AIRPI_DEFAULT_MODEL` | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Default execution target |
| `AIRPI_FAST_MODEL` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Low-latency processing model |
| `AIRPI_LARGE_MODEL` | `qwen2.5-coder-7b-q4_k_m.gguf` | Model for complex parsing |
| `AIRPI_N_THREADS` | `3` | CPU decode threads |
| `AIRPI_N_CTX_SMALL` | `2048` | Small model context window |
| `AIRPI_N_CTX_LARGE` | `2048` | Large model context window |
| `AIRPI_MAX_QUEUE` | `10` | Maximum queued requests |
| `AIRPI_HOST` | `127.0.0.1` | Bind address |
| `AIRPI_PORT` | `11435` | HTTP port |
| `AIRPI_LOG_LEVEL` | `info` | Log verbosity |
| `AIRPI_API_KEY` | `None` | Pre-shared key for Bearer authentication |

---

## Architecture

See `docs/ARCHITECTURE.md` and `docs/PRODUCT_IDENTITY.md` for detailed information on the system's structural design and product boundaries.

---

## Roadmap

The development of AirPI prioritizes robust infrastructure over experimental features.

1. **AirPI Core:** Stable local inference
2. **AirPI API:** Ollama-compatible, strictly documented interfaces
3. **AirPI Policy:** Rules for local processing, forwarding, and blocking
4. **AirPI Router:** Routing between local models, external backends, and enforcing blocks
5. **AirPI Memory Control:** Conscious, verifiable, and deletable state management
6. **AirPI Audit:** Extensive logging, decision proofs, and metrics
7. **AirPI Integrations:** Deep hooks for PI Guardian, Home Assistant, CLI, and Web-UI
8. **AirPI Operations:** systemd hardening, health checks, backups, and secure updates
9. **AirPI UI:** Refined control center for infrastructure management

### Optional Future Extensions
Features such as conversational personas, voice interfaces, or heuristic model sparring are considered strictly secondary. If implemented, they will exist as modular, optional extensions that do not interfere with the core infrastructural guarantees.

---

## Integration

### PI Guardian Router

AirPI acts as the intelligence backend for PI Guardian. Configure PI Guardian to target AirPI using the standard Ollama convention:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11435
```

### Home Assistant

AirPI can securely process smart home automation routines. Connect Home Assistant via the `Ollama` integration by directing the base URL to the local instance (e.g., `http://192.168.x.x:11435`).

---

## Contributing

Contributions must adhere to the principles outlined in `docs/PRODUCT_IDENTITY.md`.

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Validate compliance with the product identity guidelines
3. Add tests for new functionality
4. Run `pytest` to verify stability
5. Submit a pull request with a clear description of infrastructural impact

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
