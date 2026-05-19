# AirPI: LLM Inference Server für Raspberry Pi 5

Ollama-kompatibler Inference-Server auf Basis von `llama-cpp-python`.  
Designed für **Raspberry Pi 5 · 8GB RAM · NVMe SSD**.

## Warum AirPI statt Ollama?

| Feature | Ollama | AirPI |
|---------|--------|-------|
| 7B Modelle (4GB GGUF) | Ja | Ja |
| 14B Modelle via mmap-Paging | Nein | Ja (NVMe als Backing) |
| Concurrency-Control | Nein | Semaphore(1), volle CPU-Zeit |
| Keep-Alive konfigurierbar | 5 min (default) | 15 min (konfigurierbar) |
| Direktes llama.cpp auf ARM | Nein (Daemon) | Ja (kein Overhead) |
| Ollama-kompatible API | n/a | Ja (drop-in) |

## Performance (Pi 5, gemessen)

| Modell | RAM | tok/sec |
|--------|-----|---------|
| Qwen2.5-Coder 1.5B Q4_K_M | 1.2 GB | 10 bis 15 |
| Qwen2.5-Coder 3B Q4_K_M | 2.5 GB | 6 bis 10 |
| Qwen2.5-Coder 7B Q4_K_M | 4.1 GB | 2 bis 4 |
| Qwen2.5-Coder 14B Q4_K_M | 8.1 GB | 1 bis 2 (mmap via NVMe) |

---

## Installation

### 1. Python-Umgebung

```bash
cd /home/alex/AirPI
python3 -m venv .venv
source .venv/bin/activate
```

### 2. llama-cpp-python für ARM64 bauen

llama-cpp-python muss für aarch64 (Pi 5) kompiliert werden. Das dauert ungefähr 10 bis 15 Minuten.

```bash
# ARM NEON Optimierungen aktivieren
CMAKE_ARGS="-DLLAMA_NATIVE=on -DLLAMA_BLAS=OFF" \
  pip install llama-cpp-python --no-binary llama-cpp-python
```

### 3. Weitere Dependencies

```bash
pip install -r requirements.txt
```

### 4. Modelle herunterladen

```bash
# NVMe-Pfad anlegen
sudo mkdir -p /data/models
sudo chown pi:pi /data/models

# Qwen2.5-Coder 1.5B (klein, schnell, für tägliche Tasks)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --local-dir /data/models

# Qwen2.5-Coder 7B (groß, für komplexe Analysen)
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir /data/models

# Optional: 14B (via mmap/NVMe-Paging, ungefähr 1 bis 2 tok/sec)
# huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF \
#   qwen2.5-coder-14b-instruct-q4_k_m.gguf \
#   --local-dir /data/models
```

---

## Starten

### Manuell (Test)

```bash
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 11435
```

### Als Systemd-Service

```bash
sudo cp systemd/airpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now airpi
sudo systemctl status airpi
```

---

## Testen

```bash
# Health Check
curl http://localhost:11435/health

# Modelle auflisten
curl http://localhost:11435/api/tags

# Generate (non-streaming)
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4_k_m.gguf",
    "prompt": "Was ist 2+2?",
    "stream": false
  }'

# Generate (streaming)
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4_k_m.gguf",
    "prompt": "Erkläre Fibonacci in Python.",
    "stream": true
  }'
```

---

## PI Guardian Integration

Damit PI Guardian AirPI statt Ollama nutzt, `OLLAMA_BASE_URL` in der Router-Config ändern:

```bash
# /home/alex/pi-guardian/router/.env (oder config.py)
OLLAMA_BASE_URL=http://127.0.0.1:11435
```

AirPI ist Ollama-kompatibel, keine weiteren Änderungen nötig.

---

## Konfiguration (Umgebungsvariablen)

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `AIRPI_MODELS_DIR` | `/data/models` | Pfad zu GGUF-Dateien |
| `AIRPI_DEFAULT_MODEL` | `qwen2.5-coder-1.5b-q4_k_m.gguf` | Standard-Modell |
| `AIRPI_LARGE_MODEL` | `qwen2.5-coder-7b-q4_k_m.gguf` | Modell für komplexe Prompts |
| `AIRPI_N_THREADS` | `4` | CPU-Threads (Pi 5 hat 4 Kerne) |
| `AIRPI_N_CTX_SMALL` | `4096` | Context-Länge für kleine Modelle |
| `AIRPI_N_CTX_LARGE` | `2048` | Context-Länge für große Modelle |
| `AIRPI_MMAP` | `true` | OS-mmap-Paging (NVMe als Backing) |
| `AIRPI_MLOCK` | `false` | Modell in RAM fixieren (nicht empfohlen für 7B+) |
| `AIRPI_MAX_QUEUE` | `10` | Max. wartende Requests bevor 503 |
| `AIRPI_KEEP_ALIVE_TIMEOUT` | `900` | Sekunden bis Modell entladen wird |
| `AIRPI_HOST` | `127.0.0.1` | Bind-Adresse |
| `AIRPI_PORT` | `11435` | Port |

---

## Architektur

```
Request → /api/generate
    ↓
Queue-Guard (MAX_QUEUE=10) → 503 wenn voll
    ↓
asyncio.Semaphore(1) → serialisiert LLM-Calls
    ↓
ModelManager.get(model_name)
    ├─ Cache-Hit → sofort zurück
    └─ Cache-Miss → asyncio.to_thread(_load) → Llama(use_mmap=True)
    ↓
llama_cpp.Llama(prompt, ...)
    ├─ stream=False → blocking in Thread → GenerateResponse
    └─ stream=True  → Queue-Bridge Thread↔Async → StreamingResponse
    ↓
Antwort im Ollama-Format
```

**Warum `Semaphore(1)`?**  
Der Pi 5 (ARM Cortex-A76, 4 Kerne) verarbeitet einen LLM-Call mit allen 4 Threads in 2 bis 4 tok/sec. Zwei parallele Calls würden je 2 Threads bekommen und beide mit ungefähr 0.8 tok/sec laufen, deutlich schlechter. Serialisierung ist auf ARM ohne GPU die effizientere Strategie.

**Warum `use_mmap=True`?**  
Für 14B Modelle (8GB GGUF) auf einem 8GB Pi: Das OS kann Layer, die aktuell nicht im Forward-Pass benötigt werden, transparent auf die NVMe auslagern (200 bis 400 MB/s). Der Effekt: 14B läuft mit ungefähr 1 bis 2 tok/sec statt gar nicht. Für 7B (4GB) liegt das gesamte Modell im RAM.
