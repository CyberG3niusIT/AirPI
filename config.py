from __future__ import annotations

import os

# ── Model Storage ─────────────────────────────────────────────────────────────

MODELS_DIR: str = os.environ.get("AIRPI_MODELS_DIR", "/data/models")

DEFAULT_MODEL: str = os.environ.get(
    "AIRPI_DEFAULT_MODEL", "qwen2.5-coder-1.5b-q4_k_m.gguf"
)
LARGE_MODEL: str = os.environ.get(
    "AIRPI_LARGE_MODEL", "qwen2.5-coder-7b-q4_k_m.gguf"
)

# ── Inference ─────────────────────────────────────────────────────────────────

# Pi 5 hat 4× ARM Cortex-A76 — alle 4 Threads nutzen
N_THREADS: int = int(os.environ.get("AIRPI_N_THREADS", "4"))

# Context-Länge pro Modellgröße (kleiner = weniger RAM)
N_CTX_SMALL: int = int(os.environ.get("AIRPI_N_CTX_SMALL", "4096"))
N_CTX_LARGE: int = int(os.environ.get("AIRPI_N_CTX_LARGE", "2048"))

# mmap=True: OS-Paging via NVMe — ermöglicht Modelle > RAM-Größe
MMAP: bool = os.environ.get("AIRPI_MMAP", "true").lower() == "true"

# mlock=False: Pi 5 hat 8GB, 14B braucht Paging — nie forcen
MLOCK: bool = os.environ.get("AIRPI_MLOCK", "false").lower() == "true"

# Max. gleichzeitig ausstehende Generate-Requests
MAX_QUEUE: int = int(os.environ.get("AIRPI_MAX_QUEUE", "10"))

# Sekunden ohne Nutzung bis Modell aus RAM entladen wird
KEEP_ALIVE_TIMEOUT: int = int(os.environ.get("AIRPI_KEEP_ALIVE_TIMEOUT", "900"))  # 15 min

# Sekunden bis eine inaktive Session aus dem Cache entfernt wird
SESSION_TTL: int = int(os.environ.get("AIRPI_SESSION_TTL", "1800"))  # 30 min

# Speculative Decoding via prompt-lookup (n-gram, zero extra RAM) oder draft model
SPECULATIVE: bool = os.environ.get("AIRPI_SPECULATIVE", "false").lower() == "true"

# Dateiname des Draft-Modells für echtes Draft-Target Speculative Decoding (1.5B → Gemma/7B)
# Nur aktiv wenn SPECULATIVE=true und das Modell eine "large"-Kennung trägt
SPECULATIVE_DRAFT_MODEL: str | None = os.environ.get("AIRPI_SPECULATIVE_DRAFT_MODEL")

# ── Server ────────────────────────────────────────────────────────────────────

HOST: str = os.environ.get("AIRPI_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("AIRPI_PORT", "11435"))

LOG_LEVEL: str = os.environ.get("AIRPI_LOG_LEVEL", "INFO")

# ── Keyword-basierte Modellauswahl ────────────────────────────────────────────
# Prompts mit diesen Keywords werden mit dem größeren Modell bearbeitet
LARGE_MODEL_KEYWORDS: list[str] = [
    "architektur", "refactor", "analyse", "debug", "komplex",
    "architecture", "complex", "analysis",
]
