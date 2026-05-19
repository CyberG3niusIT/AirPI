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

# Pi 5 CPU-only turbo profile. 3 decode threads measured slightly faster than 4
# because one core remains available for uvicorn, kernel work, and memory paging.
N_THREADS: int = int(os.environ.get("AIRPI_N_THREADS", "3"))

# Context-Länge pro Modellgröße (kleiner = weniger RAM und schnellerer KV-Cache)
N_CTX_SMALL: int = int(os.environ.get("AIRPI_N_CTX_SMALL", "2048"))
N_CTX_LARGE: int = int(os.environ.get("AIRPI_N_CTX_LARGE", "2048"))

# Prompt ingestion tuning. These values are conservative for Raspberry Pi 5
# and can be overridden from /etc/airpi/airpi.env.
N_THREADS_BATCH: int = int(os.environ.get("AIRPI_N_THREADS_BATCH", "4"))
N_BATCH_SMALL: int = int(os.environ.get("AIRPI_N_BATCH_SMALL", "1024"))
N_BATCH_LARGE: int = int(os.environ.get("AIRPI_N_BATCH_LARGE", "512"))
N_UBATCH_SMALL: int = int(os.environ.get("AIRPI_N_UBATCH_SMALL", "512"))
N_UBATCH_LARGE: int = int(os.environ.get("AIRPI_N_UBATCH_LARGE", "256"))
FLASH_ATTN: bool = os.environ.get("AIRPI_FLASH_ATTN", "true").lower() == "true"

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
