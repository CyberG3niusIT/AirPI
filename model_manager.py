from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from config import (
    KEEP_ALIVE_TIMEOUT,
    LARGE_MODEL_KEYWORDS,
    LARGE_MODEL,
    MLOCK,
    MMAP,
    MODELS_DIR,
    N_CTX_LARGE,
    N_CTX_SMALL,
    N_THREADS,
)

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Schlüsselwörter die ein Modell als "groß" klassifizieren (7B, 13B, 14B)
_LARGE_SIZE_MARKERS = ("7b", "8b", "13b", "14b", "70b")


def _is_large_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(m in lower for m in _LARGE_SIZE_MARKERS)


def select_model_for_prompt(prompt: str, preferred_model: str | None = None) -> str:
    """Wählt Modell anhand Prompt-Keywords. preferred_model überschreibt alles."""
    if preferred_model:
        return preferred_model
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in LARGE_MODEL_KEYWORDS):
        return LARGE_MODEL
    from config import DEFAULT_MODEL
    return DEFAULT_MODEL


class ModelManager:
    """Verwaltet geladene Llama-Instanzen im Speicher.

    Hält Modelle nach dem ersten Load warm. Entlädt automatisch nach
    KEEP_ALIVE_TIMEOUT Sekunden ohne Nutzung (via evict_stale()).
    Ein asyncio.Semaphore(1) serialisiert alle Generate-Calls — der Pi 5
    arbeitet einen LLM-Call mit vollen 4 Threads statt zwei parallel mit
    je 2 Threads (deutlich effizienter auf ARM).
    """

    def __init__(self) -> None:
        # name → (llama_instance, last_used_timestamp)
        self._instances: dict[str, tuple["Llama", float]] = {}
        self._lock = asyncio.Lock()
        # Serialisiert alle generate()-Aufrufe global
        self.semaphore = asyncio.Semaphore(1)

    async def get(self, model_name: str) -> "Llama":
        """Gibt geladene Instanz zurück, lädt bei Bedarf (blockierend in Thread)."""
        async with self._lock:
            if model_name in self._instances:
                llm, _ = self._instances[model_name]
                self._instances[model_name] = (llm, time.monotonic())
                logger.debug("model cache hit: %s", model_name)
                return llm

        logger.info("loading model: %s", model_name)
        llm = await asyncio.to_thread(self._load, model_name)

        async with self._lock:
            # Doppeltes Laden vermeiden (andere Coroutine könnte auch geladen haben)
            if model_name not in self._instances:
                self._instances[model_name] = (llm, time.monotonic())
        return llm

    def _load(self, model_name: str) -> "Llama":
        from llama_cpp import Llama

        path = os.path.join(MODELS_DIR, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {path}")

        large = _is_large_model(model_name)
        n_ctx = N_CTX_LARGE if large else N_CTX_SMALL

        logger.info(
            "llama_cpp load: model=%s n_ctx=%d n_threads=%d mmap=%s mlock=%s",
            model_name, n_ctx, N_THREADS, MMAP, MLOCK,
        )
        return Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=N_THREADS,
            use_mmap=MMAP,
            use_mlock=MLOCK,
            verbose=False,
        )

    async def evict_stale(self) -> list[str]:
        """Entlädt Modelle die länger als KEEP_ALIVE_TIMEOUT nicht genutzt wurden.

        Wird als periodischer Background-Task aufgerufen.
        Gibt Liste der entladenen Modell-Namen zurück.
        """
        now = time.monotonic()
        async with self._lock:
            stale = [
                name for name, (_, last_used) in self._instances.items()
                if now - last_used > KEEP_ALIVE_TIMEOUT
            ]
            for name in stale:
                del self._instances[name]
                logger.info("evicted stale model: %s", name)
        return stale

    @property
    def loaded_models(self) -> list[str]:
        return list(self._instances.keys())

    async def warmup(self, model_name: str) -> None:
        """Lädt Modell beim Start vor — erster echter Request ist dann sofort schnell."""
        try:
            await self.get(model_name)
            logger.info("warmup OK: %s", model_name)
        except Exception as exc:
            logger.warning("warmup fehlgeschlagen (nicht kritisch): %s — %s", model_name, exc)


# Singleton — wird in server.py beim Startup initialisiert
manager = ModelManager()
