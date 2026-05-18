from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

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
    SESSION_TTL,
    SPECULATIVE,
    SPECULATIVE_DRAFT_MODEL,
)

if TYPE_CHECKING:
    from llama_cpp import Llama
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_LARGE_SIZE_MARKERS = ("4b", "7b", "8b", "13b", "14b", "70b")


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


# ── KV-Cache-fähiger Generate-Wrapper ────────────────────────────────────────

def _generate_with_cache(
    llm: "Llama",
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    reset: bool,
) -> dict:
    """Ruft llm.generate() direkt auf, um reset=False zu unterstützen.

    Llama.__call__() leitet reset= nicht weiter — daher dieser Wrapper.
    Bei reset=False findet llama-cpp automatisch den längsten gemeinsamen
    Prefix mit dem gecachten KV-State und verarbeitet nur neue Token.
    """
    import llama_cpp

    prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    stop_bytes = [s.encode("utf-8") for s in stop if s]
    completion_tokens: list[int] = []
    finish_reason = "length"
    accumulated = b""

    for token in llm.generate(prompt_tokens, temp=temperature, top_p=top_p, reset=reset):
        if llama_cpp.llama_vocab_is_eog(llm._model.vocab, token):
            finish_reason = "stop"
            break
        completion_tokens.append(token)
        # Immer vollständige Liste detokenizen (nicht inkrementell) für korrekte Byte-Grenzen
        accumulated = llm.detokenize(completion_tokens)
        stop_hit = any(sb in accumulated for sb in stop_bytes)
        if stop_hit:
            finish_reason = "stop"
            break
        if len(completion_tokens) >= max_tokens:
            break

    text = accumulated.decode("utf-8", errors="replace")
    for s in stop:
        if text.endswith(s):
            text = text[: -len(s)]
            break

    return {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completion_tokens),
            "total_tokens": len(prompt_tokens) + len(completion_tokens),
        },
    }


def _stream_with_cache(
    llm: "Llama",
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    reset: bool,
    token_queue: "asyncio.Queue[str | None]",
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Synchrone Streaming-Variante für run_in_executor.

    Sendet decodierte Token-Strings in die Queue; None = Sentinel (fertig/Fehler).
    """
    import llama_cpp

    prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    stop_bytes = [s.encode("utf-8") for s in stop if s]
    completion_tokens: list[int] = []

    try:
        for token in llm.generate(prompt_tokens, temp=temperature, top_p=top_p, reset=reset):
            if llama_cpp.llama_vocab_is_eog(llm._model.vocab, token):
                break
            completion_tokens.append(token)
            accumulated = llm.detokenize(completion_tokens)
            text = accumulated.decode("utf-8", errors="replace")
            loop.call_soon_threadsafe(token_queue.put_nowait, text)
            if any(sb in accumulated for sb in stop_bytes) or len(completion_tokens) >= max_tokens:
                break
    except Exception as exc:
        logger.error("stream_with_cache error: %s", exc)
    finally:
        loop.call_soon_threadsafe(token_queue.put_nowait, None)


# ── Draft-Modell für Speculative Decoding ────────────────────────────────────

class LlamaModelDraft:
    """Wraps eine kleine Llama-Instanz als Draft-Modell für Speculative Decoding.

    Das 1.5B-Modell schlägt K Token vor; das große Target-Modell verifiziert
    alle K in einem Prefill-Pass — schneller als K sequenzielle Decode-Schritte.
    """

    def __init__(self, draft_llm: "Llama") -> None:
        self._draft = draft_llm

    def __call__(
        self,
        input_ids: "npt.NDArray[np.intc]",
        /,
        **kwargs: Any,
    ) -> "npt.NDArray[np.intc]":
        num_pred = kwargs.get("num_pred_tokens", 5)
        draft_tokens: list[int] = []
        # temp=0.0 (greedy) für maximale Akzeptanzrate beim Target
        for token in self._draft.generate(input_ids.tolist(), reset=False, temp=0.0):
            draft_tokens.append(token)
            if len(draft_tokens) >= num_pred:
                break
        return np.array(draft_tokens, dtype=np.intc)


# ── Model Manager ─────────────────────────────────────────────────────────────

class ModelManager:
    """Verwaltet Llama-Instanzen und Session-KV-Caches.

    - Semaphore(1) serialisiert alle LLM-Calls (ARM ohne GPU: effizienter als parallel)
    - Sessions tracken welche session_id auf welchem Modell KV-Cache hat
    - Bei reset=False findet llama-cpp automatisch den Prefix-Match und spart
      die Re-Tokenisierung des System-Prompts bei Multi-Step-Agent-Runs
    """

    def __init__(self) -> None:
        self._instances: dict[str, tuple["Llama", float]] = {}
        self._lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(1)

        # session_id → (model_name, last_used_ts)
        self._sessions: dict[str, tuple[str, float]] = {}
        self._session_lock = asyncio.Lock()

    # ── Model Cache ───────────────────────────────────────────────────────────

    async def get(self, model_name: str) -> "Llama":
        async with self._lock:
            if model_name in self._instances:
                llm, _ = self._instances[model_name]
                self._instances[model_name] = (llm, time.monotonic())
                logger.debug("model cache hit: %s", model_name)
                return llm

        logger.info("loading model: %s", model_name)
        llm = await asyncio.to_thread(self._load, model_name)

        async with self._lock:
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
        draft_model = None

        if SPECULATIVE:
            try:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                if large and SPECULATIVE_DRAFT_MODEL:
                    draft_path = os.path.join(MODELS_DIR, SPECULATIVE_DRAFT_MODEL)
                    if os.path.isfile(draft_path):
                        draft_llm = Llama(
                            model_path=draft_path,
                            n_ctx=N_CTX_SMALL,
                            n_threads=N_THREADS,
                            use_mmap=MMAP,
                            use_mlock=False,
                            verbose=False,
                        )
                        draft_model = LlamaModelDraft(draft_llm)
                        logger.info(
                            "speculative: draft-target enabled (draft=%s target=%s)",
                            SPECULATIVE_DRAFT_MODEL, model_name,
                        )
                    else:
                        draft_model = LlamaPromptLookupDecoding(num_pred_tokens=10, max_ngram_size=2)
                        logger.warning("speculative: draft file not found, using prompt-lookup: %s", model_name)
                else:
                    draft_model = LlamaPromptLookupDecoding(num_pred_tokens=10, max_ngram_size=2)
                    logger.info("speculative: prompt-lookup enabled: %s", model_name)
            except Exception as exc:
                logger.warning("speculative decoding disabled (setup failed): %s", exc)

        logger.info(
            "llama_cpp load: model=%s n_ctx=%d n_threads=%d mmap=%s mlock=%s speculative=%s",
            model_name, n_ctx, N_THREADS, MMAP, MLOCK, draft_model is not None,
        )
        return Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=N_THREADS,
            use_mmap=MMAP,
            use_mlock=MLOCK,
            verbose=False,
            draft_model=draft_model,
        )

    # ── Session KV-Cache ──────────────────────────────────────────────────────

    def _is_session_valid(self, session_id: str, model_name: str) -> bool:
        if session_id not in self._sessions:
            return False
        bound_model, _ = self._sessions[session_id]
        return bound_model == model_name

    async def _touch_session(self, session_id: str, model_name: str) -> None:
        async with self._session_lock:
            self._sessions[session_id] = (model_name, time.monotonic())

    # ── Generate ──────────────────────────────────────────────────────────────

    async def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str],
        session_id: str | None = None,
    ) -> dict:
        llm = await self.get(model_name)
        cache_hit = session_id is not None and self._is_session_valid(session_id, model_name)
        reset = not cache_hit

        result = await asyncio.to_thread(
            _generate_with_cache, llm, prompt, max_tokens, temperature, top_p, stop, reset,
        )

        if session_id is not None:
            await self._touch_session(session_id, model_name)

        logger.info(
            "generate: model=%s session=%s cache_hit=%s prompt_tokens=%d completion_tokens=%d",
            model_name, session_id or "-", cache_hit,
            result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"],
        )
        return result

    async def stream_generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str],
        session_id: str | None = None,
    ) -> "asyncio.Queue[str | None]":
        llm = await self.get(model_name)
        cache_hit = session_id is not None and self._is_session_valid(session_id, model_name)
        reset = not cache_hit

        if session_id is not None:
            await self._touch_session(session_id, model_name)

        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=50)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None, _stream_with_cache,
            llm, prompt, max_tokens, temperature, top_p, stop, reset, queue, loop,
        )

        logger.info(
            "stream: model=%s session=%s cache_hit=%s",
            model_name, session_id or "-", cache_hit,
        )
        return queue

    # ── Eviction ──────────────────────────────────────────────────────────────

    async def evict_stale(self) -> list[str]:
        now = time.monotonic()

        async with self._lock:
            stale = [
                name for name, (_, last_used) in self._instances.items()
                if now - last_used > KEEP_ALIVE_TIMEOUT
            ]
            for name in stale:
                del self._instances[name]
                logger.info("evicted stale model: %s", name)

        async with self._session_lock:
            dead = [
                sid for sid, (mname, ts) in self._sessions.items()
                if mname in stale or now - ts > SESSION_TTL
            ]
            for sid in dead:
                del self._sessions[sid]
            if dead:
                logger.info("evicted %d stale session(s)", len(dead))

        return stale

    @property
    def loaded_models(self) -> list[str]:
        return list(self._instances.keys())

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    async def warmup(self, model_name: str) -> None:
        try:
            await self.get(model_name)
            logger.info("warmup OK: %s", model_name)
        except Exception as exc:
            logger.warning("warmup fehlgeschlagen (nicht kritisch): %s — %s", model_name, exc)


# Singleton
manager = ModelManager()
