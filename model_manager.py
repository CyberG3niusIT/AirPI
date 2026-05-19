from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import numpy as np

from config import (
    DEFAULT_MODEL,
    FAST_MODEL,
    FAST_MODEL_ALIASES,
    KEEP_ALIVE_TIMEOUT,
    LARGE_MODEL_KEYWORDS,
    LARGE_MODEL,
    FLASH_ATTN,
    MLOCK,
    MMAP,
    N_BATCH_LARGE,
    N_BATCH_SMALL,
    MODELS_DIR,
    N_CTX_LARGE,
    N_CTX_SMALL,
    N_THREADS,
    N_THREADS_BATCH,
    N_UBATCH_LARGE,
    N_UBATCH_SMALL,
    SESSION_TTL,
    SPECULATIVE,
    SPECULATIVE_DRAFT_MODEL,
)

if TYPE_CHECKING:
    from llama_cpp import Llama
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_LARGE_SIZE_MARKERS = ("4b", "7b", "8b", "13b", "14b", "70b")
StreamItem = str | Exception | None


class ModelState:
    COLD = "cold"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    RECOVERING = "recovering"
    FAILED = "failed"
    EVICTED = "evicted"


def _is_large_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(m in lower for m in _LARGE_SIZE_MARKERS)


def _is_recoverable_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, ValueError) and (
        "could not broadcast input array" in text
        or "shape mismatch" in text
    )


def select_model_for_prompt(prompt: str, preferred_model: str | None = None) -> str:
    """Wählt Modell anhand Alias, explizitem Modell oder Prompt-Keywords."""
    if preferred_model:
        if preferred_model.lower() in FAST_MODEL_ALIASES:
            return FAST_MODEL
        if preferred_model.lower() == "default":
            return DEFAULT_MODEL
        if preferred_model.lower() == "large":
            return LARGE_MODEL
        return preferred_model
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in LARGE_MODEL_KEYWORDS):
        return LARGE_MODEL
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
    token_queue: "asyncio.Queue[StreamItem]",
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Synchrone Streaming-Variante für run_in_executor.

    Sendet decodierte Token-Strings in die Queue; None = Sentinel.
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
        loop.call_soon_threadsafe(token_queue.put_nowait, exc)
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
        self._states: dict[str, str] = {}
        self._last_errors: dict[str, str] = {}
        self._recovery_count = 0
        self._eviction_count = 0
        self._last_recovery: dict[str, str | float] | None = None
        self._warmup_ok = False
        self._warmup_error: str | None = None

    # ── Model Cache ───────────────────────────────────────────────────────────

    def _set_state(self, model_name: str, state: str, error: str | None = None) -> None:
        self._states[model_name] = state
        if error is None:
            self._last_errors.pop(model_name, None)
        else:
            self._last_errors[model_name] = error

    async def get(self, model_name: str) -> "Llama":
        async with self._lock:
            if model_name in self._instances:
                llm, _ = self._instances[model_name]
                self._instances[model_name] = (llm, time.monotonic())
                self._set_state(model_name, ModelState.READY)
                logger.debug("model cache hit: %s", model_name)
                return llm

        logger.info("loading model: %s", model_name)
        self._set_state(model_name, ModelState.LOADING)
        try:
            llm = await asyncio.to_thread(self._load, model_name)
        except Exception as exc:
            self._set_state(model_name, ModelState.FAILED, str(exc))
            raise

        async with self._lock:
            if model_name not in self._instances:
                self._instances[model_name] = (llm, time.monotonic())
            self._set_state(model_name, ModelState.READY)
        return llm

    def _load(self, model_name: str) -> "Llama":
        from llama_cpp import Llama

        path = os.path.join(MODELS_DIR, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {path}")

        large = _is_large_model(model_name)
        n_ctx = N_CTX_LARGE if large else N_CTX_SMALL
        n_batch = N_BATCH_LARGE if large else N_BATCH_SMALL
        n_ubatch = N_UBATCH_LARGE if large else N_UBATCH_SMALL
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
                            n_threads_batch=N_THREADS_BATCH,
                            n_batch=N_BATCH_SMALL,
                            n_ubatch=N_UBATCH_SMALL,
                            flash_attn=FLASH_ATTN,
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
                elif large:
                    draft_model = LlamaPromptLookupDecoding(num_pred_tokens=10, max_ngram_size=2)
                    logger.info("speculative: prompt-lookup enabled: %s", model_name)
                else:
                    logger.info(
                        "speculative: disabled for small model due stability preference: %s",
                        model_name,
                    )
            except Exception as exc:
                logger.warning("speculative decoding disabled (setup failed): %s", exc)

        logger.info(
            (
                "llama_cpp load: model=%s n_ctx=%d n_threads=%d "
                "n_threads_batch=%d n_batch=%d n_ubatch=%d flash_attn=%s "
                "mmap=%s mlock=%s speculative=%s"
            ),
            model_name,
            n_ctx,
            N_THREADS,
            N_THREADS_BATCH,
            n_batch,
            n_ubatch,
            FLASH_ATTN,
            MMAP,
            MLOCK,
            draft_model is not None,
        )
        return Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=N_THREADS,
            n_threads_batch=N_THREADS_BATCH,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attn=FLASH_ATTN,
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

    async def _invalidate_model_state(self, model_name: str) -> None:
        async with self._lock:
            self._set_state(model_name, ModelState.RECOVERING)
            removed = self._instances.pop(model_name, None)
            if removed is not None:
                logger.warning("invalidated model instance after generation failure: %s", model_name)

        async with self._session_lock:
            stale_sessions = [
                session_id
                for session_id, (bound_model, _) in self._sessions.items()
                if bound_model == model_name
            ]
            for session_id in stale_sessions:
                del self._sessions[session_id]
            if stale_sessions:
                logger.warning(
                    "invalidated %d session(s) for model=%s after generation failure",
                    len(stale_sessions),
                    model_name,
                )
        self._recovery_count += 1
        self._last_recovery = {
            "model": model_name,
            "timestamp": time.time(),
        }

    # ── Generate ──────────────────────────────────────────────────────────────

    def _cache_hit(self, session_id: str | None, model_name: str) -> bool:
        return session_id is not None and self._is_session_valid(session_id, model_name)

    async def _drain_stream_queue(self, queue: "asyncio.Queue[StreamItem]") -> AsyncGenerator[str, None]:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

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
        cache_hit = self._cache_hit(session_id, model_name)
        reset = not cache_hit
        self._set_state(model_name, ModelState.GENERATING)
        try:
            result = await asyncio.to_thread(
                _generate_with_cache, llm, prompt, max_tokens, temperature, top_p, stop, reset,
            )
        except Exception as exc:
            if not _is_recoverable_generation_error(exc):
                raise

            logger.warning(
                "recoverable generate failure detected for model=%s session=%s cache_hit=%s: %s",
                model_name,
                session_id or "-",
                cache_hit,
                exc,
            )
            await self._invalidate_model_state(model_name)
            llm = await self.get(model_name)
            result = await asyncio.to_thread(
                _generate_with_cache, llm, prompt, max_tokens, temperature, top_p, stop, True,
            )
            cache_hit = False
        finally:
            if model_name in self._instances:
                self._set_state(model_name, ModelState.READY)

        if session_id is not None:
            await self._touch_session(session_id, model_name)
        result["airpi"] = {"cache_hit": cache_hit}

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
    ) -> AsyncGenerator[str, None]:
        llm = await self.get(model_name)
        cache_hit = self._cache_hit(session_id, model_name)
        reset = not cache_hit
        self._set_state(model_name, ModelState.GENERATING)

        if session_id is not None:
            await self._touch_session(session_id, model_name)

        yielded = False
        try:
            queue: asyncio.Queue[StreamItem] = asyncio.Queue(maxsize=50)
            loop = asyncio.get_running_loop()
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _stream_with_cache,
                    llm, prompt, max_tokens, temperature, top_p, stop, reset, queue, loop,
                )
            )
            async for token in self._drain_stream_queue(queue):
                yielded = True
                yield token
            await worker
        except Exception as exc:
            if yielded or not _is_recoverable_generation_error(exc):
                self._set_state(model_name, ModelState.FAILED, str(exc))
                raise

            logger.warning(
                "recoverable stream failure detected for model=%s session=%s cache_hit=%s: %s",
                model_name,
                session_id or "-",
                cache_hit,
                exc,
            )
            await self._invalidate_model_state(model_name)
            llm = await self.get(model_name)
            queue = asyncio.Queue(maxsize=50)
            loop = asyncio.get_running_loop()
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _stream_with_cache,
                    llm, prompt, max_tokens, temperature, top_p, stop, True, queue, loop,
                )
            )
            async for token in self._drain_stream_queue(queue):
                yield token
            await worker
        finally:
            if model_name in self._instances:
                self._set_state(model_name, ModelState.READY)

        logger.info(
            "stream: model=%s session=%s cache_hit=%s",
            model_name, session_id or "-", cache_hit,
        )

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
                self._set_state(name, ModelState.EVICTED)
                self._eviction_count += 1
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

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    @property
    def last_recovery(self) -> dict[str, str | float] | None:
        return self._last_recovery

    @property
    def warmup_ok(self) -> bool:
        return self._warmup_ok

    @property
    def warmup_error(self) -> str | None:
        return self._warmup_error

    def runtime_status(self) -> dict:
        return {
            "models": {
                name: {
                    "state": self._states.get(name, ModelState.COLD),
                    "loaded": name in self._instances,
                    "last_error": self._last_errors.get(name),
                }
                for name in sorted(set(self._states) | set(self._instances))
            },
            "loaded_models": self.loaded_models,
            "active_sessions": self.active_sessions,
            "recovery_count": self.recovery_count,
            "eviction_count": self.eviction_count,
            "last_recovery": self.last_recovery,
            "warmup_ok": self.warmup_ok,
            "warmup_error": self.warmup_error,
        }

    async def warmup(self, model_name: str) -> None:
        try:
            await self.get(model_name)
            self._warmup_ok = True
            self._warmup_error = None
            logger.info("warmup OK: %s", model_name)
        except Exception as exc:
            self._warmup_ok = False
            self._warmup_error = str(exc)
            logger.warning("warmup failed (non-critical): %s - %s", model_name, exc)


# Singleton
manager = ModelManager()
