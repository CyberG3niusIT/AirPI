"""AirPI — Ollama-kompatibler Inference Server für Raspberry Pi 5.

Implementiert die Ollama HTTP API (/api/generate, /api/tags, /health) mit:
- llama-cpp-python (GGUF, direkter ARM-optimierter Inference)
- asyncio.Semaphore(1) — serialisierte LLM-Calls für maximale CPU-Auslastung
- OS-mmap-Paging für Modelle > RAM (use_mmap=True)
- Automatischer Keep-Alive: Modelle bleiben 15 min im Speicher
- Warmup beim Start: DEFAULT_MODEL wird direkt in RAM geladen

Port: 11435 (Ollama bleibt auf 11434, beide laufen parallel)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from model_manager import manager, select_model_for_prompt

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger("airpi")

# ── Queue-Tiefe (wird nicht atomar gezählt, aber Semaphore reicht für Pi) ────

_queue_depth: int = 0

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("AirPI starting on %s:%d", config.HOST, config.PORT)

    # Modell beim Start vorladen (Warmup)
    await manager.warmup(config.DEFAULT_MODEL)

    # Periodischer Eviction-Task (alle 5 Minuten)
    async def _evict_loop() -> None:
        while True:
            await asyncio.sleep(300)
            evicted = await manager.evict_stale()
            if evicted:
                logger.info("evicted models: %s", evicted)

    evict_task = asyncio.create_task(_evict_loop())

    yield

    evict_task.cancel()
    logger.info("AirPI shutdown")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AirPI Inference Server",
    description="Ollama-kompatibler LLM Inference Server für Raspberry Pi 5",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response Models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=lambda: ["</s>"])
    # keep_alive wird für Kompatibilität akzeptiert, aber intern ignoriert
    # (ModelManager übernimmt das selbst)
    keep_alive: str | None = None
    # Optionales Modell-Override
    preferred_model: str | None = None


class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0     # Nanosekunden (Ollama-Format)
    eval_count: int = 0         # Token-Anzahl
    eval_duration: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "loaded_models": manager.loaded_models}


@app.get("/api/tags")
async def list_models() -> dict:
    """Ollama-kompatibles Modell-Listing."""
    models_dir = config.MODELS_DIR
    models = []

    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".gguf"):
                fpath = os.path.join(models_dir, fname)
                stat = os.stat(fpath)
                models.append({
                    "name": fname,
                    "model": fname,
                    "modified_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)
                    ),
                    "size": stat.st_size,
                    "digest": "",
                    "details": {
                        "format": "gguf",
                        "loaded": fname in manager.loaded_models,
                    },
                })

    return {"models": models}


@app.post("/api/generate", response_model=None)
async def generate(request: GenerateRequest) -> StreamingResponse | GenerateResponse:
    """Ollama-kompatibler Generate-Endpoint.

    Akzeptiert denselben JSON-Body wie `POST /api/generate` bei Ollama.
    Unterstützt streaming (stream=true) und non-streaming (stream=false).
    """
    global _queue_depth

    # Queue-Guard: zu viele wartende Requests → 503
    if _queue_depth >= config.MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_queue_full",
                "message": f"Maximale Queue-Tiefe ({config.MAX_QUEUE}) erreicht. Bitte später erneut versuchen.",
            },
        )

    # Modell auswählen (preferred_model überschreibt Keywords)
    model_name = select_model_for_prompt(request.prompt, request.preferred_model or request.model)

    _queue_depth += 1
    start_ns = time.perf_counter_ns()

    try:
        # Semaphore serialisiert LLM-Calls — Pi 5 arbeitet einen mit 4 Threads statt
        # zwei parallel mit je 2 (auf ARM ohne GPU effizienter)
        async with manager.semaphore:
            llm = await manager.get(model_name)

            if request.stream:
                return StreamingResponse(
                    _stream_generate(llm, request, model_name, start_ns),
                    media_type="application/x-ndjson",
                )
            else:
                return await _blocking_generate(llm, request, model_name, start_ns)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "model_not_found", "message": str(exc)}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("generate error: model=%s", model_name)
        raise HTTPException(status_code=500, detail={"error": "inference_error", "message": str(exc)}) from exc
    finally:
        _queue_depth -= 1


async def _blocking_generate(
    llm: "object",
    request: GenerateRequest,
    model_name: str,
    start_ns: int,
) -> GenerateResponse:
    """Non-streaming Generate — gibt vollständige Antwort zurück."""
    result = await asyncio.to_thread(
        llm,
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        echo=False,
    )

    elapsed_ns = time.perf_counter_ns() - start_ns
    choice = result["choices"][0]
    text = choice.get("text", "")
    tokens = result.get("usage", {}).get("completion_tokens", 0)

    logger.info(
        "generate done: model=%s tokens=%d elapsed_ms=%d",
        model_name, tokens, elapsed_ns // 1_000_000,
    )

    return GenerateResponse(
        model=model_name,
        response=text,
        done=True,
        done_reason=choice.get("finish_reason", "stop"),
        total_duration=elapsed_ns,
        eval_count=tokens,
        eval_duration=elapsed_ns,
    )


async def _stream_generate(
    llm: "object",
    request: GenerateRequest,
    model_name: str,
    start_ns: int,
) -> AsyncGenerator[bytes, None]:
    """Streaming Generate — liefert JSON-Lines im Ollama-Format."""
    token_count = 0

    try:
        # llama_cpp streaming ist synchron — wir wrappen es in einen Thread
        # und nutzen eine Queue als Brücke zwischen Thread und async Generator
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=50)
        loop = asyncio.get_event_loop()

        def _run_sync():
            try:
                for chunk in llm(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    echo=False,
                    stream=True,
                ):
                    text = chunk["choices"][0].get("text", "")
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # Sentinel
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, None)
                logger.error("stream error: %s", exc)

        asyncio.get_event_loop().run_in_executor(None, _run_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            token_count += 1
            chunk = json.dumps({
                "model": model_name,
                "response": token,
                "done": False,
            }) + "\n"
            yield chunk.encode()

        elapsed_ns = time.perf_counter_ns() - start_ns
        final = json.dumps({
            "model": model_name,
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed_ns,
            "eval_count": token_count,
            "eval_duration": elapsed_ns,
        }) + "\n"
        yield final.encode()

        logger.info(
            "stream done: model=%s tokens=%d elapsed_ms=%d",
            model_name, token_count, elapsed_ns // 1_000_000,
        )

    except Exception as exc:
        logger.exception("stream generate error: model=%s", model_name)
        error = json.dumps({"error": str(exc), "done": True}) + "\n"
        yield error.encode()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,          # Wichtig: 1 Worker — llama-cpp ist nicht fork-safe
        log_level=config.LOG_LEVEL.lower(),
    )
