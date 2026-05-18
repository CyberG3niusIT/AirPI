"""AirPI v2 — Stateful Session Inference Server für Raspberry Pi 5.

Neu in v2:
- session_id in /api/generate: KV-Cache-Reuse zwischen aufeinanderfolgenden
  Requests (z.B. Multi-Step-Agent-Runs) — spart 70-90% Prefill-Compute
- Speculative Decoding: LlamaPromptLookupDecoding oder 1.5B-Draft-Target
- active_sessions im /health-Endpoint sichtbar

Port: 11435 (Ollama: 11434)
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

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger("airpi")

_queue_depth: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("AirPI v2 starting on %s:%d", config.HOST, config.PORT)
    await manager.warmup(config.DEFAULT_MODEL)

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


app = FastAPI(
    title="AirPI Inference Server",
    description="Ollama-kompatibler LLM Inference Server für Raspberry Pi 5",
    version="2.0.0",
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
    keep_alive: str | None = None
    preferred_model: str | None = None
    # KV-Cache-Reuse: gleiche session_id → reset=False → nur neue Token werden verarbeitet
    session_id: str | None = None


class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "loaded_models": manager.loaded_models,
        "active_sessions": manager.active_sessions,
    }


@app.get("/api/tags")
async def list_models() -> dict:
    models = []
    if os.path.isdir(config.MODELS_DIR):
        for fname in os.listdir(config.MODELS_DIR):
            if fname.endswith(".gguf"):
                fpath = os.path.join(config.MODELS_DIR, fname)
                stat = os.stat(fpath)
                models.append({
                    "name": fname,
                    "model": fname,
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                    "size": stat.st_size,
                    "digest": "",
                    "details": {"format": "gguf", "loaded": fname in manager.loaded_models},
                })
    return {"models": models}


@app.post("/api/generate", response_model=None)
async def generate(request: GenerateRequest) -> StreamingResponse | GenerateResponse:
    global _queue_depth

    if _queue_depth >= config.MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "inference_queue_full",
                "message": f"Maximale Queue-Tiefe ({config.MAX_QUEUE}) erreicht.",
            },
        )

    model_name = select_model_for_prompt(request.prompt, request.preferred_model or request.model)
    _queue_depth += 1
    start_ns = time.perf_counter_ns()

    try:
        async with manager.semaphore:
            if request.stream:
                return StreamingResponse(
                    _stream_generate(request, model_name, start_ns),
                    media_type="application/x-ndjson",
                )
            else:
                return await _blocking_generate(request, model_name, start_ns)

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
    request: GenerateRequest,
    model_name: str,
    start_ns: int,
) -> GenerateResponse:
    result = await manager.generate(
        model_name=model_name,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        session_id=request.session_id,
    )

    elapsed_ns = time.perf_counter_ns() - start_ns
    choice = result["choices"][0]
    tokens = result.get("usage", {}).get("completion_tokens", 0)

    logger.info("blocking done: model=%s tokens=%d elapsed_ms=%d", model_name, tokens, elapsed_ns // 1_000_000)

    return GenerateResponse(
        model=model_name,
        response=choice.get("text", ""),
        done=True,
        done_reason=choice.get("finish_reason", "stop"),
        total_duration=elapsed_ns,
        eval_count=tokens,
        eval_duration=elapsed_ns,
    )


async def _stream_generate(
    request: GenerateRequest,
    model_name: str,
    start_ns: int,
) -> AsyncGenerator[bytes, None]:
    token_count = 0
    try:
        queue = await manager.stream_generate(
            model_name=model_name,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            session_id=request.session_id,
        )

        while True:
            token = await queue.get()
            if token is None:
                break
            token_count += 1
            yield (json.dumps({"model": model_name, "response": token, "done": False}) + "\n").encode()

        elapsed_ns = time.perf_counter_ns() - start_ns
        yield (json.dumps({
            "model": model_name,
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed_ns,
            "eval_count": token_count,
            "eval_duration": elapsed_ns,
        }) + "\n").encode()

        logger.info("stream done: model=%s tokens=%d elapsed_ms=%d", model_name, token_count, elapsed_ns // 1_000_000)

    except Exception as exc:
        logger.exception("stream error: model=%s", model_name)
        yield (json.dumps({"error": str(exc), "done": True}) + "\n").encode()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        log_level=config.LOG_LEVEL.lower(),
    )
