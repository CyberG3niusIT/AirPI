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
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from model_manager import manager, select_model_for_prompt

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger("airpi")

_queue_depth: int = 0


class ErrorCode:
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    QUEUE_FULL = "QUEUE_FULL"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    MODEL_RECOVERED = "MODEL_RECOVERED"
    SERVICE_NOT_READY = "SERVICE_NOT_READY"


@dataclass
class RuntimeMetrics:
    requests_total: int = 0
    request_errors_total: int = 0
    tokens_total: int = 0
    request_duration_seconds_total: float = 0.0
    cache_hit_total: int = 0
    generated_by_model: dict[str, int] = field(default_factory=dict)

    def record_request(self, model_name: str, duration_seconds: float, tokens: int, cache_hit: bool) -> None:
        self.requests_total += 1
        self.tokens_total += tokens
        self.request_duration_seconds_total += duration_seconds
        if cache_hit:
            self.cache_hit_total += 1
        self.generated_by_model[model_name] = self.generated_by_model.get(model_name, 0) + 1

    def record_error(self) -> None:
        self.request_errors_total += 1


metrics = RuntimeMetrics()


def _error_payload(code: str, message: str, retryable: bool, request_id: str) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "request_id": request_id,
        }
    }


def _http_error(status_code: int, code: str, message: str, retryable: bool, request_id: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=_error_payload(code, message, retryable, request_id),
    )


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    metrics.record_error()
    return JSONResponse(
        status_code=422,
        content={"detail": _error_payload(
            ErrorCode.INVALID_REQUEST,
            "Request validation failed",
            False,
            request_id,
        )},
    )


# ── Request / Response Models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    format: str | None = None
    response_format: str | None = None
    required_json_keys: list[str] | None = None
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


def _wants_json_response(request: GenerateRequest) -> bool:
    return request.format == "json" or request.response_format == "json_object"


def _json_prompt(prompt: str, required_keys: list[str] | None) -> str:
    key_text = ""
    if required_keys:
        key_text = " Required keys: " + ", ".join(required_keys) + "."
    return (
        "Return exactly one valid JSON object. No markdown. No prose."
        f"{key_text}\n\n"
        f"{prompt}"
    )


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and not stripped[index + end:].strip():
            return parsed
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No valid JSON object found")


def _normalize_json_response(text: str, required_keys: list[str] | None) -> str:
    parsed = _extract_json_object(text)
    missing = [key for key in (required_keys or []) if key not in parsed]
    if missing:
        raise ValueError("Missing required JSON keys: " + ", ".join(missing))
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _repair_json_prompt(raw_text: str, required_keys: list[str] | None) -> str:
    key_text = ""
    if required_keys:
        key_text = " Required keys: " + ", ".join(required_keys) + "."
    return (
        "Convert the following text to exactly one valid JSON object."
        " No markdown. No prose."
        f"{key_text}\n\n"
        f"{raw_text}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/live")
async def live() -> dict:
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    runtime = manager.runtime_status()
    return {
        "status": "ok",
        **runtime,
        "queue_depth": _queue_depth,
        "max_queue": config.MAX_QUEUE,
    }


@app.get("/ready")
async def ready() -> dict:
    models_dir_exists = os.path.isdir(config.MODELS_DIR)
    default_model_path = os.path.join(config.MODELS_DIR, config.DEFAULT_MODEL)
    fast_model_path = os.path.join(config.MODELS_DIR, config.FAST_MODEL)
    default_model_exists = os.path.isfile(default_model_path)
    fast_model_exists = os.path.isfile(fast_model_path)
    ready_state = models_dir_exists and default_model_exists and fast_model_exists and _queue_depth < config.MAX_QUEUE
    payload = {
        "status": "ready" if ready_state else "not_ready",
        "models_dir_exists": models_dir_exists,
        "default_model_exists": default_model_exists,
        "fast_model_exists": fast_model_exists,
        "queue_depth": _queue_depth,
        "max_queue": config.MAX_QUEUE,
        "runtime": manager.runtime_status(),
    }
    if not ready_state:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    runtime = manager.runtime_status()
    lines = [
        "# HELP airpi_requests_total Total generate requests.",
        "# TYPE airpi_requests_total counter",
        f"airpi_requests_total {metrics.requests_total}",
        "# HELP airpi_request_errors_total Total generate request errors.",
        "# TYPE airpi_request_errors_total counter",
        f"airpi_request_errors_total {metrics.request_errors_total}",
        "# HELP airpi_request_duration_seconds_total Total generate request duration.",
        "# TYPE airpi_request_duration_seconds_total counter",
        f"airpi_request_duration_seconds_total {metrics.request_duration_seconds_total:.6f}",
        "# HELP airpi_tokens_total Total generated completion tokens.",
        "# TYPE airpi_tokens_total counter",
        f"airpi_tokens_total {metrics.tokens_total}",
        "# HELP airpi_cache_hit_total Total session cache hits.",
        "# TYPE airpi_cache_hit_total counter",
        f"airpi_cache_hit_total {metrics.cache_hit_total}",
        "# HELP airpi_queue_depth Current queued generate requests.",
        "# TYPE airpi_queue_depth gauge",
        f"airpi_queue_depth {_queue_depth}",
        "# HELP airpi_active_sessions Current active session count.",
        "# TYPE airpi_active_sessions gauge",
        f"airpi_active_sessions {runtime['active_sessions']}",
        "# HELP airpi_loaded_models Current loaded model count.",
        "# TYPE airpi_loaded_models gauge",
        f"airpi_loaded_models {len(runtime['loaded_models'])}",
        "# HELP airpi_recovery_total Total model recoveries.",
        "# TYPE airpi_recovery_total counter",
        f"airpi_recovery_total {runtime['recovery_count']}",
        "# HELP airpi_eviction_total Total model evictions.",
        "# TYPE airpi_eviction_total counter",
        f"airpi_eviction_total {runtime['eviction_count']}",
    ]
    for model_name, count in sorted(metrics.generated_by_model.items()):
        escaped = model_name.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'airpi_model_requests_total{{model="{escaped}"}} {count}')
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


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
async def generate(request: GenerateRequest, http_request: Request) -> StreamingResponse | GenerateResponse:
    global _queue_depth
    request_id = http_request.headers.get("x-request-id", str(uuid.uuid4()))

    if _queue_depth >= config.MAX_QUEUE:
        metrics.record_error()
        raise _http_error(
            503,
            ErrorCode.QUEUE_FULL,
            f"Inference queue is full ({config.MAX_QUEUE}).",
            True,
            request_id,
        )

    model_name = select_model_for_prompt(request.prompt, request.preferred_model or request.model)
    _queue_depth += 1
    start_ns = time.perf_counter_ns()

    if request.stream:
        return StreamingResponse(
            _stream_generate(request, model_name, start_ns, request_id),
            media_type="application/x-ndjson",
        )

    try:
        async with manager.semaphore:
            return await _blocking_generate(request, model_name, start_ns)

    except FileNotFoundError as exc:
        metrics.record_error()
        raise _http_error(
            404,
            ErrorCode.MODEL_NOT_FOUND,
            "Model is not available",
            False,
            request_id,
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        metrics.record_error()
        logger.exception("generate error: model=%s", model_name)
        raise _http_error(
            500,
            ErrorCode.INFERENCE_FAILED,
            "Inference failed",
            True,
            request_id,
        ) from exc
    finally:
        _queue_depth -= 1


async def _blocking_generate(
    request: GenerateRequest,
    model_name: str,
    start_ns: int,
) -> GenerateResponse:
    prompt = _json_prompt(request.prompt, request.required_json_keys) if _wants_json_response(request) else request.prompt
    result = await manager.generate(
        model_name=model_name,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        session_id=request.session_id,
    )

    elapsed_ns = time.perf_counter_ns() - start_ns
    choice = result["choices"][0]
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    response_text = choice.get("text", "")

    if _wants_json_response(request):
        try:
            response_text = _normalize_json_response(response_text, request.required_json_keys)
        except ValueError:
            repair = await manager.generate(
                model_name=model_name,
                prompt=_repair_json_prompt(response_text, request.required_json_keys),
                max_tokens=request.max_tokens,
                temperature=0.0,
                top_p=1.0,
                stop=request.stop,
                session_id=None,
            )
            repair_choice = repair["choices"][0]
            repair_tokens = repair.get("usage", {}).get("completion_tokens", 0)
            tokens += repair_tokens
            response_text = _normalize_json_response(repair_choice.get("text", ""), request.required_json_keys)

    cache_hit = bool(result.get("airpi", {}).get("cache_hit"))
    metrics.record_request(model_name, elapsed_ns / 1_000_000_000, tokens, cache_hit)

    logger.info("blocking done: model=%s tokens=%d elapsed_ms=%d", model_name, tokens, elapsed_ns // 1_000_000)

    return GenerateResponse(
        model=model_name,
        response=response_text,
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
    request_id: str,
) -> AsyncGenerator[bytes, None]:
    global _queue_depth
    token_count = 0
    try:
        async with manager.semaphore:
            async for token in manager.stream_generate(
                model_name=model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                session_id=request.session_id,
            ):
                token_count += 1
                yield (json.dumps({"model": model_name, "response": token, "done": False}) + "\n").encode()

        elapsed_ns = time.perf_counter_ns() - start_ns
        metrics.record_request(model_name, elapsed_ns / 1_000_000_000, token_count, False)
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

    except FileNotFoundError:
        metrics.record_error()
        yield (json.dumps({
            "model": model_name,
            "response": "",
            "done": True,
            "error": _error_payload(
                ErrorCode.MODEL_NOT_FOUND,
                "Model is not available",
                False,
                request_id,
            )["error"],
        }) + "\n").encode()
    except Exception as exc:
        metrics.record_error()
        logger.exception("stream error: model=%s", model_name)
        yield (json.dumps({
            "model": model_name,
            "response": "",
            "done": True,
            "error": _error_payload(
                ErrorCode.INFERENCE_FAILED,
                "Inference failed",
                True,
                request_id,
            )["error"],
        }) + "\n").encode()
    finally:
        _queue_depth -= 1


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        log_level=config.LOG_LEVEL.lower(),
    )
