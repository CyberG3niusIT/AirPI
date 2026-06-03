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
import dataclasses
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from model_manager import build_json_grammar, manager, select_model_for_prompt
from memory.manager import get_memory_manager
from memory.graph import GraphBuilder, merge_graph_overlays
from core.policy import PolicyEngine, Action
from core.router import ModelRouter, BackendType
from core.audit import AuditLogger, AuditRecord

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger("airpi")

_policy_engine = PolicyEngine()
_model_router = ModelRouter(default_local_model=config.DEFAULT_MODEL)
_audit_logger = AuditLogger(logger=logger)

_queue_depth: int = 0

# ── Graph TTL-Cache ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class _GraphCache:
    data: dict
    built_at: float   # time.monotonic()
    entry_count: int  # len(all_active()) zum Zeitpunkt des Builds

_graph_cache: _GraphCache | None = None
_GRAPH_CACHE_TTL: float = 30.0  # Sekunden


def _graph_cache_valid(entry_count: int) -> bool:
    """True wenn Cache ≤30s alt UND Eintragsanzahl unverändert (kein Write seit letztem Build)."""
    if _graph_cache is None:
        return False
    return (
        time.monotonic() - _graph_cache.built_at < _GRAPH_CACHE_TTL
        and _graph_cache.entry_count == entry_count
    )


def _invalidate_graph_cache() -> None:
    """Cache sofort invalidieren — wird nach Memory-Writes aufgerufen."""
    global _graph_cache
    _graph_cache = None


class ErrorCode:
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    QUEUE_FULL = "QUEUE_FULL"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
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
_server_start = time.monotonic()


def _parse_extraction(raw: str) -> list:
    """Robust JSON array extraction from LLM output (Fix A1).

    Tries three strategies before giving up:
    1. Direct json.loads of the stripped string.
    2. Non-greedy regex to find the first [...] block.
    3. Greedy regex for nested/multi-line arrays.
    Returns [] on any failure — never raises.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        pass
    # Strategy 2: non-greedy regex
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Strategy 3: greedy regex
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


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

_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
if os.path.isdir(_UI_DIR):
    app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")


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
    prompt: str = Field(min_length=1, max_length=config.MAX_PROMPT_CHARS)
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
    session_id: str | None = Field(default=None, max_length=config.MAX_SESSION_ID_LENGTH)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
            raise ValueError("session_id contains invalid characters")
        return value


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
    cache_hit_rate: float | None = None
    if metrics.requests_total > 0:
        cache_hit_rate = metrics.cache_hit_total / metrics.requests_total
    return {
        "status": "ok",
        **runtime,
        "queue_depth": _queue_depth,
        "max_queue": config.MAX_QUEUE,
        "uptime_seconds": round(time.monotonic() - _server_start, 2),
        "tokens_generated_total": metrics.tokens_total,
        "cache_hit_rate": cache_hit_rate,
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

    try:
        requested_model = request.preferred_model if request.preferred_model is not None else request.model
        model_name = select_model_for_prompt(request.prompt, requested_model)
    except ValueError as exc:
        metrics.record_error()
        raise _http_error(
            422,
            ErrorCode.INVALID_REQUEST,
            "Invalid model name",
            False,
            request_id,
        ) from exc

    # 1. Policy check
    policy_decision = _policy_engine.evaluate({"prompt": request.prompt, "model": model_name})

    # 2. Routing
    route = _model_router.route(policy_decision, {"model": model_name})

    # 3. Audit
    _audit_logger.log(AuditRecord(
        request_type="generate",
        policy_decision=policy_decision.action.value,
        backend_selected=route.backend.value,
        reason=route.reason,
        metadata={"prompt_length": len(request.prompt), "request_id": request_id}
    ))

    if route.blocked:
        metrics.record_error()
        raise _http_error(
            403,
            ErrorCode.INVALID_REQUEST,
            f"Request blocked by policy: {route.reason}",
            False,
            request_id,
        )

    # In Phase 2, we only support LOCAL backend execution.
    # We update the model_name to whatever the router decided, if applicable.
    model_name = route.model_name or model_name

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
    except asyncio.TimeoutError as exc:
        metrics.record_error()
        raise _http_error(
            504,
            ErrorCode.INFERENCE_TIMEOUT,
            "Inference timed out",
            True,
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
    wants_json = _wants_json_response(request)
    grammar = build_json_grammar(request.required_json_keys) if wants_json else None
    prompt = _json_prompt(request.prompt, request.required_json_keys) if wants_json else request.prompt
    result = await manager.generate(
        model_name=model_name,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        session_id=request.session_id,
        grammar=grammar,
    )

    elapsed_ns = time.perf_counter_ns() - start_ns
    choice = result["choices"][0]
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    response_text = choice.get("text", "")

    if wants_json:
        try:
            response_text = _normalize_json_response(response_text, request.required_json_keys)
        except ValueError:
            if grammar is not None:
                # Grammar war aktiv — invalides JSON sollte unmöglich sein
                logger.error(
                    "grammar-constrained output failed normalization: %r", response_text
                )
                raise HTTPException(
                    status_code=500,
                    detail=_error_payload(
                        ErrorCode.INFERENCE_FAILED,
                        "Grammar-constrained output is not valid JSON",
                        True,
                        "",
                    ),
                )
            # Grammar nicht verfügbar: Repair-Retry (Fallback)
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
        grammar = build_json_grammar(request.required_json_keys) if _wants_json_response(request) else None
        async with manager.semaphore:
            async for token in manager.stream_generate(
                model_name=model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                session_id=request.session_id,
                grammar=grammar,
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




# ── Memory Endpoints ─────────────────────────────────────────────────────────

class MemoryStoreRequest(BaseModel):
    content: str = Field(min_length=1)
    category: str = "fact"


class MemoryDeleteRequest(BaseModel):
    keyword: str = Field(min_length=1)


@app.post("/memory/store")
async def memory_store(request: MemoryStoreRequest) -> dict:
    """Speichert einen Fakt manuell."""
    try:
        get_memory_manager().store_fact(request.content, source="user", category=request.category)
        _invalidate_graph_cache()  # Graph-Cache invalidieren — neue Daten verfügbar
        return {"ok": True, "content": request.content}
    except Exception as exc:
        logger.exception("memory/store failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/memory/delete")
async def memory_delete(request: MemoryDeleteRequest) -> dict:
    """Markiert alle Eintraege mit keyword als inaktiv."""
    try:
        count = get_memory_manager().delete_fact(request.keyword)
        if count > 0:
            _invalidate_graph_cache()  # Graph-Cache invalidieren — Daten entfernt
        return {"ok": True, "deleted": count, "keyword": request.keyword}
    except Exception as exc:
        logger.exception("memory/delete failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/memory")
async def memory_get() -> dict:
    """Gibt den aktuellen Memory-Inhalt zurueck."""
    try:
        mem = get_memory_manager()
        # DB-Calls in Thread-Pool um Event Loop nicht zu blockieren
        entries = await asyncio.to_thread(mem.all_active)
        context = await asyncio.to_thread(mem.get_context)
        return {
            "content": context,
            "entries": entries,
        }
    except Exception as exc:
        logger.exception("memory GET failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

# ── Graph Endpoints ──────────────────────────────────────────────────────────

class GraphManualEdgeCreateRequest(BaseModel):
    source_key: str = Field(min_length=1, max_length=160)
    target_key: str = Field(min_length=1, max_length=160)
    source_label: str = Field(default="", max_length=160)
    target_label: str = Field(default="", max_length=160)
    relation_type: str = Field(default="manual", min_length=1, max_length=64)
    label: str = Field(default="", max_length=80)
    directed: bool = True
    confidence: int = Field(default=80, ge=0, le=100)
    note: str = Field(default="", max_length=500)

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_\- ]{1,64}", cleaned):
            raise ValueError("relation_type may only contain letters, numbers, spaces, '_' and '-'")
        return cleaned


class GraphManualEdgeUpdateRequest(BaseModel):
    relation_type: str | None = Field(default=None, min_length=1, max_length=64)
    label: str | None = Field(default=None, max_length=80)
    directed: bool | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    note: str | None = Field(default=None, max_length=500)
    source_label: str | None = Field(default=None, max_length=160)
    target_label: str | None = Field(default=None, max_length=160)

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_\- ]{1,64}", cleaned):
            raise ValueError("relation_type may only contain letters, numbers, spaces, '_' and '-'")
        return cleaned


class GraphAutoEdgeHideRequest(BaseModel):
    edge_key: str = Field(min_length=1, max_length=400)
    note: str = Field(default="", max_length=500)


class GraphEdgeOverrideClearRequest(BaseModel):
    edge_key: str = Field(min_length=1, max_length=400)
    action: str = Field(default="hide", pattern="^hide$")


class GraphManualNodeCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=160)
    type: str = Field(default="concept", pattern=r"^(person|place|tech|date|concept)$")
    note: str = Field(default="", max_length=500)
    confidence: int = Field(default=80, ge=0, le=100)


class GraphManualNodeUpdateRequest(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=160)
    type: str | None = Field(default=None, pattern=r"^(person|place|tech|date|concept)$")
    note: str | None = Field(default=None, max_length=500)
    confidence: int | None = Field(default=None, ge=0, le=100)


@app.get("/graph")
async def graph_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/graph.html", status_code=307)


@app.get("/graph/data")
async def graph_data() -> dict:
    global _graph_cache
    try:
        mem = get_memory_manager()
        # DB-Call in Thread-Pool um Event Loop nicht zu blockieren
        entries = await asyncio.to_thread(mem.all_active)

        # Cache-Hit: Graph innerhalb TTL und keine neuen Entries
        if _graph_cache_valid(len(entries)):
            return _graph_cache.data

        # Cache-Miss: Graph vollständig aufbauen (in Thread-Pool)
        graph = await asyncio.to_thread(GraphBuilder().build, entries)
        result = merge_graph_overlays(
            graph,
            manual_edges=mem.list_manual_edges(),
            manual_nodes=mem.list_manual_nodes(),
            edge_overrides=mem.list_edge_overrides(),
        )
        _graph_cache = _GraphCache(
            data=result,
            built_at=time.monotonic(),
            entry_count=len(entries),
        )
        return result
    except Exception as exc:
        logger.exception("graph/data failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/graph/manual-edges")
async def graph_manual_edges() -> dict:
    try:
        return {"edges": get_memory_manager().list_manual_edges()}
    except Exception as exc:
        logger.exception("graph/manual-edges failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/graph/manual-edges")
async def graph_manual_edge_create(request: GraphManualEdgeCreateRequest) -> dict:
    try:
        edge = get_memory_manager().add_manual_edge(**request.model_dump())
        if edge is None:
            raise HTTPException(status_code=400, detail={"error": "invalid manual edge"})
        return {"ok": True, "edge": edge}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/manual-edges POST failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.patch("/graph/manual-edges/{edge_id}")
async def graph_manual_edge_update(edge_id: int, request: GraphManualEdgeUpdateRequest) -> dict:
    try:
        edge = get_memory_manager().update_manual_edge(
            edge_id,
            request.model_dump(exclude_unset=True),
        )
        if edge is None:
            raise HTTPException(status_code=404, detail={"error": "manual edge not found"})
        return {"ok": True, "edge": edge}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/manual-edges PATCH failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.delete("/graph/manual-edges/{edge_id}")
async def graph_manual_edge_delete(edge_id: int) -> dict:
    try:
        deleted = get_memory_manager().delete_manual_edge(edge_id)
        if not deleted:
            raise HTTPException(status_code=404, detail={"error": "manual edge not found"})
        return {"ok": True, "deleted": edge_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/manual-edges DELETE failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/graph/auto-edges/hide")
async def graph_auto_edge_hide(request: GraphAutoEdgeHideRequest) -> dict:
    try:
        override = get_memory_manager().hide_auto_edge(request.edge_key, request.note)
        if override is None:
            raise HTTPException(status_code=400, detail={"error": "invalid edge override"})
        return {"ok": True, "override": override}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/auto-edges/hide failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/graph/nodes")
async def graph_nodes_list() -> dict:
    """Listet alle manuellen Graph-Knoten."""
    try:
        return {"nodes": get_memory_manager().list_manual_nodes()}
    except Exception as exc:
        logger.exception("graph/nodes GET failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/graph/nodes")
async def graph_node_create(request: GraphManualNodeCreateRequest) -> dict:
    """Erstellt einen manuellen Graph-Knoten."""
    try:
        node = get_memory_manager().add_manual_node(**request.model_dump())
        if node is None:
            raise HTTPException(status_code=400, detail={"error": "invalid manual node"})
        return {"ok": True, "node": node}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/nodes POST failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.patch("/graph/nodes/{node_id}")
async def graph_node_update(node_id: int, request: GraphManualNodeUpdateRequest) -> dict:
    """Aktualisiert einen manuellen Graph-Knoten."""
    try:
        node = get_memory_manager().update_manual_node(
            node_id,
            request.model_dump(exclude_unset=True),
        )
        if node is None:
            raise HTTPException(status_code=404, detail={"error": "manual node not found"})
        return {"ok": True, "node": node}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/nodes PATCH failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.delete("/graph/nodes/{node_id}")
async def graph_node_delete(node_id: int) -> dict:
    """Soft-löscht einen manuellen Graph-Knoten."""
    try:
        deleted = get_memory_manager().delete_manual_node(node_id)
        if not deleted:
            raise HTTPException(status_code=404, detail={"error": "manual node not found"})
        return {"ok": True, "deleted": node_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/nodes DELETE failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.delete("/graph/auto-edges/override")
async def graph_auto_edge_override_clear(request: GraphEdgeOverrideClearRequest) -> dict:
    try:
        cleared = get_memory_manager().clear_edge_override(request.edge_key, request.action)
        if not cleared:
            raise HTTPException(status_code=404, detail={"error": "edge override not found"})
        return {"ok": True, "cleared": request.edge_key, "action": request.action}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("graph/auto-edges/override DELETE failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# ── Chat Endpoint (/api/chat — Ollama-kompatibel) ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="")
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system: str = ""


class ChatResponse(BaseModel):
    model: str
    message: dict
    done: bool


def _apply_chatml(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _build_chat_prompt(request: ChatRequest) -> str:
    """Baut den vollständigen ChatML-Prompt — injiziert request.system als erste Message."""
    messages = list(request.messages)
    if request.system:
        messages = [ChatMessage(role="system", content=request.system)] + messages
    return _apply_chatml(messages)


async def _stream_chat(
    model_name: str,
    prompt: str,
    request: ChatRequest,
) -> AsyncGenerator[bytes, None]:
    """NDJSON-Token-Stream für /api/chat (UI erwartet chunk.message.content)."""
    global _queue_depth
    _queue_depth += 1
    token_count = 0
    start_ns = time.perf_counter_ns()
    try:
        async with manager.semaphore:
            async for token in manager.stream_generate(
                model_name=model_name,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                stop=["<|im_end|>", "<|im_start|>"],
            ):
                token_count += 1
                yield (json.dumps({
                    "model": model_name,
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }) + "\n").encode()

        elapsed_ns = time.perf_counter_ns() - start_ns
        yield (json.dumps({
            "model": model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "eval_count": token_count,
            "eval_duration": elapsed_ns,
        }) + "\n").encode()
        metrics.record_request(model_name, elapsed_ns / 1_000_000_000, token_count, False)

    except Exception as exc:
        metrics.record_error()
        logger.exception("api/chat stream failed: model=%s", model_name)
        yield (json.dumps({
            "model": model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "error": str(exc),
        }) + "\n").encode()
    finally:
        _queue_depth -= 1


@app.post("/api/chat", response_model=None)
async def api_chat(request: ChatRequest, http_request: Request) -> StreamingResponse | dict:
    global _queue_depth
    request_id = http_request.headers.get("x-request-id", str(uuid.uuid4()))

    if _queue_depth >= config.MAX_QUEUE:
        metrics.record_error()
        raise _http_error(503, ErrorCode.QUEUE_FULL,
                          f"Inference queue is full ({config.MAX_QUEUE}).", True, request_id)

    last_content = request.messages[-1].content if request.messages else ""
    preferred = request.model or None
    try:
        model_name = select_model_for_prompt(last_content, preferred)
    except ValueError as exc:
        raise _http_error(422, ErrorCode.INVALID_REQUEST, "Invalid model name",
                          False, request_id) from exc

    prompt = _build_chat_prompt(request)

    # 1. Policy check
    policy_decision = _policy_engine.evaluate({"prompt": prompt, "model": model_name})

    # 2. Routing
    route = _model_router.route(policy_decision, {"model": model_name})

    # 3. Audit
    _audit_logger.log(AuditRecord(
        request_type="chat",
        policy_decision=policy_decision.action.value,
        backend_selected=route.backend.value,
        reason=route.reason,
        metadata={"messages_count": len(request.messages), "request_id": request_id}
    ))

    if route.blocked:
        metrics.record_error()
        raise _http_error(
            403,
            ErrorCode.INVALID_REQUEST,
            f"Request blocked by policy: {route.reason}",
            False,
            request_id,
        )

    model_name = route.model_name or model_name

    # Streaming: UI sendet stream=true — gibt NDJSON-Chunks zurück
    if request.stream:
        return StreamingResponse(
            _stream_chat(model_name, prompt, request),
            media_type="application/x-ndjson",
        )

    # Nicht-Streaming: Semaphore schützt vor Concurrency-Korruption im LLM
    _queue_depth += 1
    start_ns = time.perf_counter_ns()
    try:
        async with manager.semaphore:
            result = await manager.generate(
                model_name=model_name,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                stop=["<|im_end|>", "<|im_start|>"],
            )
    except FileNotFoundError as exc:
        metrics.record_error()
        raise _http_error(404, ErrorCode.MODEL_NOT_FOUND, "Model is not available",
                          False, request_id) from exc
    except Exception as exc:
        metrics.record_error()
        logger.exception("api/chat inference failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    finally:
        _queue_depth -= 1

    elapsed_ns = time.perf_counter_ns() - start_ns
    choices = result.get("choices", [{}])
    text = choices[0].get("text", "") if choices else ""
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    metrics.record_request(model_name, elapsed_ns / 1_000_000_000, tokens, False)
    return {
        "model": model_name,
        "message": {"role": "assistant", "content": text},
        "done": True,
        "eval_count": tokens,
        "eval_duration": elapsed_ns,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        log_level=config.LOG_LEVEL.lower(),
    )
