from __future__ import annotations

import json
import tempfile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.responses import JSONResponse

import server


class DummyRequest:
    headers: dict[str, str] = {}


class DummySemaphore:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ServerContractTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        server._queue_depth = 0
        server.metrics = server.RuntimeMetrics()
        self.original_semaphore = server.manager.semaphore
        server.manager.semaphore = DummySemaphore()

    def tearDown(self) -> None:
        server.manager.semaphore = self.original_semaphore

    async def test_live_health_and_metrics_contract(self) -> None:
        live = await server.live()
        health = await server.health()
        metrics = await server.prometheus_metrics()

        self.assertEqual(live, {"status": "ok"})
        self.assertIn("loaded_models", health)
        self.assertIn("queue_depth", health)
        self.assertIn("airpi_requests_total", metrics.body.decode())
        self.assertIn("airpi_queue_depth", metrics.body.decode())

    async def test_ready_reports_not_ready_without_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server.config, "MODELS_DIR", tmpdir):
                with patch.object(server.config, "DEFAULT_MODEL", "missing.gguf"):
                    response = await server.ready()

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "not_ready")
        self.assertTrue(payload["models_dir_exists"])
        self.assertFalse(payload["default_model_exists"])

    async def test_generate_success_records_metrics(self) -> None:
        payload = {
            "choices": [{"text": "4", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "airpi": {"cache_hit": True},
        }
        with patch.object(server.manager, "generate", new=AsyncMock(return_value=payload)):
            response = await server.generate(
                server.GenerateRequest(model="test.gguf", prompt="2+2", stream=False),
                DummyRequest(),
            )

        self.assertEqual(response.response, "4")
        self.assertEqual(server.metrics.requests_total, 1)
        self.assertEqual(server.metrics.cache_hit_total, 1)
        self.assertEqual(server.metrics.tokens_total, 1)

    async def test_generate_model_not_found_error_contract(self) -> None:
        with patch.object(server.manager, "generate", new=AsyncMock(side_effect=FileNotFoundError("missing"))):
            with self.assertRaises(HTTPException) as raised:
                await server.generate(
                    server.GenerateRequest(model="missing.gguf", prompt="hello", stream=False),
                    DummyRequest(),
                )

        detail = raised.exception.detail["error"]
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(detail["code"], server.ErrorCode.MODEL_NOT_FOUND)
        self.assertEqual(detail["message"], "Model is not available")
        self.assertFalse(detail["retryable"])
        self.assertIn("request_id", detail)

    async def test_queue_full_error_contract(self) -> None:
        server._queue_depth = server.config.MAX_QUEUE
        with self.assertRaises(HTTPException) as raised:
            await server.generate(
                server.GenerateRequest(model="test.gguf", prompt="hello", stream=False),
                DummyRequest(),
            )

        detail = raised.exception.detail["error"]
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(detail["code"], server.ErrorCode.QUEUE_FULL)
        self.assertTrue(detail["retryable"])

    async def test_validation_payload_shape(self) -> None:
        payload = server._error_payload(
            server.ErrorCode.INVALID_REQUEST,
            "Request validation failed",
            False,
            "req-1",
        )

        self.assertEqual(payload["error"]["code"], server.ErrorCode.INVALID_REQUEST)
        self.assertFalse(payload["error"]["retryable"])
        self.assertEqual(payload["error"]["request_id"], "req-1")

    async def test_stream_model_not_found_uses_final_error_frame(self) -> None:
        async def failing_stream(**kwargs):
            raise FileNotFoundError("missing")
            yield "unreachable"

        with patch.object(server.manager, "stream_generate", new=failing_stream):
            chunks = [
                chunk async for chunk in server._stream_generate(
                    server.GenerateRequest(model="missing.gguf", prompt="hello", stream=True),
                    "missing.gguf",
                    0,
                    "req-1",
                )
            ]

        frame = chunks[-1].decode()
        self.assertIn(server.ErrorCode.MODEL_NOT_FOUND, frame)
        self.assertIn('"done": true', frame)
