"""Tests für /api/chat, /v1/chat/completions, /api/ps, /api/show."""

from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

import server
try:
    from server import (
        ChatMessage,
        ChatRequest,
        ChatResponse,
        OpenAIChatRequest,
        OpenAIMessage,
        ShowRequest,
    )
except ImportError:
    pytest.skip("Chat API not yet implemented in server.py", allow_module_level=True)


class DummyRequest:
    headers: dict = {}


def _make_chat_result(text: str = "Hello there!") -> dict:
    return {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "airpi": {"cache_hit": False},
    }


class ChatEndpointTests(IsolatedAsyncioTestCase):

    async def test_chat_returns_assistant_message(self) -> None:
        result = _make_chat_result("Hi!")
        mock_llm = MagicMock()
        mock_llm.metadata = {}  # kein Template → ChatML-Fallback
        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=AsyncMock(return_value=result)):
                response = await server.chat(
                    ChatRequest(
                        model="test.gguf",
                        messages=[ChatMessage(role="user", content="Hello")],
                        stream=False,
                    ),
                    DummyRequest(),
                )
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.message["role"], "assistant")
        self.assertEqual(response.message["content"], "Hi!")
        self.assertTrue(response.done)

    async def test_chat_applies_chatml_fallback_when_no_template(self) -> None:
        """Ohne GGUF-Template → ChatML-Format an generate() übergeben."""
        result = _make_chat_result("ok")
        mock_llm = MagicMock()
        mock_llm.metadata = {}
        captured: list[str] = []

        async def capture_generate(**kwargs):
            captured.append(kwargs["prompt"])
            return result

        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=capture_generate):
                await server.chat(
                    ChatRequest(
                        model="test.gguf",
                        messages=[
                            ChatMessage(role="system", content="You are helpful."),
                            ChatMessage(role="user", content="Hi"),
                        ],
                        stream=False,
                    ),
                    DummyRequest(),
                )

        self.assertEqual(len(captured), 1)
        prompt = captured[0]
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    async def test_chat_uses_jinja2_template_when_available(self) -> None:
        """Mit GGUF-Template → Jinja2ChatFormatter wird verwendet."""
        result = _make_chat_result("ok")
        # Minimales ChatML-artiges Template (Qwen-Stil)
        template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        mock_llm = MagicMock()
        mock_llm.metadata = {
            "tokenizer.chat_template": template,
            "tokenizer.ggml.eos_token_id": "151645",
        }
        mock_llm.detokenize = MagicMock(return_value=b"<|im_end|>")
        captured: list[str] = []

        async def capture_generate(**kwargs):
            captured.append(kwargs["prompt"])
            return result

        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=capture_generate):
                await server.chat(
                    ChatRequest(
                        model="test.gguf",
                        messages=[ChatMessage(role="user", content="Hello")],
                        stream=False,
                    ),
                    DummyRequest(),
                )

        self.assertEqual(len(captured), 1)
        prompt = captured[0]
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("Hello", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    async def test_chat_rejects_invalid_model_name(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await server.chat(
                ChatRequest(
                    model="../etc/passwd.gguf",
                    messages=[ChatMessage(role="user", content="hi")],
                ),
                DummyRequest(),
            )
        self.assertEqual(raised.exception.status_code, 422)

    async def test_chat_queue_full_returns_503(self) -> None:
        server._queue_depth = server.config.MAX_QUEUE
        try:
            with self.assertRaises(HTTPException) as raised:
                await server.chat(
                    ChatRequest(
                        model="test.gguf",
                        messages=[ChatMessage(role="user", content="hi")],
                    ),
                    DummyRequest(),
                )
            self.assertEqual(raised.exception.status_code, 503)
        finally:
            server._queue_depth = 0

    async def test_chat_model_not_found_returns_404(self) -> None:
        mock_llm = MagicMock()
        mock_llm.metadata = {}
        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=AsyncMock(side_effect=FileNotFoundError)):
                with self.assertRaises(HTTPException) as raised:
                    await server.chat(
                        ChatRequest(
                            model="missing.gguf",
                            messages=[ChatMessage(role="user", content="hi")],
                        ),
                        DummyRequest(),
                    )
        self.assertEqual(raised.exception.status_code, 404)


class OpenAICompatTests(IsolatedAsyncioTestCase):

    async def test_openai_chat_completions_maps_to_chat(self) -> None:
        result = _make_chat_result("Hello from OpenAI compat!")
        mock_llm = MagicMock()
        mock_llm.metadata = {}
        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=AsyncMock(return_value=result)):
                response = await server.openai_chat(
                    OpenAIChatRequest(
                        model="test.gguf",
                        messages=[OpenAIMessage(role="user", content="hi")],
                    ),
                    DummyRequest(),
                )
        from starlette.responses import JSONResponse
        self.assertIsInstance(response, JSONResponse)
        body = json.loads(response.body)
        self.assertIn("choices", body)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertIn("Hello from OpenAI compat!", body["choices"][0]["message"]["content"])
        self.assertIn("id", body)
        self.assertEqual(body["object"], "chat.completion")

    async def test_openai_chat_handles_string_stop(self) -> None:
        """stop kann String oder Liste sein — beide Varianten müssen funktionieren."""
        result = _make_chat_result("ok")
        mock_llm = MagicMock()
        mock_llm.metadata = {}
        with patch.object(server.manager, "get", new=AsyncMock(return_value=mock_llm)):
            with patch.object(server.manager, "generate", new=AsyncMock(return_value=result)):
                response = await server.openai_chat(
                    OpenAIChatRequest(
                        model="test.gguf",
                        messages=[OpenAIMessage(role="user", content="hi")],
                        stop="</s>",
                    ),
                    DummyRequest(),
                )
        self.assertIsNotNone(response)


class ApiStatusEndpointTests(IsolatedAsyncioTestCase):

    async def test_api_ps_returns_loaded_models(self) -> None:
        with patch.object(type(server.manager), "loaded_models", new_callable=PropertyMock) as mock_lm:
            mock_lm.return_value = ["model-a.gguf", "model-b.gguf"]
            response = await server.list_running_models()
        self.assertIn("models", response)
        names = [m["name"] for m in response["models"]]
        self.assertIn("model-a.gguf", names)
        self.assertIn("model-b.gguf", names)

    async def test_api_ps_returns_empty_when_nothing_loaded(self) -> None:
        with patch.object(type(server.manager), "loaded_models", new_callable=PropertyMock) as mock_lm:
            mock_lm.return_value = []
            response = await server.list_running_models()
        self.assertEqual(response["models"], [])

    async def test_api_show_returns_model_details(self) -> None:
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = pathlib.Path(tmpdir) / "test-4b-q4_k_m.gguf"
            model_file.write_bytes(b"fake")
            with patch("server.resolve_model_path", return_value=model_file):
                response = await server.show_model(ShowRequest(name="test-4b-q4_k_m.gguf"))
        self.assertIn("details", response)
        self.assertEqual(response["details"]["format"], "gguf")
        self.assertEqual(response["details"]["quantization_level"], "Q4_K_M")
        self.assertIn("modelfile", response)

    async def test_api_show_rejects_invalid_model_name(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await server.show_model(ShowRequest(name="../etc/passwd"))
        self.assertEqual(raised.exception.status_code, 404)


class ApplyChatTemplateTests(IsolatedAsyncioTestCase):

    def test_chatml_fallback_includes_all_roles(self) -> None:
        from model_manager import apply_chat_template
        mock_llm = MagicMock()
        mock_llm.metadata = {}
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = apply_chat_template(mock_llm, messages)
        self.assertIn("<|im_start|>system\nBe helpful.<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant\nHi there!<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant\n", prompt.split("<|im_end|>")[-1])

    def test_chatml_fallback_used_when_jinja2_fails(self) -> None:
        from model_manager import apply_chat_template
        mock_llm = MagicMock()
        mock_llm.metadata = {"tokenizer.chat_template": "{{ INVALID JINJA }}"}
        mock_llm.detokenize = MagicMock(return_value=b"</s>")
        messages = [{"role": "user", "content": "test"}]
        # Sollte nicht werfen — fällt auf ChatML-Fallback zurück
        prompt = apply_chat_template(mock_llm, messages)
        self.assertIn("test", prompt)
        self.assertIn("<|im_start|>", prompt)
