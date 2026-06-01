from __future__ import annotations

import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from model_manager import (
    ModelManager,
    _is_recoverable_generation_error,
    resolve_model_path,
    select_model_for_prompt,
    validate_model_name,
)


class RecoverableGenerationErrorTests(IsolatedAsyncioTestCase):
    def test_select_model_resolves_fast_lane_alias(self) -> None:
        with patch("model_manager.FAST_MODEL", "fast-model.gguf"):
            self.assertEqual(select_model_for_prompt("short answer", "fast"), "fast-model.gguf")
            self.assertEqual(select_model_for_prompt("short answer", "airpi-fast"), "fast-model.gguf")

    def test_select_model_resolves_builtin_aliases(self) -> None:
        with patch("model_manager.DEFAULT_MODEL", "default-model.gguf"):
            with patch("model_manager.LARGE_MODEL", "large-model.gguf"):
                self.assertEqual(select_model_for_prompt("short answer", "default"), "default-model.gguf")
                self.assertEqual(select_model_for_prompt("short answer", "large"), "large-model.gguf")

    def test_select_model_keeps_explicit_model_name(self) -> None:
        self.assertEqual(select_model_for_prompt("short answer", "custom.gguf"), "custom.gguf")

    def test_select_model_rejects_path_like_or_non_gguf_names(self) -> None:
        invalid_names = [
            "../secret.gguf",
            "/tmp/model.gguf",
            "nested/model.gguf",
            "nested\\model.gguf",
            "model.bin",
            " model.gguf",
            "",
        ]

        for invalid_name in invalid_names:
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(ValueError):
                    select_model_for_prompt("short answer", invalid_name)

    def test_validate_model_name_accepts_plain_gguf_basename(self) -> None:
        self.assertEqual(validate_model_name("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"), "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

    def test_resolve_model_path_rejects_symlink_escape(self) -> None:
        with TemporaryDirectory() as models_dir:
            with TemporaryDirectory() as outside_dir:
                outside_model = Path(outside_dir) / "outside.gguf"
                outside_model.write_bytes(b"gguf")
                (Path(models_dir) / "linked.gguf").symlink_to(outside_model)

                with self.assertRaises(ValueError):
                    resolve_model_path("linked.gguf", models_dir)

    def test_recoverable_generation_error_detection(self) -> None:
        self.assertTrue(
            _is_recoverable_generation_error(ValueError("could not broadcast input array from shape (1,) into shape (2,)"))
        )
        self.assertTrue(_is_recoverable_generation_error(ValueError("shape mismatch in llama_cpp")))
        self.assertFalse(_is_recoverable_generation_error(RuntimeError("shape mismatch in llama_cpp")))
        self.assertFalse(_is_recoverable_generation_error(ValueError("some other error")))

    async def test_generate_retries_after_recoverable_failure(self) -> None:
        manager = ModelManager()
        manager.get = AsyncMock(side_effect=["llm-v1", "llm-v2"])
        manager._invalidate_model_state = AsyncMock()
        manager._is_session_valid = lambda session_id, model_name: False
        manager._touch_session = AsyncMock()

        success_payload = {
            "choices": [{"text": "recovered", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

        with patch("model_manager.asyncio.to_thread", new=AsyncMock(side_effect=[
            ValueError("could not broadcast input array from shape (1,) into shape (2,)"),
            success_payload,
        ])) as to_thread:
            result = await manager.generate(
                model_name="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
                prompt="hello",
                max_tokens=16,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>"],
                session_id=None,
            )

        self.assertEqual(result, success_payload)
        self.assertEqual(manager.get.await_count, 2)
        self.assertEqual(to_thread.await_count, 2)
        manager._invalidate_model_state.assert_awaited_once_with("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
        manager._touch_session.assert_not_awaited()

    async def test_stream_generate_retries_after_recoverable_failure(self) -> None:
        manager = ModelManager()
        manager.get = AsyncMock(side_effect=["llm-v1", "llm-v2"])
        manager._invalidate_model_state = AsyncMock()
        manager._is_session_valid = lambda session_id, model_name: False
        manager._touch_session = AsyncMock()

        calls = []

        def fake_stream_with_cache(llm, prompt, max_tokens, temperature, top_p, stop, reset, queue, loop, grammar=None):
            calls.append((llm, reset))
            if len(calls) == 1:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ValueError("could not broadcast input array from shape (1,) into shape (2,)"),
                )
            else:
                loop.call_soon_threadsafe(queue.put_nowait, "recovered")
            loop.call_soon_threadsafe(queue.put_nowait, None)

        async def inline_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("model_manager._stream_with_cache", new=fake_stream_with_cache):
            with patch("model_manager.asyncio.to_thread", new=inline_to_thread):
                chunks = [
                    chunk
                    async for chunk in manager.stream_generate(
                        model_name="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
                        prompt="hello",
                        max_tokens=16,
                        temperature=0.7,
                        top_p=0.9,
                        stop=["</s>"],
                        session_id=None,
                    )
                ]

        self.assertEqual(chunks, ["recovered"])
        self.assertEqual(manager.get.await_count, 2)
        self.assertEqual(calls, [("llm-v1", True), ("llm-v2", True)])
        manager._invalidate_model_state.assert_awaited_once_with("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")

    async def test_invalidate_model_state_removes_only_target_model(self) -> None:
        manager = ModelManager()
        manager._instances["target-model"] = ("llm-a", 1.0)
        manager._instances["other-model"] = ("llm-b", 2.0)
        manager._sessions["session-a"] = ("target-model", 1.0)
        manager._sessions["session-b"] = ("other-model", 2.0)

        await manager._invalidate_model_state("target-model")

        self.assertNotIn("target-model", manager._instances)
        self.assertIn("other-model", manager._instances)
        self.assertNotIn("session-a", manager._sessions)
        self.assertIn("session-b", manager._sessions)

    def test_load_passes_pi_tuning_options_to_llama(self) -> None:
        manager = ModelManager()
        calls = []

        class FakeLlama:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_llama_module = types.SimpleNamespace(Llama=FakeLlama)

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            model_path.write_bytes(b"gguf")

            with patch.dict(sys.modules, {"llama_cpp": fake_llama_module}):
                with patch("model_manager.MODELS_DIR", tmpdir):
                    with patch("model_manager.N_CTX_SMALL", 2048):
                        with patch("model_manager.N_THREADS", 3):
                            with patch("model_manager.N_THREADS_BATCH", 4):
                                with patch("model_manager.N_BATCH_SMALL", 1024):
                                    with patch("model_manager.N_UBATCH_SMALL", 512):
                                        with patch("model_manager.FLASH_ATTN", True):
                                            manager._load(model_path.name)

        self.assertEqual(calls[0]["model_path"], str(model_path))
        self.assertEqual(calls[0]["n_ctx"], 2048)
        self.assertEqual(calls[0]["n_threads"], 3)
        self.assertEqual(calls[0]["n_threads_batch"], 4)
        self.assertEqual(calls[0]["n_batch"], 1024)
        self.assertEqual(calls[0]["n_ubatch"], 512)
        self.assertTrue(calls[0]["flash_attn"])
