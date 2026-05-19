from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from model_manager import ModelManager, _is_recoverable_generation_error


class RecoverableGenerationErrorTests(IsolatedAsyncioTestCase):
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
                model_name="qwen2.5-coder-1.5b-q4_k_m.gguf",
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
        manager._invalidate_model_state.assert_awaited_once_with("qwen2.5-coder-1.5b-q4_k_m.gguf")
        manager._touch_session.assert_not_awaited()

    async def test_stream_generate_retries_after_recoverable_failure(self) -> None:
        manager = ModelManager()
        manager.get = AsyncMock(side_effect=["llm-v1", "llm-v2"])
        manager._invalidate_model_state = AsyncMock()
        manager._is_session_valid = lambda session_id, model_name: False
        manager._touch_session = AsyncMock()

        calls = []

        def fake_stream_with_cache(llm, prompt, max_tokens, temperature, top_p, stop, reset, queue, loop):
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
                        model_name="qwen2.5-coder-1.5b-q4_k_m.gguf",
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
        manager._invalidate_model_state.assert_awaited_once_with("qwen2.5-coder-1.5b-q4_k_m.gguf")

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
