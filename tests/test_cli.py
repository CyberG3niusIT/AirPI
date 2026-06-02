"""Tests for CLI QW-9 (flags) and QW-10 (multiline input)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from click.testing import CliRunner

# Make sure we can import cli.main from the worktree
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.main import (
    cli,
    _read_multiline_input,
    _open_editor_input,
    _print_plain_status,
    _build_status_table,
)


# ── QW-9: chat flags ──────────────────────────────────────────────────────────

class TestChatSystemFlag(unittest.TestCase):

    def _mock_stream(self, *args, **kwargs):
        """Yield one token then stats."""
        yield ("Hello!", None)
        yield ("", {"eval_count": 1, "eval_duration": 1_000_000_000})

    @patch("cli.main._resolve_model", return_value="test.gguf")
    @patch("cli.main._stream_chat")
    def test_system_flag_sets_prompt(self, mock_stream, mock_resolve):
        """--system sets the system message in the chat history."""
        mock_stream.side_effect = self._mock_stream
        runner = CliRunner()

        captured_messages: list = []

        def capturing_stream(url, model, messages, session):
            captured_messages.extend(messages)
            yield ("Hi!", None)
            yield ("", {"eval_count": 1, "eval_duration": 1_000_000_000})

        mock_stream.side_effect = capturing_stream

        result = runner.invoke(
            cli,
            ["chat", "--system", "Du bist ein Python-Experte", "--url", "http://fake"],
            input="Hello\nexit\n",
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            any(
                m.get("role") == "system" and "Python-Experte" in m.get("content", "")
                for m in captured_messages
            ),
            f"System prompt not found in messages: {captured_messages}",
        )

    @patch("cli.main._resolve_model", return_value="test.gguf")
    @patch("cli.main._stream_chat")
    def test_system_file_flag_loads_prompt(self, mock_stream, mock_resolve):
        """--system-file loads system prompt from a file."""
        captured_messages: list = []

        def capturing_stream(url, model, messages, session):
            captured_messages.extend(messages)
            yield ("Hi!", None)
            yield ("", {"eval_count": 1, "eval_duration": 1_000_000_000})

        mock_stream.side_effect = capturing_stream

        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("prompt.txt").write_text("Du bist ein Rust-Experte")
            result = runner.invoke(
                cli,
                ["chat", "--system-file", "prompt.txt", "--url", "http://fake"],
                input="Hello\nexit\n",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            any(
                m.get("role") == "system" and "Rust-Experte" in m.get("content", "")
                for m in captured_messages
            ),
            f"System file prompt not found in messages: {captured_messages}",
        )

    @patch("cli.main._resolve_model", return_value="test.gguf")
    @patch("cli.main._stream_chat")
    def test_json_flag_outputs_json_lines(self, mock_stream, mock_resolve):
        """--json produces one JSON object per line with role/content."""

        def json_stream(url, model, messages, session):
            yield ("Hello from model", None)
            yield ("", {"eval_count": 1, "eval_duration": 1_000_000_000})

        mock_stream.side_effect = json_stream

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["chat", "--json", "--url", "http://fake"],
            input="Hi\nexit\n",
        )

        self.assertEqual(result.exit_code, 0, result.output)
        # JSON line may be preceded by the echoed prompt in CliRunner output
        json_objects = []
        for line in result.output.splitlines():
            # Strip potential "You> " prefix from CliRunner stdout capture
            stripped = line.strip()
            # Find JSON object anywhere in the line
            brace_idx = stripped.find("{")
            if brace_idx != -1:
                candidate = stripped[brace_idx:]
                try:
                    obj = json.loads(candidate)
                    if "role" in obj:
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    pass
        self.assertTrue(len(json_objects) >= 1, f"No JSON objects in output:\n{result.output}")
        self.assertEqual(json_objects[0]["role"], "assistant")
        self.assertIn("Hello from model", json_objects[0]["content"])

    def test_chat_help_shows_new_flags(self):
        """Help output must list --system, --system-file, --json."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        self.assertIn("--system", result.output)
        self.assertIn("--system-file", result.output)
        self.assertIn("--json", result.output)


# ── QW-9: status flags ────────────────────────────────────────────────────────

class TestStatusFlags(unittest.TestCase):

    def _good_status(self):
        return (
            True,
            {"status": "ok", "queue_depth": 0, "max_queue": 10,
             "runtime": {"active_sessions": 1, "recovery_count": 0,
                         "loaded_models": ["test.gguf"], "uptime_seconds": 3661}},
            True,
            {"status": "ready"},
        )

    @patch("cli.main._fetch_status")
    def test_plain_flag_outputs_plain_text(self, mock_fetch):
        mock_fetch.return_value = self._good_status()
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--plain", "--url", "http://fake"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("status:", result.output)
        self.assertIn("model:", result.output)
        self.assertIn("health:", result.output)
        # Should NOT contain Rich markup
        self.assertNotIn("[bold", result.output)

    def test_status_help_shows_new_flags(self):
        """Help output must list --watch and --plain."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        self.assertIn("--watch", result.output)
        self.assertIn("--plain", result.output)

    @patch("cli.main._fetch_status")
    def test_plain_status_content(self, mock_fetch):
        """Plain status output has key: value lines."""
        mock_fetch.return_value = self._good_status()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plain_status(*self._good_status())
        output = buf.getvalue()
        self.assertIn("status: ok", output)
        self.assertIn("model: test.gguf", output)
        self.assertIn("uptime: 1h 1m 1s", output)


# ── QW-9: exit codes ─────────────────────────────────────────────────────────

class TestExitCodes(unittest.TestCase):

    @patch("cli.main.requests.get")
    def test_status_exit_1_on_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--url", "http://fake"])
        self.assertEqual(result.exit_code, 1)

    @patch("cli.main.requests.get")
    def test_status_exit_2_on_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout("timed out")
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--url", "http://fake"])
        self.assertEqual(result.exit_code, 2)


# ── QW-10: multiline input ────────────────────────────────────────────────────

class TestReadMultilineInput(unittest.TestCase):

    def test_single_line_no_continuation(self):
        """A line without trailing \\ is returned as-is."""
        with patch("cli.main.console") as mock_console:
            mock_console.input.side_effect = ["Hello world"]
            result = _read_multiline_input("You> ")
        self.assertEqual(result, "Hello world")

    def test_backslash_continuation_two_lines(self):
        """A line ending with \\ triggers continuation; both lines are joined."""
        with patch("cli.main.console") as mock_console:
            mock_console.input.side_effect = ["First line\\", "Second line"]
            result = _read_multiline_input("You> ")
        self.assertEqual(result, "First line\nSecond line")

    def test_backslash_continuation_three_lines(self):
        """Multiple backslash lines are all joined."""
        with patch("cli.main.console") as mock_console:
            mock_console.input.side_effect = ["Line1\\", "Line2\\", "Line3"]
            result = _read_multiline_input("You> ")
        self.assertEqual(result, "Line1\nLine2\nLine3")

    def test_eoferror_returns_none(self):
        """EOF/Ctrl-D returns None."""
        with patch("cli.main.console") as mock_console:
            mock_console.input.side_effect = EOFError
            result = _read_multiline_input("You> ")
        self.assertIsNone(result)

    def test_keyboard_interrupt_returns_none(self):
        """Ctrl-C returns None."""
        with patch("cli.main.console") as mock_console:
            mock_console.input.side_effect = KeyboardInterrupt
            result = _read_multiline_input("You> ")
        self.assertIsNone(result)


class TestOpenEditorInput(unittest.TestCase):

    def test_editor_content_returned(self):
        """Content written by the editor is returned."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            tmppath = f.name

        def fake_mkstemp(suffix):
            fd = os.open(tmppath, os.O_RDWR)
            return fd, tmppath

        def fake_run(cmd, **kwargs):
            Path(tmppath).write_text("Hello from editor")
            m = MagicMock()
            m.returncode = 0
            return m

        with patch("cli.main.tempfile.mkstemp", side_effect=fake_mkstemp):
            with patch("cli.main.subprocess.run", side_effect=fake_run):
                result = _open_editor_input()

        self.assertEqual(result, "Hello from editor")

    def test_empty_editor_returns_none(self):
        """If the editor saves an empty file, None is returned."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            tmppath = f.name

        def fake_mkstemp(suffix):
            fd = os.open(tmppath, os.O_RDWR)
            return fd, tmppath

        def fake_run(cmd, **kwargs):
            Path(tmppath).write_text("")
            m = MagicMock()
            m.returncode = 0
            return m

        with patch("cli.main.tempfile.mkstemp", side_effect=fake_mkstemp):
            with patch("cli.main.subprocess.run", side_effect=fake_run):
                result = _open_editor_input()

        self.assertIsNone(result)

    def test_uses_editor_env_var(self):
        """$EDITOR env var is respected."""
        called_with = []

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            tmppath = f.name

        def fake_mkstemp(suffix):
            fd = os.open(tmppath, os.O_RDWR)
            return fd, tmppath

        def fake_run(cmd, **kwargs):
            called_with.extend(cmd)
            Path(tmppath).write_text("content")
            m = MagicMock()
            m.returncode = 0
            return m

        with patch.dict(os.environ, {"EDITOR": "vim"}):
            with patch("cli.main.tempfile.mkstemp", side_effect=fake_mkstemp):
                with patch("cli.main.subprocess.run", side_effect=fake_run):
                    _open_editor_input()

        self.assertEqual(called_with[0], "vim")

    def test_fallback_editor_is_nano(self):
        """If $EDITOR not set, falls back to nano."""
        called_with = []

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            tmppath = f.name

        def fake_mkstemp(suffix):
            fd = os.open(tmppath, os.O_RDWR)
            return fd, tmppath

        def fake_run(cmd, **kwargs):
            called_with.extend(cmd)
            Path(tmppath).write_text("content")
            m = MagicMock()
            m.returncode = 0
            return m

        env = {k: v for k, v in os.environ.items() if k != "EDITOR"}
        with patch.dict(os.environ, env, clear=True):
            with patch("cli.main.tempfile.mkstemp", side_effect=fake_mkstemp):
                with patch("cli.main.subprocess.run", side_effect=fake_run):
                    _open_editor_input()

        self.assertEqual(called_with[0], "nano")


# ── QW-10: .edit in chat REPL ─────────────────────────────────────────────────

class TestChatEditCommand(unittest.TestCase):

    @patch("cli.main._resolve_model", return_value="test.gguf")
    @patch("cli.main._stream_chat")
    @patch("cli.main._open_editor_input")
    def test_edit_command_sends_editor_content(self, mock_editor, mock_stream, mock_resolve):
        """Typing .edit opens the editor and sends its content as the next message."""
        editor_content = "Tell me about Rust."
        mock_editor.return_value = editor_content

        captured_messages: list = []

        def capturing_stream(url, model, messages, session):
            captured_messages.extend(messages)
            yield ("Sure!", None)
            yield ("", {"eval_count": 1, "eval_duration": 1_000_000_000})

        mock_stream.side_effect = capturing_stream

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["chat", "--url", "http://fake"],
            input=".edit\nexit\n",
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_editor.assert_called_once()
        user_msgs = [m for m in captured_messages if m.get("role") == "user"]
        self.assertTrue(
            any(editor_content in m.get("content", "") for m in user_msgs),
            f"Editor content not in messages: {captured_messages}",
        )

    @patch("cli.main._resolve_model", return_value="test.gguf")
    @patch("cli.main._stream_chat")
    @patch("cli.main._open_editor_input")
    def test_empty_edit_skips_send(self, mock_editor, mock_stream, mock_resolve):
        """.edit with empty result does NOT send a message."""
        mock_editor.return_value = None
        mock_stream.side_effect = []  # Should never be called

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["chat", "--url", "http://fake"],
            input=".edit\nexit\n",
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
