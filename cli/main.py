"""AirPI CLI — Terminal interface for the AirPI LLM Inference Server."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Iterator

import click
import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from rich.markup import escape

console = Console()

DEFAULT_URL = "http://localhost:11435"
DEFAULT_SESSION = "default"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and clearly."

BENCH_PROMPTS = [
    "Explain what a Raspberry Pi is in two sentences.",
    "Write a Python function that reverses a string.",
    "What is the difference between RAM and storage?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_auth_headers() -> dict:
    api_key = os.environ.get("AIRPI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _stream_chat(url: str, model: str, messages: list[dict], session: str) -> Iterator[tuple[str, dict | None]]:
    """Yield (token_text, final_stats_or_None) from /api/chat (applies chat template properly)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "session_id": session,
    }
    try:
        with requests.post(
            f"{url}/api/chat",
            json=payload,
            headers=_get_auth_headers(),
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    yield ("", data)
                else:
                    yield (data.get("message", {}).get("content", ""), None)
    except requests.exceptions.Timeout:
        sys.exit(2)
    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]Cannot connect to AirPI at {url}. Is the server running?[/bold red]")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code >= 500:
            console.print(f"[bold red]Server error: {exc}[/bold red]")
            sys.exit(1)
        raise click.ClickException(f"Server error: {exc}")


def _stream_generate(url: str, model: str, prompt: str, session: str) -> Iterator[tuple[str, dict | None]]:
    """Yield (token_text, final_stats_or_None) from /api/generate (raw, no chat template)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "session_id": session,
    }
    try:
        with requests.post(
            f"{url}/api/generate",
            json=payload,
            headers=_get_auth_headers(),
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    yield ("", data)
                else:
                    yield (data.get("response", ""), None)
    except requests.exceptions.Timeout:
        sys.exit(2)
    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]Cannot connect to AirPI at {url}. Is the server running?[/bold red]")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code >= 500:
            console.print(f"[bold red]Server error: {exc}[/bold red]")
            sys.exit(1)
        raise click.ClickException(f"Server error: {exc}")


def _get_json(url: str, path: str) -> dict:
    try:
        resp = requests.get(f"{url}{path}", headers=_get_auth_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        sys.exit(2)
    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]Cannot connect to AirPI at {url}. Is the server running?[/bold red]")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code >= 500:
            console.print(f"[bold red]Server error: {exc}[/bold red]")
            sys.exit(1)
        raise click.ClickException(f"Server error: {exc}")


def _resolve_model(url: str, model: str | None) -> str:
    """If model is None, fetch the default from /api/tags and pick the first available."""
    if model:
        return model
    try:
        data = _get_json(url, "/api/tags")
        models = data.get("models", [])
        if models:
            return models[0]["name"]
    except click.ClickException:
        pass
    # Fallback to config default name
    return "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"


# ── CLI Group ─────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("2.0.0", prog_name="airpi")
def cli():
    """AirPI — LLM Inference CLI for Raspberry Pi."""


# ── chat ──────────────────────────────────────────────────────────────────────

def _read_multiline_input(prompt_str: str) -> str | None:
    """Read potentially multi-line input with backslash continuation.

    Returns the assembled string, or None on EOF/KeyboardInterrupt.
    Lines ending with \\ are continued; the backslash is stripped and the next
    line is appended (with a newline separator).
    """
    lines: list[str] = []
    current_prompt = prompt_str
    while True:
        try:
            line = console.input(current_prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        if line.endswith("\\"):
            lines.append(line[:-1])
            current_prompt = "[bold yellow]...[/bold yellow] "
        else:
            lines.append(line)
            break
    return "\n".join(lines)


def _open_editor_input() -> str | None:
    """Open $EDITOR with a temp file and return its contents after save/close.

    Returns None if the file was empty or on error.
    """
    editor = os.environ.get("EDITOR", "nano")
    fd, tmpfile = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        result = subprocess.run([editor, tmpfile])
        if result.returncode != 0:
            console.print(f"[yellow]Editor exited with code {result.returncode}[/yellow]")
        content = Path(tmpfile).read_text(encoding="utf-8").strip()
    except Exception as exc:
        console.print(f"[red]Editor error: {exc}[/red]")
        content = ""
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass
    return content if content else None


@cli.command()
@click.option("--session", "-s", default=DEFAULT_SESSION, show_default=True, help="Session name (KV-cache key)")
@click.option("--model", "-m", default=None, help="Model name (default: server default)")
@click.option("--url", "-u", default=DEFAULT_URL, show_default=True, help="AirPI server URL")
@click.option("--system", default=None, help="System prompt for this session")
@click.option("--system-file", "system_file", default=None, type=click.Path(exists=True, dir_okay=False), help="Load system prompt from file")
@click.option("--json", "json_output", is_flag=True, default=False, help="Machine-readable output: one JSON object per line")
def chat(session: str, model: str | None, url: str, system: str | None, system_file: str | None, json_output: bool):
    """Interactive chat with streaming output (REPL style)."""

    # ── Resolve system prompt ─────────────────────────────────────────────────
    if system_file:
        try:
            system_prompt = Path(system_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            console.print(f"[bold red]Cannot read system file: {exc}[/bold red]")
            sys.exit(1)
    elif system:
        system_prompt = system
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    resolved_model = _resolve_model(url, model)

    if not json_output:
        console.print(
            Panel(
                f"[bold cyan]AirPI Chat[/bold cyan]  "
                f"session=[bold green]{escape(session)}[/bold green]  "
                f"model=[dim]{escape(resolved_model)}[/dim]  "
                f"url=[dim]{url}[/dim]\n"
                "[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to leave, "
                "[bold].edit[/bold] to open editor, Ctrl+C / Ctrl+D to abort.[/dim]",
                box=box.ROUNDED,
            )
        )

    history: list[dict] = [{"role": "system", "content": system_prompt}]

    while True:
        # ── QW-10: read input with backslash-continuation ─────────────────────
        if json_output:
            # In JSON mode, use plain input (no rich markup in prompt)
            try:
                raw = input("You> ")
            except (EOFError, KeyboardInterrupt):
                if not json_output:
                    console.print("\n[dim]Bye.[/dim]")
                break
            # Still support backslash continuation in json mode
            lines = []
            line = raw
            while line.endswith("\\"):
                lines.append(line[:-1])
                try:
                    line = input("... ")
                except (EOFError, KeyboardInterrupt):
                    line = ""
                    break
            lines.append(line)
            user_input = "\n".join(lines).strip()
        else:
            raw = _read_multiline_input("[bold yellow]You>[/bold yellow] ")
            if raw is None:
                console.print("\n[dim]Bye.[/dim]")
                break
            user_input = raw.strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            if not json_output:
                console.print("[dim]Bye.[/dim]")
            break

        # ── QW-10: .edit command ──────────────────────────────────────────────
        if user_input == ".edit":
            content = _open_editor_input()
            if not content:
                console.print("[dim]Empty message — aborted.[/dim]")
                continue
            preview = content[:100] + "..." if len(content) > 100 else content
            console.print(f"[dim]Nachricht: {escape(preview)}[/dim]")
            user_input = content

        # ── Memory shortcuts ──────────────────────────────────────────────────
        if user_input.lower().startswith("merke:"):
            fact = user_input[6:].strip()
            if fact:
                try:
                    resp = requests.post(
                        f"{url}/memory/store",
                        json={"content": fact, "category": "fact"},
                        headers=_get_auth_headers(),
                        timeout=10,
                    )
                    resp.raise_for_status()
                    if not json_output:
                        console.print(f"[green]✓ Gespeichert:[/green] {escape(fact)}")
                except Exception as exc:
                    if not json_output:
                        console.print(f"[red]Fehler beim Speichern: {exc}[/red]")
            continue

        if user_input.lower().startswith("vergiss:"):
            keyword = user_input[8:].strip()
            if keyword:
                try:
                    resp = requests.post(
                        f"{url}/memory/delete",
                        json={"keyword": keyword},
                        headers=_get_auth_headers(),
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if not json_output:
                        console.print(
                            f"[green]✓ Einträge mit '{escape(keyword)}' entfernt.[/green] "
                            f"[dim]({data.get('deleted', 0)} Einträge)[/dim]"
                        )
                except Exception as exc:
                    if not json_output:
                        console.print(f"[red]Fehler beim Löschen: {exc}[/red]")
            continue

        if user_input.lower() in ("memory?", "was weißt du?", "memory"):
            try:
                resp = requests.get(f"{url}/memory", headers=_get_auth_headers(), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("content", "").strip()
                if not json_output:
                    if content:
                        console.print(Panel(content, title="[bold cyan]AirPI Memory[/bold cyan]", box=box.ROUNDED))
                    else:
                        console.print("[dim]Noch keine Einträge im Memory.[/dim]")
                else:
                    print(json.dumps({"role": "memory", "content": content}))
            except Exception as exc:
                if not json_output:
                    console.print(f"[red]Fehler beim Lesen des Memory: {exc}[/red]")
            continue
        # ── Ende Memory shortcuts ──────────────────────────────────────────────

        history.append({"role": "user", "content": user_input})
        if not json_output:
            console.print(f"[bold blue]AirPI>[/bold blue] ", end="")

        final_stats: dict | None = None
        token_count = 0
        start_time = time.monotonic()
        assistant_reply = ""

        try:
            for token_text, stats in _stream_chat(url, resolved_model, history, session):
                if stats is not None:
                    final_stats = stats
                else:
                    if json_output:
                        # In JSON mode, buffer; emit at end
                        assistant_reply += token_text
                    else:
                        console.print(token_text, end="", markup=False)
                        assistant_reply += token_text
                    token_count += 1
        except click.ClickException as exc:
            if json_output:
                print(json.dumps({"role": "error", "content": exc.format_message()}))
            else:
                console.print(f"\n[red]Error: {exc.format_message()}[/red]")
            history.pop()  # remove the user message that failed
            continue

        if assistant_reply:
            history.append({"role": "assistant", "content": assistant_reply})

        elapsed = time.monotonic() - start_time

        # Prefer server-reported eval_count and eval_duration
        if final_stats:
            eval_count = final_stats.get("eval_count", token_count)
            eval_duration_ns = final_stats.get("eval_duration", 0)
            if eval_duration_ns and eval_duration_ns > 0:
                toks_per_sec = eval_count / (eval_duration_ns / 1e9)
            elif elapsed > 0:
                toks_per_sec = eval_count / elapsed
            else:
                toks_per_sec = 0.0
        else:
            eval_count = token_count
            toks_per_sec = token_count / elapsed if elapsed > 0 else 0.0

        if json_output:
            print(json.dumps({"role": "assistant", "content": assistant_reply}))
        else:
            console.print()
            console.print(
                f"[dim][ tok/s: {toks_per_sec:.1f} | session: "
                f"[bold green]{escape(session)}[/bold green] ][/dim]"
            )


# ── bench ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default=None, help="Model name (default: server default)")
@click.option("--url", "-u", default=DEFAULT_URL, show_default=True, help="AirPI server URL")
@click.option("--runs", "-n", default=3, show_default=True, help="Number of benchmark runs")
def bench(model: str | None, url: str, runs: int):
    """Benchmark inference speed (tokens/second)."""

    resolved_model = _resolve_model(url, model)
    # Extract a short display name
    model_display = resolved_model.replace(".gguf", "").replace("-", " ").title()

    console.print(
        f"[bold cyan]AirPI Benchmark[/bold cyan]  "
        f"model=[dim]{escape(resolved_model)}[/dim]  runs=[dim]{runs}[/dim]"
    )
    console.print()

    results: list[tuple[int, float, float]] = []  # (tokens, time_s, tok_s)

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
    table.add_column("Run", style="dim", width=5)
    table.add_column("Prompt (truncated)", style="dim", max_width=30)
    table.add_column("Tokens", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Tok/s", justify="right", style="bold green")

    for i, prompt in enumerate(BENCH_PROMPTS[:runs], start=1):
        console.print(f"[dim]Run {i}/{runs}…[/dim]", end="\r")

        final_stats: dict | None = None
        token_count = 0
        start_time = time.monotonic()

        try:
            for _, stats in _stream_generate(url, resolved_model, prompt, f"bench-{i}"):
                if stats is not None:
                    final_stats = stats
                else:
                    token_count += 1
        except click.ClickException as exc:
            console.print(f"\n[red]Error on run {i}: {exc.format_message()}[/red]")
            continue

        elapsed = time.monotonic() - start_time

        if final_stats:
            eval_count = final_stats.get("eval_count", token_count)
            eval_duration_ns = final_stats.get("eval_duration", 0)
            if eval_duration_ns and eval_duration_ns > 0:
                toks_per_sec = eval_count / (eval_duration_ns / 1e9)
                elapsed = eval_duration_ns / 1e9
            else:
                toks_per_sec = eval_count / elapsed if elapsed > 0 else 0.0
        else:
            eval_count = token_count
            toks_per_sec = token_count / elapsed if elapsed > 0 else 0.0

        results.append((eval_count, elapsed, toks_per_sec))
        prompt_preview = prompt[:28] + "…" if len(prompt) > 28 else prompt
        table.add_row(
            str(i),
            prompt_preview,
            str(eval_count),
            f"{elapsed:.2f}",
            f"{toks_per_sec:.1f}",
        )

    console.print()
    console.print(table)

    if not results:
        console.print("[red]No successful runs.[/red]")
        return

    tok_s_values = [r[2] for r in results]
    median_tps = statistics.median(tok_s_values)
    min_tps = min(tok_s_values)
    max_tps = max(tok_s_values)

    # Summary stats table
    stats_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    stats_table.add_column("Stat", style="bold")
    stats_table.add_column("Value", style="bold green")
    stats_table.add_row("Median", f"{median_tps:.1f} tok/s")
    stats_table.add_row("Min", f"{min_tps:.1f} tok/s")
    stats_table.add_row("Max", f"{max_tps:.1f} tok/s")
    console.print(stats_table)

    # Summary line
    console.print(
        f"\n[bold]AirPI: {median_tps:.1f} tok/s "
        f"({model_display}, Pi 5)[/bold]"
    )

    # GitHub badge URL
    badge_label = urllib.parse.quote(f"{median_tps:.1f} tok/s", safe="")
    badge_url = f"https://img.shields.io/badge/AirPI-{badge_label}-blue"
    console.print(f"\n[dim]GitHub badge:[/dim]")
    console.print(f"![AirPI]({badge_url})")


# ── models ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--url", "-u", default=DEFAULT_URL, show_default=True, help="AirPI server URL")
def models(url: str):
    """List available models on the server."""

    data = _get_json(url, "/api/tags")
    model_list = data.get("models", [])

    # Fetch config defaults from /health for comparison
    fast_model = None
    default_model = None
    try:
        health = _get_json(url, "/health")
        loaded = health.get("loaded_models", [])
        # Try /ready to get model path info
        ready_data = requests.get(f"{url}/ready", timeout=5).json()
    except Exception:
        pass

    # We'll compare against known defaults from config module names
    # The server doesn't expose these directly, so we pattern-match
    FAST_HINT = "0.5b"
    DEFAULT_HINT = "1.5b"

    if not model_list:
        console.print("[yellow]No models found. Check your models directory.[/yellow]")
        return

    table = Table(title="AirPI Models", box=box.ROUNDED, show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")
    table.add_column("Tags", style="bold")

    for m in model_list:
        name: str = m.get("name", "")
        size_bytes: int = m.get("size", 0)
        modified: str = m.get("modified_at", "")
        loaded: bool = m.get("details", {}).get("loaded", False)

        # Human-readable size
        if size_bytes >= 1_073_741_824:
            size_str = f"{size_bytes / 1_073_741_824:.1f} GB"
        elif size_bytes >= 1_048_576:
            size_str = f"{size_bytes / 1_048_576:.1f} MB"
        elif size_bytes > 0:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = "—"

        tags = []
        name_lower = name.lower()
        if FAST_HINT in name_lower:
            tags.append("[yellow]FAST[/yellow]")
        if DEFAULT_HINT in name_lower:
            tags.append("[green]DEFAULT[/green]")
        if loaded:
            tags.append("[cyan]loaded[/cyan]")

        table.add_row(name, size_str, modified[:19] if modified else "—", " ".join(tags))

    console.print(table)


# ── status ────────────────────────────────────────────────────────────────────

def _fetch_status(url: str) -> tuple[bool, dict, bool, dict]:
    """Fetch /health and /ready. Returns (health_ok, health_data, ready_ok, ready_data)."""
    health_ok = False
    health_data: dict = {}
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        health_data = resp.json()
        health_ok = resp.status_code == 200 and health_data.get("status") == "ok"
    except requests.exceptions.Timeout:
        sys.exit(2)
    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]Cannot connect to AirPI at {url}[/bold red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Health check failed: {exc}[/bold red]")
        sys.exit(1)

    ready_ok = False
    ready_data: dict = {}
    try:
        resp = requests.get(f"{url}/ready", timeout=10)
        ready_data = resp.json()
        ready_ok = ready_data.get("status") == "ready"
    except Exception:
        pass

    return health_ok, health_data, ready_ok, ready_data


def _build_status_table(url: str, health_ok: bool, health_data: dict, ready_ok: bool, ready_data: dict) -> Table:
    """Build and return the status rich Table."""
    ok_style = "bold green"
    warn_style = "bold yellow"
    err_style = "bold red"

    runtime = health_data.get("runtime", health_data)

    queue_depth = health_data.get("queue_depth", "—")
    max_queue = health_data.get("max_queue", "—")
    active_sessions = runtime.get("active_sessions", "—")
    recovery_count = runtime.get("recovery_count", "—")
    loaded_models = runtime.get("loaded_models", [])
    uptime_secs = runtime.get("uptime_seconds")

    if isinstance(uptime_secs, (int, float)):
        h = int(uptime_secs // 3600)
        m = int((uptime_secs % 3600) // 60)
        s = int(uptime_secs % 60)
        uptime_str = f"{h}h {m}m {s}s"
    else:
        uptime_str = "—"

    overall_ok = health_ok and ready_ok
    overall_style = ok_style if overall_ok else (warn_style if health_ok else err_style)
    overall_label = "OK" if overall_ok else ("DEGRADED" if health_ok else "DOWN")

    table = Table(
        title=f"[{overall_style}]AirPI Status: {overall_label}[/{overall_style}]",
        box=box.ROUNDED,
        show_header=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    def _status_row(label: str, value, ok: bool | None = None):
        if ok is True:
            val_str = f"[{ok_style}]{escape(str(value))}[/{ok_style}]"
        elif ok is False:
            val_str = f"[{err_style}]{escape(str(value))}[/{err_style}]"
        else:
            val_str = escape(str(value))
        table.add_row(label, val_str)

    _status_row("Health", health_data.get("status", "unknown"), health_ok)
    _status_row("Ready", ready_data.get("status", "unknown"), ready_ok)
    _status_row("Queue Depth", f"{queue_depth} / {max_queue}",
                ok=isinstance(queue_depth, int) and isinstance(max_queue, int) and queue_depth < max_queue)
    _status_row("Active Sessions", str(active_sessions))
    _status_row("Loaded Models", ", ".join(loaded_models) if loaded_models else "none")
    _status_row("Recovery Count", str(recovery_count),
                ok=recovery_count == 0 if isinstance(recovery_count, int) else None)
    _status_row("Uptime", uptime_str)

    return table


def _print_plain_status(health_ok: bool, health_data: dict, ready_ok: bool, ready_data: dict) -> None:
    """Print status as plain key: value lines suitable for pipes."""
    runtime = health_data.get("runtime", health_data)
    loaded_models = runtime.get("loaded_models", [])
    uptime_secs = runtime.get("uptime_seconds")

    if isinstance(uptime_secs, (int, float)):
        h = int(uptime_secs // 3600)
        m = int((uptime_secs % 3600) // 60)
        s = int(uptime_secs % 60)
        uptime_str = f"{h}h {m}m {s}s"
    else:
        uptime_str = "unknown"

    overall_label = "ok" if (health_ok and ready_ok) else ("degraded" if health_ok else "down")
    print(f"status: {overall_label}")
    print(f"health: {health_data.get('status', 'unknown')}")
    print(f"ready: {ready_data.get('status', 'unknown')}")
    print(f"model: {', '.join(loaded_models) if loaded_models else 'none'}")
    print(f"queue_depth: {health_data.get('queue_depth', 'unknown')}")
    print(f"active_sessions: {runtime.get('active_sessions', 'unknown')}")
    print(f"recovery_count: {runtime.get('recovery_count', 'unknown')}")
    print(f"uptime: {uptime_str}")


@cli.command()
@click.option("--url", "-u", default=DEFAULT_URL, show_default=True, help="AirPI server URL")
@click.option("--watch", is_flag=True, default=False, help="Poll every 3s and refresh output in-place")
@click.option("--plain", is_flag=True, default=False, help="Plain text output for pipes (no Rich formatting)")
def status(url: str, watch: bool, plain: bool):
    """Show server health and readiness status."""

    if watch:
        try:
            while True:
                health_ok, health_data, ready_ok, ready_data = _fetch_status(url)
                if plain:
                    # Clear screen for plain watch mode
                    print("\033[2J\033[H", end="", flush=True)
                    _print_plain_status(health_ok, health_data, ready_ok, ready_data)
                else:
                    table = _build_status_table(url, health_ok, health_data, ready_ok, ready_data)
                    with Live(table, refresh_per_second=1, screen=True) as live:
                        time.sleep(3)
                        live.update(table)
                    # Re-fetch after sleep; loop continues
                    continue
                time.sleep(3)
        except KeyboardInterrupt:
            if not plain:
                console.print("\n[dim]Watch stopped.[/dim]")
        return

    health_ok, health_data, ready_ok, ready_data = _fetch_status(url)

    if not health_ok:
        # Server error / model not loaded
        if not plain:
            console.print("[bold red]Server is DOWN or returned error[/bold red]")
        else:
            print("status: down")
        sys.exit(1)

    if plain:
        _print_plain_status(health_ok, health_data, ready_ok, ready_data)
        return

    table = _build_status_table(url, health_ok, health_data, ready_ok, ready_data)
    console.print(table)

    if not ready_ok and ready_data:
        issues = []
        if not ready_data.get("models_dir_exists", True):
            issues.append("models directory not found")
        if not ready_data.get("default_model_exists", True):
            issues.append("default model file missing")
        if not ready_data.get("fast_model_exists", True):
            issues.append("fast model file missing")
        if issues:
            console.print("[bold yellow]Issues:[/bold yellow] " + ", ".join(issues))


# ── pull ──────────────────────────────────────────────────────────────────────

KNOWN_MODELS: dict[str, dict[str, str]] = {
    "qwen2.5-coder-1.5b": {
        "hf_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-0.5b": {
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    },
    "qwen2.5-coder-7b": {
        "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-q4_k_m.gguf",
    },
    "gemma-2b": {
        "hf_repo": "google/gemma-2b-GGUF",
        "filename": "gemma-2b.gguf",
    },
}

MODELS_DIR = "/opt/airpi/models"


@cli.command()
@click.argument("model_name")
@click.option("--url", "-u", default=DEFAULT_URL, show_default=True, help="AirPI server URL")
def pull(model_name: str, url: str):
    """Show how to download MODEL_NAME to the AirPI models directory.

    AirPI does not manage model downloads directly. This command prints the
    wget or huggingface-cli commands you need to run on the Pi.
    """

    # Try to match by prefix / substring
    matched_key = None
    name_lower = model_name.lower().replace(".gguf", "")
    for key in KNOWN_MODELS:
        if key in name_lower or name_lower in key:
            matched_key = key
            break

    console.print(
        Panel(
            f"[bold cyan]AirPI Pull — {escape(model_name)}[/bold cyan]\n"
            "[dim]AirPI does not include a model downloader. "
            "Run one of the commands below on your Raspberry Pi:[/dim]",
            box=box.ROUNDED,
        )
    )

    if matched_key:
        info = KNOWN_MODELS[matched_key]
        hf_repo = info["hf_repo"]
        filename = info["filename"]
        dest = f"{MODELS_DIR}/{filename}"

        console.print("\n[bold]Option 1 — wget (fastest on Pi):[/bold]")
        hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
        console.print(
            f"[green]wget -O {dest} \\\n"
            f"  {hf_url}[/green]"
        )

        console.print("\n[bold]Option 2 — huggingface-cli:[/bold]")
        console.print(
            f"[green]huggingface-cli download {hf_repo} {filename} \\\n"
            f"  --local-dir {MODELS_DIR}[/green]"
        )
    else:
        # Generic fallback — assume the user knows the HF repo
        safe_name = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
        dest = f"{MODELS_DIR}/{safe_name}"

        console.print("\n[bold]Option 1 — wget:[/bold]")
        console.print(
            f"[green]wget -O {dest} \\\n"
            f"  https://huggingface.co/<ORG>/<REPO>/resolve/main/{safe_name}[/green]"
        )

        console.print("\n[bold]Option 2 — huggingface-cli:[/bold]")
        console.print(
            f"[green]huggingface-cli download <ORG>/<REPO> {safe_name} \\\n"
            f"  --local-dir {MODELS_DIR}[/green]"
        )

        console.print(
            f"\n[yellow]Tip:[/yellow] Known shortcuts: {', '.join(KNOWN_MODELS.keys())}"
        )

    console.print(f"\n[dim]After download, restart AirPI or run [bold]airpi status --url {url}[/bold] to verify.[/dim]")


if __name__ == "__main__":
    cli()
