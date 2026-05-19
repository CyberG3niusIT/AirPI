#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class BenchResult:
    name: str
    ok: bool
    elapsed_seconds: float
    tokens: int
    tokens_per_second: float
    detail: str


def post_json(url: str, payload: dict, timeout: float) -> tuple[bool, dict | str, float]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        elapsed = time.perf_counter() - start
        return True, json.loads(body), elapsed
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - start
        return False, str(exc), elapsed


def run_case(base_url: str, model: str, name: str, prompt: str, session_id: str | None, timeout: float) -> BenchResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "max_tokens": 64,
        "temperature": 0.1,
    }
    if session_id:
        payload["session_id"] = session_id
    ok, detail, elapsed = post_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
    tokens = detail.get("eval_count", 0) if isinstance(detail, dict) else 0
    rate = tokens / elapsed if elapsed > 0 else 0.0
    return BenchResult(name, ok, elapsed, tokens, rate, json.dumps(detail) if isinstance(detail, dict) else str(detail))


def markdown(results: list[BenchResult]) -> str:
    lines = [
        "| Case | OK | Seconds | Tokens | tok/sec |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item.name} | {'yes' if item.ok else 'no'} | {item.elapsed_seconds:.3f} | "
            f"{item.tokens} | {item.tokens_per_second:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible AirPI benchmark cases.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cases = [
        ("cold_or_first", "Write one Python function that returns 42.", None),
        ("warm", "Write one Python function that returns 42.", None),
        ("session_kv_reuse", "Continue with a one sentence explanation.", "airpi-bench-session"),
    ]
    if args.dry_run:
        results = [
            BenchResult(name, True, 0.0, 0, 0.0, "dry-run")
            for name, _prompt, _session_id in cases
        ]
    else:
        results = [
            run_case(args.base_url, args.model, name, prompt, session_id, args.timeout)
            for name, prompt, session_id in cases
        ]

    if args.format == "markdown":
        print(markdown(results))
    else:
        print(json.dumps({"results": [asdict(item) for item in results]}, indent=2, sort_keys=True))
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
