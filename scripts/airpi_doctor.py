#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import urllib.error
import urllib.request


def check_http_json(url: str, timeout: float = 5.0) -> tuple[bool, object | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return True, json.loads(body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


def post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[bool, object | str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return True, json.loads(body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


def result(name: str, ok: bool, detail: object) -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an AirPI installation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--models-dir", default=os.environ.get("AIRPI_MODELS_DIR", "/data/models"))
    parser.add_argument("--model", default=os.environ.get("AIRPI_DEFAULT_MODEL", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
    parser.add_argument("--generate", action="store_true", help="Run a small generate request.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    checks = [
        result("python", sys.version_info >= (3, 11), platform.python_version()),
        result("fastapi", importlib.util.find_spec("fastapi") is not None, "importable"),
        result("uvicorn", importlib.util.find_spec("uvicorn") is not None, "importable"),
        result("llama_cpp", importlib.util.find_spec("llama_cpp") is not None, "importable"),
        result("models_dir", os.path.isdir(args.models_dir), args.models_dir),
        result("default_model", os.path.isfile(os.path.join(args.models_dir, args.model)), args.model),
    ]

    for endpoint in ("/live", "/health", "/api/tags"):
        ok, detail = check_http_json(args.base_url.rstrip("/") + endpoint)
        checks.append(result(endpoint, ok, detail))

    if args.generate:
        ok, detail = post_json(args.base_url.rstrip("/") + "/api/generate", {
            "model": args.model,
            "prompt": "Return the word ok.",
            "stream": False,
            "max_tokens": 8,
        })
        checks.append(result("/api/generate", ok and isinstance(detail, dict) and detail.get("done") is True, detail))

    payload = {"status": "ok" if all(item["ok"] for item in checks) else "failed", "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in checks:
            marker = "OK" if item["ok"] else "FAIL"
            print(f"{marker:4} {item['name']}: {item['detail']}")
        print(f"\nstatus: {payload['status']}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
