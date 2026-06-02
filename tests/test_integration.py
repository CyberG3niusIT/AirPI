"""
Integration tests against live AirPI server on localhost:11435.

Run with:
    cd /home/alex/AirPI && .venv/bin/python -m pytest tests/test_integration.py -v --timeout=60 -s

Requirements:
    - AirPI server running on localhost:11435
    - pytest-timeout installed (.venv/bin/pip install pytest-timeout)

Notes:
    - All tests are idempotent (safe to run multiple times).
    - Tests requiring LLM inference are marked @pytest.mark.slow.
    - Cleanup logic runs in teardown to remove integration-test artifacts.
"""

from __future__ import annotations

import pytest
import requests
import time
import json
import uuid

BASE = "http://localhost:11435"

# Unique tag so integration-test memory entries can be identified and cleaned up.
_TAG = "INTEGRATION_TEST_" + uuid.uuid4().hex[:8]

# Default model reported by /api/tags (first listed is fine for inference tests)
_DEFAULT_MODEL = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"


# ── Server fixture ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def require_server():
    """Skip all tests if the server is not running on localhost:11435."""
    try:
        r = requests.get(f"{BASE}/live", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("AirPI server not running on localhost:11435")


# ── Helper ────────────────────────────────────────────────────────────────────

def _store(content: str, category: str = "fact") -> requests.Response:
    return requests.post(
        f"{BASE}/memory/store",
        json={"content": content, "category": category},
        timeout=5,
    )


def _delete(keyword: str) -> requests.Response:
    return requests.post(
        f"{BASE}/memory/delete",
        json={"keyword": keyword},
        timeout=5,
    )


def _get_memory() -> dict:
    r = requests.get(f"{BASE}/memory", timeout=5)
    r.raise_for_status()
    return r.json()


# ── TestLiveness ──────────────────────────────────────────────────────────────

class TestLiveness:
    def test_live_returns_ok(self):
        r = requests.get(f"{BASE}/live", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_health_has_required_fields(self):
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data, "health missing 'status' field"
        assert "queue_depth" in data, "health missing 'queue_depth' field"
        assert "active_sessions" in data, "health missing 'active_sessions' field"
        assert "loaded_models" in data, "health missing 'loaded_models' field"
        assert data["status"] == "ok"
        assert isinstance(data["queue_depth"], int)
        assert isinstance(data["active_sessions"], int)
        assert isinstance(data["loaded_models"], list)

    def test_ready_endpoint(self):
        r = requests.get(f"{BASE}/ready", timeout=5)
        # Either 200 (ready) or 503 (not_ready) — both are valid responses.
        assert r.status_code in (200, 503), f"Unexpected status {r.status_code}"
        data = r.json()
        assert "status" in data
        assert data["status"] in ("ready", "not_ready")

    def test_metrics_prometheus_format(self):
        r = requests.get(f"{BASE}/metrics", timeout=5)
        assert r.status_code == 200
        body = r.text
        # Must contain Prometheus HELP/TYPE headers with airpi_ prefix
        assert "airpi_requests_total" in body, "Missing airpi_requests_total"
        assert "airpi_queue_depth" in body, "Missing airpi_queue_depth"
        assert "airpi_tokens_total" in body, "Missing airpi_tokens_total"
        assert "# HELP airpi_" in body, "Missing HELP lines with airpi_ prefix"
        assert "# TYPE airpi_" in body, "Missing TYPE lines with airpi_ prefix"


# ── TestApiTags ───────────────────────────────────────────────────────────────

class TestApiTags:
    def test_api_tags_returns_models_list(self):
        r = requests.get(f"{BASE}/api/tags", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data, "Response missing 'models' key"
        assert isinstance(data["models"], list)

    def test_models_have_required_fields(self):
        r = requests.get(f"{BASE}/api/tags", timeout=5)
        assert r.status_code == 200
        models = r.json()["models"]
        # At least one model must be present for these field checks to be meaningful.
        if not models:
            pytest.skip("No models available in /api/tags")
        for model in models:
            assert "name" in model, f"Model entry missing 'name': {model}"
            assert "size" in model, f"Model entry missing 'size': {model}"
            assert "modified_at" in model, f"Model entry missing 'modified_at': {model}"
            assert isinstance(model["name"], str) and model["name"]
            assert isinstance(model["size"], int) and model["size"] >= 0


# ── TestMemorySystem ──────────────────────────────────────────────────────────

class TestMemorySystem:
    """All memory tests use _TAG to namespace their artifacts for clean teardown."""

    def teardown_method(self, method):
        """Best-effort cleanup: remove all entries tagged with _TAG."""
        try:
            _delete(_TAG)
        except Exception:
            pass

    def test_store_and_retrieve_fact(self):
        content = f"Integration test fact {_TAG}"
        r = _store(content)
        assert r.status_code == 200, f"store failed: {r.text}"
        data = r.json()
        assert data.get("ok") is True
        assert data.get("content") == content

        # Verify the fact appears in /memory
        mem = _get_memory()
        contents = [e["content"] for e in mem.get("entries", [])]
        assert content in contents, f"Stored fact not found in /memory entries. Got: {contents}"

    def test_delete_fact_by_keyword(self):
        content = f"Fact to delete {_TAG}"
        _store(content)

        # Confirm it's there
        mem_before = _get_memory()
        before_contents = [e["content"] for e in mem_before.get("entries", [])]
        assert content in before_contents, "Precondition: fact should be stored before delete"

        # Delete by the unique tag keyword
        r = _delete(_TAG)
        assert r.status_code == 200, f"delete failed: {r.text}"
        data = r.json()
        assert data.get("ok") is True
        assert isinstance(data.get("deleted"), int)
        assert data["deleted"] >= 1, "Expected at least 1 deletion"

        # Confirm it's gone
        mem_after = _get_memory()
        after_contents = [e["content"] for e in mem_after.get("entries", [])]
        assert content not in after_contents, "Deleted fact should not appear in /memory anymore"

    def test_memory_content_is_valid_markdown(self):
        r = requests.get(f"{BASE}/memory", timeout=5)
        assert r.status_code == 200
        data = r.json()
        content = data.get("content", "")
        assert isinstance(content, str), "Memory content must be a string"
        assert content.startswith("# AirPI Memory"), (
            f"Memory content does not start with '# AirPI Memory'. Got: {content[:80]!r}"
        )

    def test_store_requires_content_field(self):
        # POST /memory/store without 'content' must return 422
        r = requests.post(
            f"{BASE}/memory/store",
            json={"category": "fact"},  # missing 'content'
            timeout=5,
        )
        assert r.status_code == 422, (
            f"Expected 422 for missing content field, got {r.status_code}: {r.text}"
        )


# ── TestGraphData ─────────────────────────────────────────────────────────────

class TestGraphData:
    def teardown_method(self, method):
        try:
            _delete(_TAG)
        except Exception:
            pass

    def test_graph_data_structure(self):
        r = requests.get(f"{BASE}/graph/data", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data, "graph/data missing 'nodes' key"
        assert "edges" in data, "graph/data missing 'edges' key"
        assert "meta" in data, "graph/data missing 'meta' key"
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_graph_reflects_stored_memory(self):
        # Store a unique fact and verify that new concepts appear in the graph.
        # Note: the GraphBuilder tokenises on [A-Za-z0-9\-] — underscores are
        # word boundaries, so "XGRAPH" (≥3 chars, capitalised) is the stable prefix.
        unique_prefix = "XGRAPHNODE"
        content = f"Test graph concept {unique_prefix} integration marker"
        _store(content)

        r = requests.get(f"{BASE}/graph/data", timeout=5)
        assert r.status_code == 200
        data = r.json()
        node_ids = {n.get("id", "") for n in data.get("nodes", [])}
        # The unique capitalised prefix must appear as its own graph node.
        assert unique_prefix in node_ids, (
            f"Stored unique word '{unique_prefix}' not found as graph node. Nodes: {sorted(node_ids)[:20]}"
        )

    def test_graph_redirect(self):
        # GET /graph → 307 redirect to /ui/graph.html
        r = requests.get(f"{BASE}/graph", timeout=5, allow_redirects=False)
        assert r.status_code in (301, 302, 303, 307, 308), (
            f"Expected redirect from /graph, got {r.status_code}"
        )
        location = r.headers.get("location", "")
        assert "graph.html" in location, (
            f"Redirect location should point to graph.html, got: {location!r}"
        )


# ── TestStaticUI ──────────────────────────────────────────────────────────────

class TestStaticUI:
    def test_ui_index_html_serves(self):
        r = requests.get(f"{BASE}/ui/", timeout=5)
        assert r.status_code == 200, (
            f"GET /ui/ returned {r.status_code} — static UI not mounted or index.html missing"
        )
        # Should be HTML
        ct = r.headers.get("content-type", "")
        assert "html" in ct.lower(), f"Expected HTML content-type, got: {ct}"

    def test_graph_html_serves(self):
        r = requests.get(f"{BASE}/ui/graph.html", timeout=5)
        assert r.status_code == 200, (
            f"GET /ui/graph.html returned {r.status_code}"
        )
        ct = r.headers.get("content-type", "")
        assert "html" in ct.lower(), f"Expected HTML content-type, got: {ct}"


# ── TestApiGenerate ───────────────────────────────────────────────────────────

@pytest.mark.slow
class TestApiGenerate:
    """Tests that run real LLM inference — may take > 5s on Raspberry Pi."""

    def _get_model(self) -> str:
        """Return an available model name, skipping if none found."""
        r = requests.get(f"{BASE}/api/tags", timeout=5)
        models = r.json().get("models", [])
        if not models:
            pytest.skip("No models available in /api/tags")
        return models[0]["name"]

    def test_generate_non_streaming(self):
        model = self._get_model()
        payload = {
            "model": model,
            "prompt": "Say 'ok' and nothing else.",
            "stream": False,
            "max_tokens": 10,
            "temperature": 0.0,
        }
        r = requests.post(f"{BASE}/api/generate", json=payload, timeout=120)
        assert r.status_code == 200, f"generate failed ({r.status_code}): {r.text}"
        data = r.json()
        assert "response" in data, f"Missing 'response' field: {data}"
        assert "done" in data, f"Missing 'done' field: {data}"
        assert "eval_count" in data, f"Missing 'eval_count' field: {data}"
        assert data["done"] is True
        assert isinstance(data["response"], str)
        assert isinstance(data["eval_count"], int)

    def test_generate_invalid_model_returns_error(self):
        payload = {
            "model": "this-model-does-not-exist.gguf",
            "prompt": "Hello",
            "stream": False,
            "max_tokens": 5,
        }
        r = requests.post(f"{BASE}/api/generate", json=payload, timeout=30)
        # Server should return 4xx or 5xx — not 200
        assert r.status_code >= 400, (
            f"Expected error for unknown model, got {r.status_code}: {r.text}"
        )

    def test_generate_session_id_accepted(self):
        model = self._get_model()
        session_id = "integration-test"
        payload = {
            "model": model,
            "prompt": "Reply with a single word: yes.",
            "stream": False,
            "max_tokens": 10,
            "temperature": 0.0,
            "session_id": session_id,
        }
        r = requests.post(f"{BASE}/api/generate", json=payload, timeout=120)
        assert r.status_code == 200, f"generate with session_id failed ({r.status_code}): {r.text}"
        data = r.json()
        assert "response" in data
        assert data.get("done") is True
        # No error key expected
        assert "error" not in data, f"Unexpected error in response: {data}"


# ── TestApiChat ───────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestApiChat:
    """Tests that run real LLM inference via /api/chat — may take > 5s on Raspberry Pi."""

    def _get_model(self) -> str:
        r = requests.get(f"{BASE}/api/tags", timeout=5)
        models = r.json().get("models", [])
        if not models:
            pytest.skip("No models available in /api/tags")
        return models[0]["name"]

    def test_chat_non_streaming(self):
        model = self._get_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with a single word: hello."},
            ],
            "stream": False,
            "max_tokens": 10,
            "temperature": 0.0,
        }
        r = requests.post(f"{BASE}/api/chat", json=payload, timeout=120)
        assert r.status_code == 200, f"chat failed ({r.status_code}): {r.text}"
        data = r.json()
        assert "message" in data, f"Missing 'message' field: {data}"
        assert "done" in data, f"Missing 'done' field: {data}"
        msg = data["message"]
        assert isinstance(msg, dict), f"'message' should be a dict, got: {type(msg)}"
        assert msg.get("role") == "assistant", f"Expected role='assistant', got: {msg.get('role')}"
        assert "content" in msg, f"'message' missing 'content' key: {msg}"
        assert isinstance(msg["content"], str)
        assert data["done"] is True

    def test_chat_system_message_respected(self):
        """System prompt 'Antworte nur mit JAWOHL.' should steer the LLM response."""
        model = self._get_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Antworte nur mit JAWOHL. Kein anderer Text."},
                {"role": "user", "content": "Bist du bereit?"},
            ],
            "stream": False,
            "max_tokens": 20,
            "temperature": 0.0,
        }
        r = requests.post(f"{BASE}/api/chat", json=payload, timeout=120)
        assert r.status_code == 200, f"chat system test failed ({r.status_code}): {r.text}"
        data = r.json()
        content = data.get("message", {}).get("content", "").upper()
        assert "JAWOHL" in content, (
            f"Expected 'JAWOHL' in response (system prompt test). Got: {content!r}"
        )
