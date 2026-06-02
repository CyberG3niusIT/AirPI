"""
pytest konfig + Session-Start Cleanup für Integrationstests.
Entfernt INTEGRATION_TEST_* Artefakte aus der echten Memory-DB
bevor neue Tests laufen — schützt vor Verschmutzung durch abgebrochene Läufe.
"""
import requests
import pytest

_BASE = "http://localhost:11435"
_CLEANUP_KEYWORDS = ["INTEGRATION_TEST_", "XGRAPHNODE", "XGRAPH"]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live-server integration tests")
    config.addinivalue_line("markers", "slow: tests that may take > 5s")


def pytest_sessionstart(session):
    """Bereinige Test-Artefakte aus vorherigen Läufen."""
    try:
        r = requests.get(f"{_BASE}/live", timeout=2)
        if r.status_code != 200:
            return
    except Exception:
        return  # Server nicht erreichbar — Unit-Tests laufen trotzdem

    for kw in _CLEANUP_KEYWORDS:
        try:
            requests.post(
                f"{_BASE}/memory/delete",
                json={"keyword": kw},
                timeout=5,
            )
        except Exception:
            pass
