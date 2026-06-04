import pytest
from fastapi.testclient import TestClient
from server import app
import config
from unittest.mock import patch

@pytest.fixture
def client():
    return TestClient(app)

def test_api_key_required(client):
    with patch('config.API_KEY', 'test_key'):
        response = client.get("/api/tags")
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"

def test_api_key_valid(client):
    with patch('config.API_KEY', 'test_key'):
        response = client.get("/api/tags", headers={"Authorization": "Bearer test_key"})
        assert response.status_code == 200

def test_api_key_disabled(client):
    with patch('config.API_KEY', None):
        response = client.get("/api/tags")
        assert response.status_code == 200
