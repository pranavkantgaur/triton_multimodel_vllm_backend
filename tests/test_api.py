"""
Integration-style tests for the FastAPI application.

The Triton client is fully mocked so no real Triton server is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from manager.config import AppConfig, ModelConfig, ResourceConfig, TritonConfig
from manager.model_manager import ModelManager
from manager.triton_client import TritonClientError


def _make_config() -> AppConfig:
    return AppConfig(
        models={
            "phi3-mini": ModelConfig(
                name="phi3-mini",
                model_path="/models/phi3-mini",
                vram_gb=8.0,
                description="Phi-3 Mini",
            ),
            "mistral-7b": ModelConfig(
                name="mistral-7b",
                model_path="/models/mistral-7b",
                vram_gb=14.0,
                description="Mistral 7B",
            ),
        },
        triton=TritonConfig(http_url="http://localhost:8000"),
        resources=ResourceConfig(total_vram_gb=24.0, reserved_vram_gb=2.0),
    )


def _make_triton_mock():
    mock = MagicMock()
    mock.load_model = AsyncMock()
    mock.unload_model = AsyncMock()
    mock.generate = AsyncMock(
        return_value={"text_output": "Paris is the capital of France."}
    )
    return mock


@pytest.fixture()
def client():
    """Return a TestClient with mocked ModelManager and TritonClient."""
    cfg = _make_config()
    triton_mock = _make_triton_mock()

    # Patch load_config and TritonClient so the lifespan builds our mocks.
    with (
        patch("api.app.load_config", return_value=cfg),
        patch("api.app.TritonClient", return_value=triton_mock),
    ):
        import api.app as app_module

        with TestClient(app_module.app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# List models
# ---------------------------------------------------------------------------


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    names = {m["name"] for m in data["models"]}
    assert names == {"phi3-mini", "mistral-7b"}
    assert data["total_vram_budget_gb"] == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def test_load_model(client):
    r = client.post("/v1/models/phi3-mini/load")
    assert r.status_code == 200
    assert r.json()["status"] == "loaded"


def test_load_unknown_model_returns_404(client):
    r = client.post("/v1/models/unknown-xyz/load")
    assert r.status_code == 404


def test_load_model_triton_error_returns_502(client):
    import api.app as app_module

    app_module._manager._triton.load_model = AsyncMock(
        side_effect=TritonClientError("boom")
    )
    r = client.post("/v1/models/phi3-mini/load")
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------


def test_unload_model(client):
    # First load it so it's tracked
    client.post("/v1/models/phi3-mini/load")
    r = client.post("/v1/models/phi3-mini/unload")
    assert r.status_code == 200
    assert r.json()["status"] == "unloaded"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def test_generate(client):
    r = client.post(
        "/v1/models/phi3-mini/generate",
        json={"text_input": "What is the capital of France?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "phi3-mini"
    assert "Paris" in data["text_output"]


def test_generate_unknown_model_returns_404(client):
    r = client.post(
        "/v1/models/nonexistent/generate",
        json={"text_input": "Hello"},
    )
    assert r.status_code == 404
