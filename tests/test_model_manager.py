"""
Unit tests for ModelManager.

All Triton API calls are mocked so no real Triton server is required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from manager.config import AppConfig, ModelConfig, ResourceConfig, TritonConfig
from manager.model_manager import (
    InsufficientVRAMError,
    ModelManager,
    ModelNotFoundError,
)
from manager.triton_client import TritonClientError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(total_vram: float = 24.0, reserved: float = 2.0) -> AppConfig:
    """Create a minimal AppConfig with three models."""
    models = {
        "llama3-8b": ModelConfig(
            name="llama3-8b",
            model_path="/models/llama3-8b",
            vram_gb=16.0,
        ),
        "mistral-7b": ModelConfig(
            name="mistral-7b",
            model_path="/models/mistral-7b",
            vram_gb=14.0,
        ),
        "phi3-mini": ModelConfig(
            name="phi3-mini",
            model_path="/models/phi3-mini",
            vram_gb=8.0,
        ),
    }
    return AppConfig(
        models=models,
        triton=TritonConfig(http_url="http://localhost:8000"),
        resources=ResourceConfig(total_vram_gb=total_vram, reserved_vram_gb=reserved),
    )


def _make_manager(total_vram: float = 24.0, reserved: float = 2.0) -> ModelManager:
    cfg = _make_config(total_vram, reserved)
    triton = MagicMock()
    triton.load_model = AsyncMock()
    triton.unload_model = AsyncMock()
    return ModelManager(cfg, triton)


# ---------------------------------------------------------------------------
# Basic load / unload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_loaded_calls_triton():
    mgr = _make_manager()
    await mgr.ensure_loaded("phi3-mini")
    mgr._triton.load_model.assert_awaited_once_with("phi3-mini")
    assert "phi3-mini" in mgr.loaded_models


@pytest.mark.asyncio
async def test_ensure_loaded_cache_hit_does_not_reload():
    mgr = _make_manager()
    await mgr.ensure_loaded("phi3-mini")
    mgr._triton.load_model.reset_mock()
    await mgr.ensure_loaded("phi3-mini")
    mgr._triton.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_unload_removes_model():
    mgr = _make_manager()
    await mgr.ensure_loaded("phi3-mini")
    await mgr.unload("phi3-mini")
    mgr._triton.unload_model.assert_awaited_once_with("phi3-mini")
    assert "phi3-mini" not in mgr.loaded_models


@pytest.mark.asyncio
async def test_unload_noop_when_not_loaded():
    mgr = _make_manager()
    await mgr.unload("phi3-mini")
    mgr._triton.unload_model.assert_not_awaited()


# ---------------------------------------------------------------------------
# VRAM accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vram_accounting():
    mgr = _make_manager(total_vram=24.0, reserved=2.0)  # budget = 22 GiB
    assert mgr.free_vram_gb == pytest.approx(22.0)
    await mgr.ensure_loaded("phi3-mini")  # 8 GiB
    assert mgr.used_vram_gb == pytest.approx(8.0)
    assert mgr.free_vram_gb == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_loaded_unknown_model_raises():
    mgr = _make_manager()
    with pytest.raises(ModelNotFoundError):
        await mgr.ensure_loaded("nonexistent-model")


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lru_eviction_when_vram_full():
    # Budget = 20 GiB; llama3-8b (16) fills most of it.
    # Then mistral-7b (14) won't fit → llama must be evicted.
    mgr = _make_manager(total_vram=22.0, reserved=2.0)  # budget = 20 GiB
    await mgr.ensure_loaded("llama3-8b")  # uses 16 GiB, 4 GiB free
    assert mgr.free_vram_gb == pytest.approx(4.0)

    # mistral-7b needs 14 GiB → triggers LRU eviction of llama3-8b
    await mgr.ensure_loaded("mistral-7b")
    mgr._triton.unload_model.assert_awaited_with("llama3-8b")
    assert "llama3-8b" not in mgr.loaded_models
    assert "mistral-7b" in mgr.loaded_models


@pytest.mark.asyncio
async def test_lru_candidate_is_least_recently_used():
    """Touch model A after B; B should be evicted first."""
    mgr = _make_manager(total_vram=30.0, reserved=2.0)  # budget = 28 GiB
    # Load phi3-mini and mistral-7b (total = 22 GiB, 6 GiB free)
    await mgr.ensure_loaded("phi3-mini")   # LRU candidate initially
    await asyncio.sleep(0.01)              # ensure distinct timestamps
    await mgr.ensure_loaded("mistral-7b")  # LRU candidate becomes phi3-mini

    # Touch phi3-mini so mistral-7b is now the oldest
    await asyncio.sleep(0.01)
    await mgr.ensure_loaded("phi3-mini")   # updates last_used for phi3-mini

    # Now load llama3-8b (16 GiB): 22 + 16 = 38 > 28, need to evict.
    # mistral-7b (14 GiB) is LRU; after eviction: 8 + 16 = 24 ≤ 28 → fits.
    await mgr.ensure_loaded("llama3-8b")
    evicted = [call.args[0] for call in mgr._triton.unload_model.await_args_list]
    assert "mistral-7b" in evicted
    assert "phi3-mini" not in evicted


# ---------------------------------------------------------------------------
# Model too large even when alone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_too_large_raises_insufficient_vram():
    # Budget = 10 GiB; llama3-8b requires 16 GiB → impossible.
    mgr = _make_manager(total_vram=12.0, reserved=2.0)  # budget = 10 GiB
    with pytest.raises(InsufficientVRAMError):
        await mgr.ensure_loaded("llama3-8b")


# ---------------------------------------------------------------------------
# Triton client errors propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triton_load_error_propagates():
    mgr = _make_manager()
    mgr._triton.load_model.side_effect = TritonClientError("Triton unreachable")
    with pytest.raises(TritonClientError):
        await mgr.ensure_loaded("phi3-mini")
    # Model should NOT be tracked as loaded after a failed load.
    assert "phi3-mini" not in mgr.loaded_models


# ---------------------------------------------------------------------------
# unload_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unload_all():
    mgr = _make_manager(total_vram=30.0, reserved=2.0)
    await mgr.ensure_loaded("phi3-mini")
    await mgr.ensure_loaded("mistral-7b")
    await mgr.unload_all()
    assert mgr.loaded_models == []


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_status():
    mgr = _make_manager()
    status = mgr.get_status()
    assert set(status["available_models"]) == {"llama3-8b", "mistral-7b", "phi3-mini"}
    assert status["loaded_models"] == []
    assert status["used_vram_gb"] == 0.0
