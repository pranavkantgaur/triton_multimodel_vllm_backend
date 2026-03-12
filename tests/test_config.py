"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from manager.config import load_config


FIXTURE_DIR = Path(__file__).parent


def test_load_default_config():
    """Verify the bundled models.yaml parses without error."""
    cfg = load_config()
    assert len(cfg.models) >= 1
    assert cfg.triton.http_url.startswith("http")
    assert cfg.resources.available_vram_gb > 0


def test_model_config_fields():
    cfg = load_config()
    for name, model in cfg.models.items():
        assert model.name == name
        assert model.model_path, f"model_path missing for {name}"
        assert model.vram_gb > 0, f"vram_gb must be > 0 for {name}"
        assert 0 < model.gpu_memory_utilization <= 1.0
        assert model.tensor_parallel_size >= 1


def test_available_vram():
    cfg = load_config()
    assert cfg.resources.available_vram_gb == (
        cfg.resources.total_vram_gb - cfg.resources.reserved_vram_gb
    )
