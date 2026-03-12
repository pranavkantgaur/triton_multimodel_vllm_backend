"""
Configuration loader for the model manager.

Reads config/models.yaml (or a path overridden via the MANAGER_CONFIG env var)
and exposes strongly-typed dataclasses for the rest of the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class ModelConfig:
    name: str
    model_path: str
    vram_gb: float
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = False
    description: str = ""


@dataclass
class TritonConfig:
    http_url: str = "http://localhost:8000"
    grpc_url: str = "localhost:8001"
    model_repository: str = "/model_repository"


@dataclass
class ResourceConfig:
    total_vram_gb: float = 24.0
    reserved_vram_gb: float = 2.0

    @property
    def available_vram_gb(self) -> float:
        return self.total_vram_gb - self.reserved_vram_gb


@dataclass
class AppConfig:
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    triton: TritonConfig = field(default_factory=TritonConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and parse the YAML configuration file."""
    config_path = Path(
        path or os.environ.get("MANAGER_CONFIG", str(_DEFAULT_CONFIG_PATH))
    )
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    models: Dict[str, ModelConfig] = {}
    for name, spec in (raw.get("models") or {}).items():
        models[name] = ModelConfig(
            name=name,
            model_path=spec["model_path"],
            vram_gb=float(spec.get("vram_gb", 0)),
            tensor_parallel_size=int(spec.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(spec.get("gpu_memory_utilization", 0.85)),
            enforce_eager=bool(spec.get("enforce_eager", False)),
            description=str(spec.get("description", "")),
        )

    triton_raw = raw.get("triton", {})
    triton = TritonConfig(
        http_url=triton_raw.get("http_url", "http://localhost:8000"),
        grpc_url=triton_raw.get("grpc_url", "localhost:8001"),
        model_repository=triton_raw.get("model_repository", "/model_repository"),
    )

    res_raw = raw.get("resources", {})
    resources = ResourceConfig(
        total_vram_gb=float(res_raw.get("total_vram_gb", 24)),
        reserved_vram_gb=float(res_raw.get("reserved_vram_gb", 2)),
    )

    return AppConfig(models=models, triton=triton, resources=resources)
