"""
ModelManager – core logic for dynamic LLM load/unload with VRAM budgeting.

Design
------
* Triton is started with ``--model-control-mode=explicit`` so no model is
  loaded at startup.
* The manager maintains an in-memory registry of *loaded* models keyed by
  name plus a last-used timestamp for LRU eviction.
* When a model is requested (``ensure_loaded``):
    1. If it is already loaded, update LRU timestamp and return.
    2. Otherwise, check whether there is enough free VRAM budget.
    3. If not, evict the least-recently-used loaded model(s) until there is.
    4. Load the requested model via the Triton API.
* All state mutations are protected by an ``asyncio.Lock`` so concurrent
  API calls are safe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from manager.config import AppConfig, ModelConfig
from manager.triton_client import TritonClient, TritonClientError

logger = logging.getLogger(__name__)


@dataclass
class LoadedModelEntry:
    name: str
    vram_gb: float
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


class ModelNotFoundError(Exception):
    """The requested model name is not in the configuration."""


class InsufficientVRAMError(Exception):
    """The requested model is too large to fit even when all others are evicted."""


class ModelManager:
    """
    Manages the lifecycle of multiple LLMs on a single Triton/vLLM node.

    Parameters
    ----------
    config:
        Full application configuration loaded from models.yaml.
    triton_client:
        Pre-constructed :class:`TritonClient` instance.
    """

    def __init__(self, config: AppConfig, triton_client: TritonClient) -> None:
        self._config = config
        self._triton = triton_client
        self._loaded: Dict[str, LoadedModelEntry] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> List[str]:
        """All model names known from the configuration file."""
        return list(self._config.models.keys())

    @property
    def loaded_models(self) -> List[str]:
        """Names of models that are currently loaded in Triton."""
        return list(self._loaded.keys())

    @property
    def used_vram_gb(self) -> float:
        """Total VRAM (GiB) consumed by currently loaded models."""
        return sum(e.vram_gb for e in self._loaded.values())

    @property
    def free_vram_gb(self) -> float:
        """VRAM (GiB) still available within the configured budget."""
        return self._config.resources.available_vram_gb - self.used_vram_gb

    async def ensure_loaded(self, model_name: str) -> ModelConfig:
        """
        Guarantee that *model_name* is loaded and ready in Triton.

        Returns the :class:`ModelConfig` for the model so the caller can
        address Triton directly if needed.

        Raises
        ------
        ModelNotFoundError
            If *model_name* is not in the configuration.
        InsufficientVRAMError
            If the model cannot fit even after evicting everything else.
        TritonClientError
            If the Triton load/unload call fails.
        """
        if model_name not in self._config.models:
            raise ModelNotFoundError(
                f"Model {model_name!r} is not in the configuration."
            )

        async with self._lock:
            if model_name in self._loaded:
                self._loaded[model_name].touch()
                logger.debug("Model %r already loaded (cache hit).", model_name)
                return self._config.models[model_name]

            model_cfg = self._config.models[model_name]
            await self._make_room_for(model_cfg)
            await self._triton.load_model(model_name)
            self._loaded[model_name] = LoadedModelEntry(
                name=model_name, vram_gb=model_cfg.vram_gb
            )
            return model_cfg

    async def unload(self, model_name: str) -> None:
        """
        Explicitly unload *model_name* from Triton and free its VRAM budget.

        No-op if the model is not currently loaded.
        """
        async with self._lock:
            if model_name not in self._loaded:
                return
            await self._triton.unload_model(model_name)
            del self._loaded[model_name]

    async def unload_all(self) -> None:
        """Unload every currently loaded model."""
        async with self._lock:
            for name in list(self._loaded.keys()):
                try:
                    await self._triton.unload_model(name)
                except TritonClientError as exc:
                    logger.warning("Failed to unload %r: %s", name, exc)
                del self._loaded[name]

    def get_status(self) -> dict:
        """Return a snapshot of the manager state for health/status endpoints."""
        return {
            "available_models": self.available_models,
            "loaded_models": self.loaded_models,
            "used_vram_gb": round(self.used_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "total_vram_budget_gb": self._config.resources.available_vram_gb,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _make_room_for(self, model_cfg: ModelConfig) -> None:
        """
        Evict loaded models (LRU order) until *model_cfg* can fit in VRAM.

        Must be called while ``_lock`` is held.
        """
        available = self._config.resources.available_vram_gb
        if model_cfg.vram_gb > available:
            raise InsufficientVRAMError(
                f"Model {model_cfg.name!r} requires {model_cfg.vram_gb} GiB but "
                f"only {available} GiB is budgeted (even when nothing else is loaded)."
            )

        while self.free_vram_gb < model_cfg.vram_gb:
            victim = self._lru_candidate()
            if victim is None:
                # Should not happen after the check above, but guard anyway.
                raise InsufficientVRAMError(
                    f"Cannot make room for {model_cfg.name!r}: "
                    f"need {model_cfg.vram_gb} GiB, free {self.free_vram_gb} GiB."
                )
            logger.info(
                "Evicting LRU model %r (%.1f GiB) to make room for %r.",
                victim,
                self._loaded[victim].vram_gb,
                model_cfg.name,
            )
            await self._triton.unload_model(victim)
            del self._loaded[victim]

    def _lru_candidate(self) -> Optional[str]:
        """Return the name of the least-recently-used loaded model, or None."""
        if not self._loaded:
            return None
        return min(self._loaded, key=lambda n: self._loaded[n].last_used)
