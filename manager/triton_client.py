"""
Async HTTP client for Triton Inference Server's model management REST API.

Triton must be started with --model-control-mode=explicit so that models
are not loaded automatically at startup and can be loaded/unloaded at runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


class TritonClientError(Exception):
    """Raised when a Triton API call fails."""


class TritonClient:
    """Thin async wrapper around Triton's HTTP model management endpoints."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        # Normalize trailing slash
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Server health
    # ------------------------------------------------------------------

    async def is_server_live(self) -> bool:
        """Return True if Triton's /v2/health/live endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self._base}/v2/health/live")
            return r.status_code == 200
        except Exception:
            return False

    async def is_server_ready(self) -> bool:
        """Return True if Triton reports itself ready."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self._base}/v2/health/ready")
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Model repository index
    # ------------------------------------------------------------------

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Return the model repository index.

        Each entry is a dict with at least ``name`` and ``state`` keys.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base}/v2/repository/index")
        if r.status_code != 200:
            raise TritonClientError(
                f"repository/index failed ({r.status_code}): {r.text}"
            )
        return r.json() or []

    # ------------------------------------------------------------------
    # Model load / unload
    # ------------------------------------------------------------------

    async def load_model(self, model_name: str) -> None:
        """
        Ask Triton to load *model_name* from the model repository.

        Raises :class:`TritonClientError` if the request fails.
        """
        url = f"{self._base}/v2/repository/models/{model_name}/load"
        logger.info("Loading model %r via Triton API ...", model_name)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url)
        if r.status_code != 200:
            raise TritonClientError(
                f"load model {model_name!r} failed ({r.status_code}): {r.text}"
            )
        logger.info("Model %r loaded successfully.", model_name)

    async def unload_model(self, model_name: str) -> None:
        """
        Ask Triton to unload *model_name* and free its GPU memory.

        Raises :class:`TritonClientError` if the request fails.
        """
        url = f"{self._base}/v2/repository/models/{model_name}/unload"
        logger.info("Unloading model %r via Triton API ...", model_name)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url)
        if r.status_code != 200:
            raise TritonClientError(
                f"unload model {model_name!r} failed ({r.status_code}): {r.text}"
            )
        logger.info("Model %r unloaded successfully.", model_name)

    # ------------------------------------------------------------------
    # Model readiness
    # ------------------------------------------------------------------

    async def is_model_ready(self, model_name: str) -> bool:
        """Return True if *model_name* is currently loaded and ready."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{self._base}/v2/models/{model_name}/ready"
                )
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Inference proxy (generate endpoint)
    # ------------------------------------------------------------------

    async def generate(
        self, model_name: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward a generate request to Triton's generate extension endpoint.

        ``payload`` should contain at least ``text_input`` and optionally a
        ``parameters`` dict.  The raw JSON response body is returned.
        """
        url = f"{self._base}/v2/models/{model_name}/generate"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url, json=payload)
        if r.status_code != 200:
            raise TritonClientError(
                f"generate on {model_name!r} failed ({r.status_code}): {r.text}"
            )
        return r.json()
