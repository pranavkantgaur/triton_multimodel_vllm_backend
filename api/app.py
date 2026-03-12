"""FastAPI application for the multi-model LLM manager.

Endpoints
---------
GET  /health                  – liveness check
GET  /v1/models               – list all configured and loaded models
POST /v1/models/{name}/load   – ensure a model is loaded (evicts LRU if needed)
POST /v1/models/{name}/unload – explicitly unload a model
POST /v1/models/{name}/generate – run text generation (ensures model loaded first)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from manager.config import load_config
from manager.model_manager import (
    InsufficientVRAMError,
    ModelManager,
    ModelNotFoundError,
)
from manager.triton_client import TritonClient, TritonClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (populated in lifespan)
# ---------------------------------------------------------------------------

_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    if _manager is None:  # pragma: no cover
        raise RuntimeError("ModelManager not initialised.")
    return _manager


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _manager
    config = load_config(os.environ.get("MANAGER_CONFIG"))
    triton_client = TritonClient(config.triton.http_url)
    _manager = ModelManager(config, triton_client)
    logger.info("ModelManager initialised with %d model(s).", len(config.models))
    yield
    # Graceful shutdown – optionally unload all models
    if os.environ.get("UNLOAD_ON_SHUTDOWN", "false").lower() == "true":
        logger.info("Unloading all models on shutdown ...")
        await _manager.unload_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Model LLM Manager",
    description=(
        "Dynamic on-demand loading and unloading of multiple LLMs "
        "backed by Triton Inference Server + vLLM backend."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    name: str
    description: str
    vram_gb: float
    loaded: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    used_vram_gb: float
    free_vram_gb: float
    total_vram_budget_gb: float


class GenerateRequest(BaseModel):
    text_input: str
    parameters: Dict[str, Any] = {}


class GenerateResponse(BaseModel):
    model_name: str
    text_output: str


class StatusResponse(BaseModel):
    status: str
    model_name: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelsResponse, tags=["models"])
async def list_models():
    """List all configured models and their current load state."""
    mgr = get_manager()
    status = mgr.get_status()
    model_infos = []
    for name, cfg in mgr._config.models.items():
        model_infos.append(
            ModelInfo(
                name=name,
                description=cfg.description,
                vram_gb=cfg.vram_gb,
                loaded=name in status["loaded_models"],
            )
        )
    return ModelsResponse(
        models=model_infos,
        used_vram_gb=status["used_vram_gb"],
        free_vram_gb=status["free_vram_gb"],
        total_vram_budget_gb=status["total_vram_budget_gb"],
    )


@app.post("/v1/models/{model_name}/load", response_model=StatusResponse, tags=["models"])
async def load_model(model_name: str):
    """
    Ensure the specified model is loaded in Triton.

    If VRAM is insufficient the least-recently-used loaded model will be
    evicted automatically.
    """
    mgr = get_manager()
    try:
        await mgr.ensure_loaded(model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientVRAMError as exc:
        raise HTTPException(status_code=507, detail=str(exc))
    except TritonClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return StatusResponse(status="loaded", model_name=model_name)


@app.post("/v1/models/{model_name}/unload", response_model=StatusResponse, tags=["models"])
async def unload_model(model_name: str):
    """Explicitly unload the specified model and reclaim its VRAM."""
    mgr = get_manager()
    try:
        await mgr.unload(model_name)
    except TritonClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return StatusResponse(status="unloaded", model_name=model_name)


@app.post(
    "/v1/models/{model_name}/generate",
    response_model=GenerateResponse,
    tags=["inference"],
)
async def generate(model_name: str, request: GenerateRequest):
    """
    Run text generation on the specified model.

    The model will be loaded automatically if it is not already loaded.
    The response is streamed from Triton's generate extension endpoint.
    """
    mgr = get_manager()
    try:
        await mgr.ensure_loaded(model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientVRAMError as exc:
        raise HTTPException(status_code=507, detail=str(exc))
    except TritonClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    payload: Dict[str, Any] = {
        "text_input": request.text_input,
        "parameters": {"stream": False, **request.parameters},
    }
    try:
        result = await mgr._triton.generate(model_name, payload)
    except TritonClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return GenerateResponse(
        model_name=model_name,
        text_output=result.get("text_output", ""),
    )
