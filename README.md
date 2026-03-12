# triton_multimodel_vllm_backend

Dynamic, on-demand loading and unloading of multiple LLMs on a single GPU node
using [Triton Inference Server](https://github.com/triton-inference-server/server)
with the [vLLM backend](https://github.com/triton-inference-server/vllm_backend).

## Overview

Running several large language models simultaneously on limited GPU VRAM is
infeasible. This project solves the problem by keeping only the *currently
requested* model(s) in VRAM and automatically evicting the
least-recently-used (LRU) model when a new one is requested.

```
┌──────────────────────────────────────────┐
│          Client application              │
│   (REST API / CLI / Python script)       │
└──────────────────────┬───────────────────┘
                       │ HTTP  :8080
                       ▼
┌──────────────────────────────────────────┐
│          Multi-Model LLM Manager         │
│  • LRU eviction when VRAM is full        │
│  • Load / unload on demand               │
│  • Tracks VRAM budget per model          │
└──────────────────────┬───────────────────┘
                       │ HTTP  :8000
                       ▼
┌──────────────────────────────────────────┐
│     Triton Inference Server              │
│   (--model-control-mode=explicit)        │
│         vLLM backend                     │
│                                          │
│  llama3-8b  │  mistral-7b  │  phi3-mini  │
│  (loaded)   │  (unloaded)  │  (loaded)   │
└──────────────────────────────────────────┘
```

### Key features

| Feature | Details |
|---|---|
| **On-demand loading** | A model is loaded into VRAM only when first requested |
| **LRU eviction** | The least-recently-used model is evicted automatically when VRAM is full |
| **VRAM budget** | Configurable total VRAM and per-model estimates prevent over-allocation |
| **REST API** | Simple HTTP API to list, load, unload, and generate from any model |
| **Offline / on-prem** | Works entirely with locally downloaded HuggingFace weights; no internet required |
| **Docker Compose** | One-command deployment of Triton + manager |

---

## Why Triton? Design rationale and tradeoffs

### The alternative: managing vLLM processes directly

The obvious simpler approach is to skip Triton and instead spin up one
`vllm serve` subprocess per model, starting a new process when a model is
requested and killing it when VRAM is needed elsewhere.  This is a valid
design and has a real advantage: the [OpenAI-compatible REST
API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
built into `vllm serve` works out of the box with standard LLM client
libraries.

### Why Triton is still the better fit here

| Concern | With Triton + vLLM backend | vLLM processes only |
|---|---|---|
| **Load / unload without restart** | `POST /v2/repository/models/{name}/load\|unload` on the *running* server; fast and clean | Must `kill` and re-`spawn` a full Python process; slower, risks port leaks |
| **Single endpoint** | One gRPC+HTTP address for all models | Each model runs on its own port; clients must track which port each model is on |
| **GPU memory release** | Triton + vLLM cooperate to free GPU context on unload; verified by the `is_model_ready` check | Memory is freed only after the OS collects the dead process; timing is non-deterministic |
| **Metrics & observability** | Triton exposes a unified Prometheus metrics endpoint covering all loaded models plus per-model vLLM stats | Each vLLM server has its own metrics endpoint; aggregation requires extra tooling |
| **Request queuing & health** | Triton handles per-model request queuing, health probes, and graceful draining | Requires custom logic or a process supervisor |
| **Production readiness** | Battle-tested NVIDIA runtime used in large-scale deployments | Growing but newer for on-prem, multi-model routing scenarios |

### Tradeoffs of using Triton

Triton does add complexity: the stack has more moving parts (Triton container,
vLLM backend plugin, manager service) compared to launching bare `vllm serve`
processes.  Clients must also use the KServe v2 inference protocol (or Triton's
`generate` extension endpoint) instead of the OpenAI-compatible API that many
libraries already support natively.

### Summary verdict

For **VRAM-constrained on-prem setups** where models must be swapped in and
out cleanly at runtime, Triton's `--model-control-mode=explicit` API offers
the most reliable and operationally predictable approach.  The programmatic
load/unload lifecycle, unified endpoint, and built-in metrics outweigh the
added complexity.  If OpenAI-client compatibility is the primary concern and
operational simplicity is preferred over observability, a pure `vllm serve`
process-management approach is a reasonable alternative.

---

## Prerequisites

- NVIDIA GPU(s) with adequate VRAM (e.g. A100 80 GB, RTX 4090 24 GB)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker / Docker Compose v2
- LLM weights downloaded locally (e.g. via `huggingface-cli download`)

---

## Quick start

### 1. Configure models

Edit `config/models.yaml`:

```yaml
models:
  llama3-8b:
    model_path: /data/models/llama3-8b   # absolute path on the host
    vram_gb: 16
    gpu_memory_utilization: 0.85
    tensor_parallel_size: 1

  mistral-7b:
    model_path: /data/models/mistral-7b
    vram_gb: 14

resources:
  total_vram_gb: 24    # total GPU VRAM available
  reserved_vram_gb: 2  # headroom / overhead
```

### 2. Generate the model repository

```bash
python scripts/setup_model_repo.py
```

This writes `model_repository/<model_name>/config.pbtxt` and
`model_repository/<model_name>/1/model.json` for every model in the YAML.
Re-run any time you edit `models.yaml`.

### 3. Start services

```bash
# Set the path where HuggingFace weights are stored on the host.
export HF_MODELS_ROOT=/data/models
export TRITON_VERSION=24.10

docker compose up
```

Triton starts on `:8000` / `:8001`, the manager on `:8080`.

---

## REST API

All manager endpoints are on port **8080**.

### List models

```bash
GET /v1/models
```

```bash
curl http://localhost:8080/v1/models
```

### Load a model

```bash
POST /v1/models/{model_name}/load
```

```bash
curl -X POST http://localhost:8080/v1/models/llama3-8b/load
```

If VRAM is full the LRU model is evicted automatically before loading.

### Unload a model

```bash
POST /v1/models/{model_name}/unload
```

```bash
curl -X POST http://localhost:8080/v1/models/llama3-8b/unload
```

### Generate text

```bash
POST /v1/models/{model_name}/generate
Content-Type: application/json

{
  "text_input": "What is the capital of France?",
  "parameters": {
    "temperature": "0.1",
    "max_tokens": "200"
  }
}
```

```bash
curl -X POST http://localhost:8080/v1/models/phi3-mini/generate \
  -H 'Content-Type: application/json' \
  -d '{"text_input": "What is the capital of France?"}'
```

The model will be loaded automatically if it is not already in VRAM.

---

## Example Python client

```bash
# List all available models
python client/example_client.py --list-models

# Load a specific model
python client/example_client.py --model mistral-7b --load

# Generate text
python client/example_client.py \
  --model mistral-7b \
  --prompt "Explain quantum entanglement in simple terms."

# Unload a model
python client/example_client.py --model mistral-7b --unload
```

---

## Repository layout

```
triton_multimodel_vllm_backend/
├── config/
│   └── models.yaml          # Model definitions & resource limits
├── model_repository/        # Triton model repository (generated)
│   ├── llama3-8b/
│   ├── mistral-7b/
│   └── phi3-mini/
├── manager/
│   ├── config.py            # YAML config loader
│   ├── triton_client.py     # Triton HTTP API wrapper
│   └── model_manager.py     # LRU eviction & VRAM budget logic
├── api/
│   └── app.py               # FastAPI application
├── client/
│   └── example_client.py    # CLI example client
├── scripts/
│   ├── setup_model_repo.py  # Generate model repository from models.yaml
│   ├── start_triton.sh      # Launch Triton container
│   └── start_manager.sh     # Launch manager service
├── tests/
│   ├── test_model_manager.py
│   ├── test_api.py
│   └── test_config.py
├── Dockerfile               # Manager service container
├── docker-compose.yml       # Full stack deployment
└── requirements.txt
```

---

## Adding a new model

1. Add an entry to `config/models.yaml`.
2. Run `python scripts/setup_model_repo.py` to create the Triton config files.
3. If using Docker Compose, restart Triton: `docker compose restart triton`.
4. Load the model via the API: `POST /v1/models/<new_model>/load`.

---

## Configuration reference

### `config/models.yaml`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_path` | string | ✓ | — | Absolute path to the model weights *inside the Triton container* (i.e. under the `/models` volume mount) |
| `vram_gb` | float | ✓ | — | Estimated peak VRAM in GiB |
| `gpu_memory_utilization` | float | | 0.85 | Fraction of GPU memory vLLM may use |
| `tensor_parallel_size` | int | | 1 | Number of GPUs for tensor-parallel inference |
| `enforce_eager` | bool | | false | Disable CUDA graph capture (slower, lower VRAM) |
| `description` | string | | "" | Human-readable description |

### Resource budget

| Field | Description |
|---|---|
| `resources.total_vram_gb` | Total VRAM available across all GPUs |
| `resources.reserved_vram_gb` | VRAM kept as overhead; effective budget = total − reserved |

---

## Running tests

```bash
pip install -r requirements.txt
pytest
```

All tests use mocked Triton clients — no GPU or running server required.

---

## How it works

### Triton explicit model control

Triton is started with `--model-control-mode=explicit`, which means:
- **No model is loaded at startup.**
- Models are loaded/unloaded via the HTTP management API:
  - `POST /v2/repository/models/{name}/load`
  - `POST /v2/repository/models/{name}/unload`

### LRU eviction

When a requested model does not fit in the remaining VRAM budget, the manager
evicts loaded models in **least-recently-used** order — i.e. the model that
has not been used for the longest time is unloaded first — until enough VRAM
is freed.

### Concurrency safety

All state mutations inside `ModelManager` are protected by an
`asyncio.Lock`, making concurrent API calls safe without data races.