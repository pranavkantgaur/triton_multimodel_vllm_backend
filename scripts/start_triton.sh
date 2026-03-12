#!/usr/bin/env bash
# start_triton.sh – Launch the Triton Inference Server with vLLM backend.
#
# Usage:
#   ./scripts/start_triton.sh [TRITON_VERSION]
#
# Environment variables:
#   TRITON_VERSION      – NGC container tag, e.g. 24.10 (default: 24.10)
#   MODEL_REPOSITORY    – Host path to the model repository (default: ./model_repository)
#   HF_MODELS_ROOT      – Host path where HuggingFace model weights live (default: /data/models)
#   GPUS                – GPU device spec passed to --gpus (default: all)
#   HTTP_PORT           – Triton HTTP port (default: 8000)
#   GRPC_PORT           – Triton gRPC port (default: 8001)
#   METRICS_PORT        – Triton metrics port (default: 8002)

set -euo pipefail

TRITON_VERSION="${1:-${TRITON_VERSION:-24.10}}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$(pwd)/model_repository}"
HF_MODELS_ROOT="${HF_MODELS_ROOT:-/data/models}"
GPUS="${GPUS:-all}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

IMAGE="nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-vllm-python-py3"

echo "Starting Triton ${TRITON_VERSION} with vLLM backend ..."
echo "  Model repository : ${MODEL_REPOSITORY}"
echo "  HF models root   : ${HF_MODELS_ROOT}"
echo "  Image            : ${IMAGE}"

docker run \
  --gpus "${GPUS}" \
  --rm \
  --net host \
  --shm-size 2G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MODEL_REPOSITORY}:/model_repository:ro" \
  -v "${HF_MODELS_ROOT}:/models:ro" \
  -p "${HTTP_PORT}:8000" \
  -p "${GRPC_PORT}:8001" \
  -p "${METRICS_PORT}:8002" \
  "${IMAGE}" \
  tritonserver \
    --model-repository=/model_repository \
    --model-control-mode=explicit \
    --log-verbose=1
