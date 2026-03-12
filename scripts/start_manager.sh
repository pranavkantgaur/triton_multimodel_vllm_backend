#!/usr/bin/env bash
# start_manager.sh – Launch the FastAPI model-manager service.
#
# Usage:
#   ./scripts/start_manager.sh
#
# Environment variables:
#   MANAGER_CONFIG  – Path to models.yaml (default: config/models.yaml)
#   MANAGER_HOST    – Bind address (default: 0.0.0.0)
#   MANAGER_PORT    – Bind port     (default: 8080)
#   TRITON_HTTP_URL – Triton HTTP URL (default: http://localhost:8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

export MANAGER_CONFIG="${MANAGER_CONFIG:-${REPO_ROOT}/config/models.yaml}"
MANAGER_HOST="${MANAGER_HOST:-0.0.0.0}"
MANAGER_PORT="${MANAGER_PORT:-8080}"

cd "${REPO_ROOT}"

echo "Starting multi-model LLM manager on ${MANAGER_HOST}:${MANAGER_PORT} ..."
echo "  Config : ${MANAGER_CONFIG}"

uvicorn api.app:app \
  --host "${MANAGER_HOST}" \
  --port "${MANAGER_PORT}" \
  --log-level info
