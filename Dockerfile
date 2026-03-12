# Multi-model LLM Manager container image.
# This runs the FastAPI manager service only; Triton runs in a separate
# container (see docker-compose.yml).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manager/ ./manager/
COPY api/ ./api/
COPY config/ ./config/

# Allow config path to be overridden at runtime via environment variable.
ENV MANAGER_CONFIG=/app/config/models.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
