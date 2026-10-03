# Serving image for the CardioRisk API.
# Only runtime deps are installed (no MLflow/Jupyter), which keeps the image small
# and the attack surface low. The model is baked in so a tag == a model version.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY models/model.joblib models/metadata.json models/

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin cardio
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# One worker per container: Prometheus counters live in process memory, so we
# scale with replicas (Kubernetes) instead of in-process workers.
CMD ["uvicorn", "cardiorisk.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
