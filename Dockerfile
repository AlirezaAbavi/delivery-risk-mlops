# Delivery-risk API service image. The trained model is pulled from MLflow
# at runtime via MLFLOW_TRACKING_URI; if the registry has no Staging model the
# service falls back to a mounted joblib model (MODEL_PATH) and finally to a bounded
# arithmetic baseline (see app/model_loader.py). It never hard-fails on startup and
# no model or data is baked into the image.
# Python 3.12 to match the pipeline/training environment (Airflow's pipeline-venv
# runs on Python 3.12). Serving and training MUST share a Python version: sklearn
# models are pickled with native (numpy) state that cannot be safely unpickled across
# CPython versions, and a mismatch segfaults the loader (uncatchable in Python).
FROM python:3.12-slim

# Unbuffered stdout so JSON logs reach the collector promptly; no .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first, as their own layer, BEFORE copying source. Docker caches layers by
# their inputs, so as long as requirements.txt is unchanged this expensive pip step is
# reused — editing app code only busts the cheap COPY layer below, not the install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the API package is needed at runtime (see .dockerignore).
COPY app ./app

# Run as an unprivileged user (isolation / security hygiene).
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8112

# Container-native liveness: Docker/orchestrators poll /health and mark the container
# unhealthy after 3 failed checks. start-period gives the model time to load before the
# first check counts against us. We shell out to stdlib urllib so no extra tool (curl) is
# required in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8112/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8112"]
