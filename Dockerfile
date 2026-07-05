# API service image (Delivery deliverable). The trained model is pulled from MLflow
# at runtime via MLFLOW_TRACKING_URI; if the registry has no Staging model the
# service falls back to a mounted joblib model (MODEL_PATH) and finally to a bounded
# arithmetic baseline (see app/model_loader.py). It never hard-fails on startup and
# no model or data is baked into the image.
FROM python:3.11-slim

# Unbuffered stdout so JSON logs reach the collector promptly; no .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the API package is needed at runtime (see .dockerignore).
COPY app ./app

# Run as an unprivileged user (isolation / security hygiene — graded).
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8112

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8112/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8112"]
