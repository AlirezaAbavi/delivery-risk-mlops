from datetime import datetime
from time import perf_counter
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

app = FastAPI(title='MLOps Delivery delivery-risk API', version='1.0.0')
REQUESTS = Counter('delivery_prediction_requests_total', 'Prediction requests', ['risk_level'])
LATENCY = Histogram('delivery_prediction_latency_seconds', 'Prediction latency')

class PredictionInput(BaseModel):
    order_id: str
    item_total: float = Field(ge=0)
    freight_total: float = Field(ge=0)
    estimated_days: float = Field(gt=0, le=90)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_weekday: int = Field(ge=0, le=6)

def predict(payload: PredictionInput) -> dict:
    # Reference baseline: bounded, explainable risk score; replace with MLflow Staging model in production.
    score = min(0.98, max(0.01, 0.06 + 0.018 * payload.estimated_days + 0.0012 * payload.freight_total + 0.012 * (payload.purchase_hour >= 18)))
    level: Literal['low', 'medium', 'high'] = 'high' if score >= .55 else 'medium' if score >= .25 else 'low'
    action = {'low': 'monitor normally', 'medium': 'confirm carrier capacity', 'high': 'prioritize fulfillment intervention'}[level]
    REQUESTS.labels(level).inc()
    return {'order_id': payload.order_id, 'late_delivery_probability': round(score, 4), 'risk_level': level, 'recommended_action': action, 'model_version': 'reference-baseline', 'latency': 0.0}

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/model-info')
def model_info(): return {'name': 'reference-baseline', 'stage': 'Staging', 'temporal_leakage_policy': 'purchase-time features only'}

@app.post('/predict')
def single_prediction(payload: PredictionInput):
    started = perf_counter()
    response = predict(payload)
    response['latency'] = round(perf_counter() - started, 6)
    return response

@app.post('/batch-predict')
def batch_prediction(payloads: list[PredictionInput]): return [single_prediction(payload) for payload in payloads]

@app.get('/metrics-summary')
def metrics_summary(): return {'generated_at': datetime.utcnow().isoformat() + 'Z', 'service': 'reference-baseline'}

@app.get('/metrics')
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
