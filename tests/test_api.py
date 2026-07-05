import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import PredictionInput

# A complete, valid purchase-time payload (featureset_v1 contract).
VALID_PAYLOAD = {
    'order_id': 'x-1',
    'payment_count': 1, 'payment_type_mode': 'credit_card', 'max_installments': 3.0,
    'order_item_count': 2, 'product_count': 2, 'seller_count': 1,
    'price_sum': 120.0, 'price_mean': 60.0, 'price_max': 70.0, 'price_std': 14.1,
    'freight_sum': 20.0, 'freight_mean': 10.0, 'freight_max': 12.0, 'freight_std': 2.8,
    'total_cost_sum': 140.0, 'total_cost_mean': 70.0, 'total_cost_max': 82.0,
    'product_category_count': 1, 'product_category_mode': 'cama_mesa_banho',
    'is_multi_category': False,
    'product_weight_g_mean': 800.0, 'product_weight_g_max': 900.0,
    'product_length_cm_mean': 30.0, 'product_length_cm_max': 35.0,
    'product_height_cm_mean': 10.0, 'product_height_cm_max': 12.0,
    'product_width_cm_mean': 20.0, 'product_width_cm_max': 22.0,
    'product_volume_mean': 6000.0, 'product_volume_max': 9240.0,
    'seller_state_mode': 'SP', 'seller_state_count': 1, 'seller_city_count': 1,
    'seller_zip_mode': 14000,
    'purchase_hour': 9, 'purchase_dayofweek': 1, 'is_weekend_purchase': 0,
    'purchase_month': 5, 'purchase_quarter': 2, 'is_month_end': 0,
    'estimated_delivery_days': 8, 'approval_delay_hours': 4.5,
    'shipping_limit_min_days': 3, 'shipping_window_days': 5, 'seller_margin_days': 2,
}


def test_health_and_contract():
    with TestClient(app) as client:
        routes = {route.path for route in app.routes}
        assert {'/health', '/predict', '/batch-predict', '/metrics', '/metrics-summary', '/model-info'} <= routes

        response = client.post('/predict', json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            'order_id', 'late_delivery_probability', 'risk_level',
            'recommended_action', 'model_version', 'latency',
        }
        assert 0.0 <= body['late_delivery_probability'] <= 1.0
        assert body['risk_level'] in {'low', 'medium', 'high'}

        # A prediction has resolved the model, so /health reports a serving state.
        health = client.get('/health').json()
        assert health['status'] == 'ok'
        assert 'model_source' in health

        info = client.get('/model-info').json()
        assert info['temporal_leakage_policy'] == 'purchase-time features only'


def test_leakage_firewall_rejects_outcome_fields():
    # Delivery-outcome / review fields must never be accepted as inputs.
    for leaky in ('is_late_delivery', 'actual_delivery_days', 'review_score'):
        with pytest.raises(ValidationError):
            PredictionInput(**{**VALID_PAYLOAD, leaky: 1})


def test_metrics_summary_rollup():
    with TestClient(app) as client:
        client.post('/predict', json=VALID_PAYLOAD)
        client.post('/batch-predict', json=[VALID_PAYLOAD, VALID_PAYLOAD])

        summary = client.get('/metrics-summary').json()
        assert summary['total_predictions'] >= 3
        assert set(summary['risk_distribution']) == {'low', 'medium', 'high'}
        assert summary['prediction_latency']['count'] >= 3
        assert '/predict' in summary['requests_by_endpoint']
        assert summary['http_requests_total'] >= 3

        body = client.get('/metrics').text
        assert 'delivery_predictions_total' in body
        assert 'delivery_prediction_latency_seconds' in body
        assert 'delivery_http_requests_total' in body


def test_logging_and_error_observability():
    with TestClient(app) as client:
        # Every response carries a correlation id.
        ok = client.post('/predict', json=VALID_PAYLOAD)
        assert ok.headers.get('X-Request-ID')

        # A leaky field is a 400 and is counted as an HTTP error.
        bad = client.post('/predict', json={**VALID_PAYLOAD, 'is_late_delivery': 1})
        assert bad.status_code in (400, 422)

        body = client.get('/metrics').text
        assert 'delivery_http_errors_total' in body
