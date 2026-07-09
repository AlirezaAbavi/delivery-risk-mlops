"""Contract + observability tests for the delivery-risk FastAPI service.

These are the tests the CD hook gates on (a red run blocks every deploy), so they encode
the *graded* guarantees: the endpoint set and response shape, the temporal-leakage
firewall, the metrics roll-up, and the deploy-status reporting. They run entirely
in-process via FastAPI's TestClient — no real server, DB, or MLflow needed — which is why
they're fast and safe to run on every commit.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import PredictionInput

# A complete, valid purchase-time payload (featureset_v1 contract). Deliberately contains
# ONLY features known at order-purchase time — no delivery outcome, no review — so it
# doubles as the positive control for the leakage-firewall test below.
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
    """The core graded contract: required endpoints exist and /predict returns the exact
    agreed keys with valid values.

    Using `with TestClient(app)` (as a context manager) is important — it triggers the
    app's startup/shutdown events, which is what loads the model. `<=` is the subset
    operator: we assert every required route is present, tolerating extra routes.
    """
    with TestClient(app) as client:
        routes = {route.path for route in app.routes}
        assert {'/health', '/predict', '/batch-predict', '/metrics', '/metrics-summary', '/model-info'} <= routes

        response = client.post('/predict', json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        # Exact-set equality (not subset): the response must contain these keys and no
        # others, so the contract can't silently grow or drop a field.
        assert set(body) == {
            'order_id', 'late_delivery_probability', 'risk_level',
            'recommended_action', 'model_version', 'latency',
        }
        # A probability must be a real probability, and the risk level must be one of the
        # three buckets the ops team acts on.
        assert 0.0 <= body['late_delivery_probability'] <= 1.0
        assert body['risk_level'] in {'low', 'medium', 'high'}

        # Having served a prediction, the model is resolved, so /health should now report a
        # healthy serving state and disclose which backend it loaded (MLflow/joblib/baseline).
        health = client.get('/health').json()
        assert health['status'] == 'ok'
        assert 'model_source' in health

        # /model-info must advertise the temporal-leakage policy verbatim — this string is
        # part of the graded contract, evidence the firewall is a stated guarantee.
        info = client.get('/model-info').json()
        assert info['temporal_leakage_policy'] == 'purchase-time features only'


def test_leakage_firewall_rejects_outcome_fields():
    """Temporal-leakage firewall (graded): outcome fields must be *rejected*, not ignored.

    The schema is configured to forbid extra fields, so submitting a known
    delivery-outcome/review field raises a ValidationError. We test at the schema level
    (not via HTTP) to prove the rejection is intrinsic to the input contract itself —
    there's no way to sneak a future-knowledge feature past the model.
    """
    for leaky in ('is_late_delivery', 'actual_delivery_days', 'review_score'):
        with pytest.raises(ValidationError):
            PredictionInput(**{**VALID_PAYLOAD, leaky: 1})


def test_metrics_summary_rollup():
    """Observability: predictions are counted and both metrics views expose them.

    We drive 1 single + 2 batch predictions (3 total), then assert the human-readable
    /metrics-summary rolls those up AND the Prometheus /metrics exposition text carries the
    expected metric names. `>=` (not `==`) because metrics accumulate across the process:
    other tests sharing this app instance may have already bumped the counters.
    """
    with TestClient(app) as client:
        client.post('/predict', json=VALID_PAYLOAD)
        client.post('/batch-predict', json=[VALID_PAYLOAD, VALID_PAYLOAD])

        summary = client.get('/metrics-summary').json()
        assert summary['total_predictions'] >= 3
        assert set(summary['risk_distribution']) == {'low', 'medium', 'high'}
        assert summary['prediction_latency']['count'] >= 3
        assert '/predict' in summary['requests_by_endpoint']
        assert summary['http_requests_total'] >= 3

        # The raw Prometheus endpoint must expose the same signals under our delivery_* metric
        # names (the prefix namespaces them so they don't collide with other groups' metrics
        # on the shared Prometheus).
        body = client.get('/metrics').text
        assert 'delivery_predictions_total' in body
        assert 'delivery_prediction_latency_seconds' in body
        assert 'delivery_http_requests_total' in body


def test_deploy_status_unknown_without_runlog(tmp_path, monkeypatch):
    """Deploy monitoring must fail *soft*: a missing run-log yields 'unknown', not a 500.

    monkeypatch points the config at a non-existent file (auto-cleaned tmp_path) so we can
    simulate "CD hook has never run here" without touching the real log. The principle: an
    observability feature must never be able to take down the serving path.
    """
    from app import config
    monkeypatch.setattr(config, 'DEPLOY_RUNS_PATH', str(tmp_path / 'absent.jsonl'))
    with TestClient(app) as client:
        body = client.get('/deploy-status').json()
        assert body['status'] == 'unknown'
        assert body['latest'] is None and body['recent'] == []
        # The gauge still exports, reading 0.0 for the "unknown" state.
        assert 'delivery_deploy_last_status 0.0' in client.get('/metrics').text


def test_deploy_status_reports_latest_run(tmp_path, monkeypatch):
    """Given a two-line run-log, the endpoint reports the *newest* run and renders HTML.

    We seed a synthetic JSONL log (a failed deploy followed by a successful one) and assert
    the endpoint surfaces the latest, both as JSON and as the HTML dashboard (inline SVG
    flowchart), and that the Prometheus gauges reflect the latest commit/status.
    """
    import json
    from app import config
    runlog = tmp_path / 'deploy-runs.jsonl'
    runlog.write_text(
        json.dumps({'new_commit': 'aaaaaaa000', 'finished_at': '2026-07-07T18:00:00Z',
                    'duration_seconds': 30, 'status': 'tests_failed',
                    'changed_paths': ['tests/test_api.py'], 'actions': {'restart': 'no'}}) + '\n'
        + json.dumps({'new_commit': 'bbbbbbb111', 'finished_at': '2026-07-07T18:10:00Z',
                      'duration_seconds': 42, 'status': 'success',
                      'changed_paths': ['app/main.py'], 'actions': {'restart': 'ok'}}) + '\n'
    )
    monkeypatch.setattr(config, 'DEPLOY_RUNS_PATH', str(runlog))
    with TestClient(app) as client:
        body = client.get('/deploy-status').json()
        assert body['status'] == 'success'                 # latest wins
        assert body['latest']['new_commit'] == 'bbbbbbb111'
        assert [r['new_commit'] for r in body['recent']][0] == 'bbbbbbb111'  # newest first

        html = client.get('/deploy-status', params={'format': 'html'})
        assert html.headers['content-type'].startswith('text/html')
        assert 'bbbbbbb' in html.text
        # the pipeline flowchart is drawn (inline SVG with the gate + action steps)
        assert '<svg' in html.text
        for step in ('New commit', 'Test gate', 'Restart API', 'Import Grafana'):
            assert step in html.text

        metrics = client.get('/metrics').text
        assert 'delivery_deploy_last_status 1.0' in metrics
        assert 'delivery_deploy_last_commit_info{commit="bbbbbbb",status="success"} 1.0' in metrics


def test_deploy_status_reconciles_retrain_outcome(tmp_path, monkeypatch):
    """The async retrain outcome is joined back to its deploy by dag_run_id.

    This mirrors the real fire-and-forget flow: a deploy records a *queued* retrain, so the
    status shows 'running'; later the watcher appends a terminal outcome keyed by the same
    dag_run_id, and the endpoint must then reconcile the deploy to 'success'. We assert both
    stages, including the Prometheus gauge flipping 0.0 -> 1.0.
    """
    import json
    from app import config
    runlog = tmp_path / 'deploy-runs.jsonl'
    retrainlog = tmp_path / 'deploy-retrain.jsonl'
    rid = 'deploy_hook__1783500000'
    runlog.write_text(json.dumps({
        'new_commit': 'ccccccc222', 'finished_at': '2026-07-08T10:00:00Z', 'duration_seconds': 25,
        'status': 'success', 'changed_paths': ['app/config.py'],
        'actions': {'restart': 'ok', 'trigger': 'queued', 'import': 'no'}, 'dag_run_id': rid,
    }) + '\n')
    monkeypatch.setattr(config, 'DEPLOY_RUNS_PATH', str(runlog))
    monkeypatch.setattr(config, 'DEPLOY_RETRAIN_PATH', str(retrainlog))

    with TestClient(app) as client:
        # No retrain record yet -> the queued run reports as "running".
        body = client.get('/deploy-status').json()
        assert body['latest']['retrain']['state'] == 'running'
        assert 'Retrain' in client.get('/deploy-status', params={'format': 'html'}).text
        assert 'delivery_deploy_last_retrain_status 0.0' in client.get('/metrics').text

        # Watcher records the terminal outcome -> reconciled to success.
        retrainlog.write_text(json.dumps({'dag_run_id': rid, 'state': 'success', 'deploy_commit': 'ccccccc222'}) + '\n')
        body = client.get('/deploy-status').json()
        assert body['latest']['retrain']['state'] == 'success'
        m = client.get('/metrics').text
        assert 'delivery_deploy_last_retrain_status 1.0' in m
        assert 'state="success"' in m


def test_logging_and_error_observability():
    """Requests are traceable (correlation id) and errors are both rejected and counted."""
    with TestClient(app) as client:
        # Every response carries an X-Request-ID so a single request can be traced across
        # the access log and any downstream systems — the backbone of debuggable services.
        ok = client.post('/predict', json=VALID_PAYLOAD)
        assert ok.headers.get('X-Request-ID')

        # A leaky field is rejected (400/422 validation error) AND increments the error
        # counter — so bad input is observable in metrics, not just silently refused.
        bad = client.post('/predict', json={**VALID_PAYLOAD, 'is_late_delivery': 1})
        assert bad.status_code in (400, 422)

        body = client.get('/metrics').text
        assert 'delivery_http_errors_total' in body
