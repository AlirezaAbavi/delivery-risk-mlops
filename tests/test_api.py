from app.main import PredictionInput, app, health, single_prediction

def test_health_and_contract():
    routes = {route.path for route in app.routes}
    assert {'/health', '/predict', '/batch-predict', '/metrics', '/metrics-summary', '/model-info'} <= routes
    assert health()['status'] == 'ok'
    response = single_prediction(PredictionInput(
        order_id='x-1', item_total=100, freight_total=10,
        estimated_days=8, purchase_hour=9, purchase_weekday=1,
    ))
    assert set(response) == {
        'order_id', 'late_delivery_probability', 'risk_level',
        'recommended_action', 'model_version', 'latency',
    }
