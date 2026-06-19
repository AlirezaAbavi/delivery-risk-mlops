# MLOps MLOps capstone: delivery-risk operations

Build an MLOps platform for an online marketplace that predicts late-delivery
risk early enough for an operations team to intervene. The supplied Olist CSVs
cover orders, order items, customers, sellers, products, payments, reviews and
geolocation. Predictions must be made while an order remains operationally
actionable; features that reveal its eventual outcome will be penalized.

Your platform must load the CSVs into Postgres, orchestrate loading/features,
training, batch prediction, monitoring, drift detection and retraining through
Airflow, track at least two credible model candidates in MLflow, and register a
final model promoted to Staging. Deliver a FastAPI service with `GET /health`,
`GET /model-info`, `POST /predict`, `POST /batch-predict`, `GET
/metrics-summary`, and `GET /metrics`.

Each prediction returns `order_id`, `late_delivery_probability`, `risk_level`,
`recommended_action`, `model_version`, and `latency`. Expose appropriate
Prometheus metrics, create a Grafana dashboard, write a Dockerfile, and
demonstrate drift detection, retraining decision logic and the service through
the evaluator UI.

Grading evaluates isolation/security, data correctness, temporal validity,
model comparison, orchestration reliability, reproducibility, API contract,
observability, version-control hygiene, and the final operational demo.
