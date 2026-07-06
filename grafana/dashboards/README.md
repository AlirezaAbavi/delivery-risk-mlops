# Shared-Grafana dashboards (course Prometheus)

Dashboards for the **shared course Grafana** at `http://localhost:3010`
(delivery org, role Editor). These target the shared **Prometheus** datasource
(`capstone-prometheus`) — the only datasource available in the shared Grafana.

## `delivery_risk_prometheus.json`

Service health + model output for the delivery delivery-risk API, read from the
shared Prometheus (datasource uid `capstone-prometheus`, the only datasource in
the shared Grafana). Covers every metric the brief requires: request count,
error count, latency (p50/p95/p99), prediction count, and risk-level
distribution — plus API-up and model-loaded status.

### Import
1. Log in to `http://localhost:3010` as `delivery`.
2. Dashboards → **New → Import** → *Upload JSON file* → this file.
3. Select the **delivery** folder, keep datasource = **Capstone Prometheus**.

### Two things to know
- **Instance pinning is mandatory.** Several groups emit *unprefixed* `delivery_*`
  metrics into the same Prometheus, so every query is filtered by the
  `$instance` template variable (default `host.docker.internal:8112`, delivery's
  API port). Without it, panels would blend other groups' data.
- **Some panels need the current API deployed.** As of writing, `:8112` runs an
  older build that only exposes `delivery_prediction_requests_total` and
  `delivery_prediction_latency_seconds` — so **API-up** and **latency** panels have
  data immediately, while request/error/prediction-count/risk-distribution
  panels stay empty until the current `app/` (with the full `app/metrics.py`
  contract) is redeployed to `:8112`.

All PromQL in this dashboard has been validated against the live Prometheus
(`localhost:9091`) — every expression parses; empty panels are a
data-availability gap, not a query error.
