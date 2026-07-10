# Grafana dashboards

## `delivery_risk_prometheus.json`

Service health + model output for the delivery-risk API, read from Prometheus. Covers request
count, error count, latency (p50/p95/p99), prediction count, and risk-level distribution — plus
API-up, model-loaded status, and the deployment row.

### How it loads

The dashboard is **auto-provisioned** — no manual import. `docker-compose.yaml` mounts this
directory into Grafana and `grafana/provisioning/` wires up:

- the **Prometheus datasource** (uid `prometheus`, pointing at `http://prometheus:9090`), and
- a dashboard provider that loads every JSON here.

So `docker compose up -d` is all you need; the dashboard appears in Grafana at
`http://localhost:3000`.

### Note

Every panel is filtered by the `$instance` template variable (default `api:8112`). Panels that
depend on prediction traffic (request/error/prediction counts, risk distribution) stay empty until
the API has served some requests — run `make bootstrap` and hit `/predict` to populate them.
