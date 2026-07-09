"""API smoke test — the DAG's ``api_smoke_test`` task.

Hits ``GET /health`` and ``POST /predict`` on the running delivery-risk service
and asserts a well-formed contract response. Exits non-zero (failing the Airflow
task) on any problem. The ``/predict`` body reuses the schema's own example so it
stays aligned with the purchase-time feature contract.

Usage:
    python -m pipeline.smoke_test            # tests http://127.0.0.1:8112
    API_BASE_URL=http://host:port python -m pipeline.smoke_test
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8112").rstrip("/")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "10"))

REQUIRED_KEYS = {
    "order_id", "late_delivery_probability", "risk_level",
    "recommended_action", "model_version", "latency",
}


def _get(path: str):
    """GET helper using only the stdlib (no requests) so the smoke test has no extra deps.

    Returns (http_status, parsed_json). We deliberately keep dependencies to zero here so
    this check can run in the leanest possible environment (e.g. a bare Airflow worker).
    """
    with urllib.request.urlopen(BASE + path, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode())


def _post(path: str, body: dict):
    """POST-JSON helper (stdlib only). Encodes the dict as a JSON body and parses the reply."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode())


def run() -> None:
    """Exercise the two endpoints that prove the service is genuinely serving predictions."""
    # 1) health — is the process up and reporting a status at all?
    status, health = _get("/health")
    assert status == 200, f"/health returned HTTP {status}"
    assert health.get("status"), f"/health payload missing status: {health}"
    print("health OK:", health)

    # 2) predict — the real test. Reuse the schema's *own* example payload as the request
    # body so this smoke test can never drift out of sync with the input contract: if the
    # feature contract changes, the example changes, and this test exercises the new shape.
    from app.schemas import PredictionInput

    payload = dict(PredictionInput.model_config["json_schema_extra"]["example"])
    status, pred = _post("/predict", payload)
    assert status == 200, f"/predict returned HTTP {status}"
    # Assert the response carries every field the API contract requires...
    missing = REQUIRED_KEYS - set(pred)
    assert not missing, f"/predict missing keys: {sorted(missing)}"
    # ...and that the probability is actually a valid probability, not a raw score or NaN.
    prob = pred["late_delivery_probability"]
    assert 0.0 <= float(prob) <= 1.0, f"probability out of range: {prob}"
    print("predict OK:", pred)
    print("SMOKE TEST PASSED")


def main() -> None:
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - any failure must fail the task
        # Turn *any* problem into a non-zero exit so the Airflow task goes red. For an HTTP
        # error we surface a snippet of the server's response body (truncated) — that
        # detail is usually what tells you *why* /predict rejected the request.
        detail = ""
        if isinstance(exc, urllib.error.HTTPError):
            detail = f" body={exc.read().decode(errors='replace')[:300]}"
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}{detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
