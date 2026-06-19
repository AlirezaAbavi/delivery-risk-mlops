# Delivery capstone starter

This starter implements the required FastAPI contract, Prometheus metrics, API
test, and an Airflow workflow skeleton. The baseline score uses purchase-time
inputs only: delivery outcome, review, and actual-delivery fields are excluded
to prevent temporal leakage.

## Local setup and tests

Create the project virtual environment from the shared offline wheelhouse:

```bash
cd ~/project
python3 -m venv .venv
.venv/bin/python -m pip install --no-index \
  --find-links /opt/MLOps/MlOps/project/capstone_stack/wheelhouse \
  -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Run the API manually when required:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8112
```

The deployed Delivery API is available at `http://127.0.0.1:8112`; use
`/health` for availability, `/metrics` for Prometheus metrics, and `/predict`
for a prediction request. MLflow for Delivery is available locally at
`http://127.0.0.1:5312`.

## Version control and submission

Use GitLab for source control and submission:

```bash
git clone ssh://git@localhost:2224/delivery-mlops-capstone/delivery-project.git
```

Commit your source, tests, documentation, and dependency manifest to your own
group project. Do not place credentials, `.venv`, data exports, or generated
artifacts in Git.

Grafana is available at `http://localhost:3010`. Use the Delivery account
issued by the course administrator to view the Delivery organization and its
Prometheus datasource.
