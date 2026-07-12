# Delivery-risk MLOps stack — common tasks.
#
#   make up          bring up the whole stack (postgres, mlflow, api, airflow, prometheus, grafana)
#   make bootstrap   run the pipeline once (load -> features -> train -> register -> predict -> monitor)
#   make down        stop the stack (keep volumes)
#   make clean       stop and remove volumes (wipes data/models)
#   make test        run the test suite inside the api image
#   make fetch-data  download the real Olist dataset from Kaggle (needs credentials)
#   make logs        tail all service logs
#
# DATA selects the pipeline's input: `sample` (default, committed synthetic data) or
# `full` (the Kaggle download fetched into olist_data/). Example:
#   make bootstrap DATA=full

DATA ?= sample
COMPOSE ?= docker compose

ifeq ($(DATA),full)
RAW_DATA_DIR := /opt/project/olist_data/olist_data
else
RAW_DATA_DIR := /opt/project/sample_data
endif

# Run a pipeline module in the airflow service's isolated venv (see airflow/Dockerfile).
PIPELINE_EXEC = $(COMPOSE) exec -e RAW_DATA_DIR=$(RAW_DATA_DIR) -T airflow /home/airflow/pipeline-venv/bin/python -m

.PHONY: up down clean bootstrap test fetch-data sample-data logs ps

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

# Full pipeline run. Each step reuses the module's own `python -m pipeline.X` entrypoint,
# exactly as the Airflow DAG runs them — so `make bootstrap` and a DAG trigger are equivalent.
bootstrap:
	@echo ">> using RAW_DATA_DIR=$(RAW_DATA_DIR)"
	$(PIPELINE_EXEC) pipeline.load_raw
	$(PIPELINE_EXEC) pipeline.features
	$(PIPELINE_EXEC) pipeline.train
	$(PIPELINE_EXEC) pipeline.register
	$(PIPELINE_EXEC) pipeline.batch_predict
	$(PIPELINE_EXEC) pipeline.monitor
	@echo ">> bootstrap complete — the API now serves the registered delivery-risk model"

# Run the contract/observability tests in the API image (no host Python needed).
test:
	$(COMPOSE) run --rm --no-deps -e LOAD_MODEL_ON_STARTUP=false -e PYTHONPATH=/app -v $(PWD)/tests:/app/tests api \
		sh -c "pip install -q pytest httpx && pytest -q tests"

# Regenerate the committed synthetic sample dataset.
sample-data:
	python scripts/make_sample_data.py

# Download the real Olist dataset from Kaggle into olist_data/ — runs in a container,
# so the host needs no Python or kaggle CLI. Credentials come from .env (see .env.example).
fetch-data:
	$(COMPOSE) run --rm fetch-data
