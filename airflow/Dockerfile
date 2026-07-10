# Airflow image for the delivery-risk stack.
#
# The pipeline modules pin their own dependency stack (notably SQLAlchemy 2.x, which
# conflicts with Airflow's SQLAlchemy 1.4). To avoid breaking Airflow, we install the
# project requirements into a SEPARATE virtualenv and run every pipeline step with that
# interpreter (PIPELINE_PYTHON in docker-compose.yaml). Airflow's own environment is
# left untouched, so its metadata DB and scheduler keep working.
FROM apache/airflow:2.10.3

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /home/airflow/pipeline-venv \
 && /home/airflow/pipeline-venv/bin/pip install --no-cache-dir --upgrade pip \
 && /home/airflow/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
