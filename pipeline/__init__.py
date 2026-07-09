"""Data/ML pipeline: raw load -> features -> train/register -> batch -> monitor.

Kept separate from ``app/`` (the serving layer). Every module is env-driven via
``.env`` so the same code runs from the CLI and under Airflow in the compose stack.
"""
