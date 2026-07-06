"""Database connectivity for the pipeline.

Single source of truth for the SQLAlchemy engine. Prefers ``DATABASE_URL`` and
falls back to assembling one from the discrete ``POSTGRES_*`` parts, so either
style of ``.env`` works.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

# --- schema layout (medallion) --------------------------------------------
# Each layer's schema is env-driven. Defaults to PG_SCHEMA_DEFAULT (``public``)
# so a single-schema local dev DB keeps working with no config; the server's
# multi-schema DB sets RAW_SCHEMA=raw, FEATURES_SCHEMA=features, etc.
_DEFAULT_SCHEMA = os.getenv("PG_SCHEMA_DEFAULT", "public")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", _DEFAULT_SCHEMA)
STAGING_SCHEMA = os.getenv("STAGING_SCHEMA", _DEFAULT_SCHEMA)
FEATURES_SCHEMA = os.getenv("FEATURES_SCHEMA", _DEFAULT_SCHEMA)
PREDICTIONS_SCHEMA = os.getenv("PREDICTIONS_SCHEMA", _DEFAULT_SCHEMA)
MONITORING_SCHEMA = os.getenv("MONITORING_SCHEMA", _DEFAULT_SCHEMA)


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "delivery")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "delivery")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide pooled engine. ``pool_pre_ping`` survives idle disconnects."""
    return create_engine(database_url(), future=True, pool_pre_ping=True)


def ping() -> bool:
    """Return True if a trivial query succeeds; used by health checks / CLIs."""
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1


def ensure_schema(name: str) -> None:
    """Create the schema if it doesn't exist (idempotent, Airflow-safe)."""
    with get_engine().begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}"'))
