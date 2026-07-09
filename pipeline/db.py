"""Database connectivity for the pipeline.

This is the single source of truth for *how the pipeline talks to Postgres*. Every
other pipeline module imports the engine and schema names from here rather than
building its own connection — so credentials, pooling, and the medallion schema
layout are configured in exactly one place.

Two configuration ideas are worth understanding:
  - Connection string: prefer a full ``DATABASE_URL`` if given, else assemble one from
    discrete ``POSTGRES_*`` parts. Both styles of ``.env`` therefore "just work".
  - Medallion schemas: the server DB is organised into layers (raw -> staging ->
    features -> predictions -> monitoring). Each layer's schema name is env-driven and
    defaults to ``public``, so a simple single-schema local DB needs no extra config.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

# --- schema layout (medallion) --------------------------------------------
# "Medallion architecture" = data flows through named quality layers. We keep each
# layer in its own Postgres schema so responsibilities (and access) are separated.
# Every name falls back to ``PG_SCHEMA_DEFAULT`` (``public``): on a laptop with one
# schema everything collapses to ``public`` and the code is unchanged; on the server
# the env sets RAW_SCHEMA=raw, FEATURES_SCHEMA=features, and so on.
_DEFAULT_SCHEMA = os.getenv("PG_SCHEMA_DEFAULT", "public")
RAW_SCHEMA = os.getenv("RAW_SCHEMA", _DEFAULT_SCHEMA)                 # untouched source tables
STAGING_SCHEMA = os.getenv("STAGING_SCHEMA", _DEFAULT_SCHEMA)         # cleaned/intermediate
FEATURES_SCHEMA = os.getenv("FEATURES_SCHEMA", _DEFAULT_SCHEMA)       # model-ready featureset
PREDICTIONS_SCHEMA = os.getenv("PREDICTIONS_SCHEMA", _DEFAULT_SCHEMA) # batch scoring output
MONITORING_SCHEMA = os.getenv("MONITORING_SCHEMA", _DEFAULT_SCHEMA)   # drift/retrain records


def database_url() -> str:
    """Resolve the SQLAlchemy connection URL.

    A full ``DATABASE_URL`` wins if present (the explicit, unambiguous option).
    Otherwise we build the standard ``postgresql+psycopg2://user:pass@host:port/db``
    string from the individual ``POSTGRES_*`` variables, each with a sensible default.
    """
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
    """Return the process-wide, connection-pooled SQLAlchemy engine.

    ``lru_cache(maxsize=1)`` makes this a lazy singleton: the engine (and its
    connection pool) is created on first call and reused thereafter — you never want
    to build a new pool per query. ``pool_pre_ping=True`` tests a pooled connection
    with a lightweight ping before handing it out, so a connection that silently died
    while idle (a common issue with long-lived pipelines) is transparently recycled
    instead of raising. ``future=True`` opts into SQLAlchemy 2.0 semantics.
    """
    return create_engine(database_url(), future=True, pool_pre_ping=True)


def ping() -> bool:
    """Return True if a trivial ``SELECT 1`` succeeds — a cheap connectivity check.

    Used by CLIs / smoke tests to confirm the DB is reachable before doing real work.
    """
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1


def ensure_schema(name: str) -> None:
    """Create the schema if it doesn't already exist (idempotent).

    Idempotency matters because Airflow may retry a task: ``CREATE SCHEMA IF NOT
    EXISTS`` is safe to run any number of times. ``begin()`` wraps it in a transaction
    that auto-commits on success. The name is quoted to be safe against odd schema
    names (it comes from our own config, not user input).
    """
    with get_engine().begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}"'))
