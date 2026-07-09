"""Load the raw Olist CSVs into Postgres (idempotent).

Each table is fully replaced on every run so the step is safely re-runnable from
Airflow. Date columns are parsed to real timestamps so downstream feature code
can do temporal arithmetic without re-parsing. Only the tables needed for
late-delivery modelling are loaded (marketing/closed-deals are ignored).

Usage:
    python -m pipeline.load_raw                 # load all tables
    python -m pipeline.load_raw --tables orders order_items
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .db import RAW_SCHEMA, ensure_schema, get_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.load_raw")


# A small, declarative description of one CSV -> table load. Using a frozen dataclass
# (immutable, hashable) turns "which files map to which tables, and which columns are
# dates" into pure data the loader loops over — far clearer than a pile of if/elif.
@dataclass(frozen=True)
class RawTable:
    name: str          # destination Postgres table
    filename: str      # source CSV (without directory)
    date_columns: tuple[str, ...] = ()  # columns pandas should parse into real timestamps
    encoding: str = "utf-8"             # some Olist files ship with a BOM (see below)


# Destination table -> source file + which columns to parse as datetimes.
RAW_TABLES: tuple[RawTable, ...] = (
    RawTable(
        "orders", "olist_orders_dataset.csv",
        date_columns=(
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ),
    ),
    RawTable("order_items", "olist_order_items_dataset.csv", date_columns=("shipping_limit_date",)),
    RawTable("customers", "olist_customers_dataset.csv"),
    RawTable("sellers", "olist_sellers_dataset.csv"),
    RawTable("products", "olist_products_dataset.csv"),
    RawTable("order_payments", "olist_order_payments_dataset.csv"),
    RawTable(
        "order_reviews", "olist_order_reviews_dataset.csv",
        date_columns=("review_creation_date", "review_answer_timestamp"),
    ),
    # Header carries a UTF-8 BOM; utf-8-sig strips it cleanly. Table name mirrors
    # the source file (canonical Olist name) so the raw layer is unambiguous.
    RawTable("product_category_name_translation", "product_category_name_translation.csv", encoding="utf-8-sig"),
)


def _data_dir() -> Path:
    """Where the source CSVs live. Env-overridable so the same code finds the data on
    a laptop and on the server (which may mount it elsewhere)."""
    return Path(os.getenv("RAW_DATA_DIR", "olist_data/olist_data"))


def load_table(spec: RawTable, data_dir: Path) -> int:
    """Load one CSV fully into its raw Postgres table; return the row count.

    Key choices:
      - Fail loudly if the file is missing (a silent partial load would corrupt every
        downstream step).
      - ``parse_dates`` converts date columns to real timestamps *at load time* so the
        feature SQL can do date arithmetic without re-parsing strings.
      - ``if_exists="replace"`` makes the load idempotent: re-running drops and rebuilds
        the table, so an Airflow retry can't create duplicate rows.
      - ``chunksize`` + ``method="multi"`` batch the INSERTs, which is dramatically
        faster than row-by-row for the larger tables.
    """
    path = data_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"missing source CSV: {path}")

    frame = pd.read_csv(
        path,
        encoding=spec.encoding,
        parse_dates=list(spec.date_columns) or None,  # None => no date parsing
    )
    frame.to_sql(
        spec.name, get_engine(), schema=RAW_SCHEMA, if_exists="replace", index=False,
        chunksize=10_000, method="multi",
    )
    log.info("loaded %s.%s: %d rows, %d cols", RAW_SCHEMA, spec.name, len(frame), frame.shape[1])
    return len(frame)


def clear_schema(schema: str) -> int:
    """Drop every base table in ``schema`` so the raw layer reloads from a clean
    slate (removes stray/renamed tables from earlier manual loads). We own the
    tables (single group role), so the drops succeed without owning the schema."""
    eng = get_engine()
    with eng.begin() as conn:
        tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :s AND table_type = 'BASE TABLE'"
        ), {"s": schema}).scalars().all()
        for t in tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{t}" CASCADE'))
    log.info("cleared schema %s: dropped %d table(s)", schema, len(tables))
    return len(tables)


def load_all(tables: list[str] | None = None, clear: bool = False) -> dict[str, int]:
    """Load every table (or a named subset) and return {table: row_count}.

    Steps: validate the requested subset (fail fast on a typo'd table name), ensure the
    raw schema exists, optionally wipe it first, then load each selected table. Returns
    the per-table counts so a caller/Airflow task can assert the load looks sane.
    """
    data_dir = _data_dir()
    selected = [t for t in RAW_TABLES if tables is None or t.name in tables]
    if tables:
        # Guard against a caller asking for a table we don't know — better a clear error
        # than silently loading nothing.
        missing = set(tables) - {t.name for t in RAW_TABLES}
        if missing:
            raise SystemExit(f"unknown table(s): {sorted(missing)}")

    ensure_schema(RAW_SCHEMA)
    if clear:
        clear_schema(RAW_SCHEMA)

    counts: dict[str, int] = {}
    for spec in selected:
        counts[spec.name] = load_table(spec, data_dir)
    total = sum(counts.values())
    log.info("done: %d tables, %d total rows into %s.%s", len(counts), total,
             get_engine().url.database, RAW_SCHEMA)
    return counts


def main() -> None:
    """CLI entry point (``python -m pipeline.load_raw``). Exposes the table subset and
    the ``--clear`` wipe as command-line flags so the step is usable both by hand and
    from an Airflow BashOperator."""
    parser = argparse.ArgumentParser(description="Load raw Olist CSVs into Postgres.")
    parser.add_argument("--tables", nargs="*", help="subset of table names (default: all)")
    parser.add_argument("--clear", action="store_true",
                        help=f"drop all existing tables in the {RAW_SCHEMA!r} schema before loading")
    args = parser.parse_args()
    load_all(args.tables, clear=args.clear)


if __name__ == "__main__":
    main()
