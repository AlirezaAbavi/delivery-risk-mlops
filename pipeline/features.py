"""Build the leak-free purchase-time feature table (featureset_v1).

Temporal-leakage firewall (graded): every feature is knowable at/around
``order_purchase_timestamp``. The estimated-delivery date and per-item
``shipping_limit_date`` are commitments *set at purchase*, so windows derived from
them are legal. The label uses the actual delivered date, but that is the target
``is_late_delivery`` — never a feature. ``order_delivered_*`` and ``review_*`` never
enter X.

Output columns are asserted equal to ``app.config.FEATURE_COLUMNS`` (the API/model
contract is authoritative), plus ``order_id``, ``is_late_delivery`` (target) and
``purchase_ts`` (temporal-split key, not a feature).

Usage:
    python -m pipeline.features
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.config import FEATURE_COLUMNS
from .db import FEATURES_SCHEMA, RAW_SCHEMA, ensure_schema, get_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.features")

FEATURESET_TABLE = os.getenv("FEATURESET_TABLE", "featureset_v1")

# One row per delivered order. CTEs aggregate items (with product+seller joins)
# and payments to order grain; the outer query derives purchase-time windows and
# calendar features. Postgres mode()/stddev_samp give the categorical mode and
# per-order dispersion the contract expects. Base tables are schema-qualified with
# RAW_SCHEMA so the query reads the raw layer regardless of search_path.
FEATURE_SQL = f"""
WITH item_products AS (
    SELECT oi.order_id, oi.product_id, oi.seller_id,
           oi.price, oi.freight_value, (oi.price + oi.freight_value) AS total_cost,
           oi.shipping_limit_date,
           p.product_category_name,
           p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
           (p.product_length_cm::numeric * p.product_height_cm * p.product_width_cm) AS product_volume,
           s.seller_state, s.seller_city, s.seller_zip_code_prefix
    FROM {RAW_SCHEMA}.order_items oi
    LEFT JOIN {RAW_SCHEMA}.products p ON p.product_id = oi.product_id
    LEFT JOIN {RAW_SCHEMA}.sellers  s ON s.seller_id  = oi.seller_id
),
item_agg AS (
    SELECT order_id,
        count(*)                                   AS order_item_count,
        count(DISTINCT product_id)                 AS product_count,
        count(DISTINCT seller_id)                  AS seller_count,
        sum(price)   AS price_sum,   avg(price)   AS price_mean,   max(price)   AS price_max,   stddev_samp(price)         AS price_std,
        sum(freight_value) AS freight_sum, avg(freight_value) AS freight_mean, max(freight_value) AS freight_max, stddev_samp(freight_value) AS freight_std,
        sum(total_cost) AS total_cost_sum, avg(total_cost) AS total_cost_mean, max(total_cost) AS total_cost_max,
        count(DISTINCT product_category_name)      AS product_category_count,
        mode() WITHIN GROUP (ORDER BY product_category_name) AS product_category_mode,
        avg(product_weight_g) AS product_weight_g_mean, max(product_weight_g) AS product_weight_g_max,
        avg(product_length_cm) AS product_length_cm_mean, max(product_length_cm) AS product_length_cm_max,
        avg(product_height_cm) AS product_height_cm_mean, max(product_height_cm) AS product_height_cm_max,
        avg(product_width_cm)  AS product_width_cm_mean,  max(product_width_cm)  AS product_width_cm_max,
        avg(product_volume)    AS product_volume_mean,    max(product_volume)    AS product_volume_max,
        mode() WITHIN GROUP (ORDER BY seller_state)          AS seller_state_mode,
        count(DISTINCT seller_state)               AS seller_state_count,
        count(DISTINCT seller_city)                AS seller_city_count,
        mode() WITHIN GROUP (ORDER BY seller_zip_code_prefix) AS seller_zip_mode,
        min(shipping_limit_date)                   AS shipping_limit_min,
        max(shipping_limit_date)                   AS shipping_limit_max
    FROM item_products
    GROUP BY order_id
),
pay_agg AS (
    SELECT order_id,
        count(*)                                            AS payment_count,
        mode() WITHIN GROUP (ORDER BY payment_type)         AS payment_type_mode,
        max(payment_installments)                           AS max_installments
    FROM {RAW_SCHEMA}.order_payments
    GROUP BY order_id
)
SELECT
    o.order_id,
    -- payment
    COALESCE(pa.payment_count, 0)                           AS payment_count,
    COALESCE(pa.payment_type_mode, 'unknown')               AS payment_type_mode,
    COALESCE(pa.max_installments, 0)::double precision      AS max_installments,
    -- items / order
    ia.order_item_count, ia.product_count, ia.seller_count,
    ia.price_sum, ia.price_mean, ia.price_max, ia.price_std,
    ia.freight_sum, ia.freight_mean, ia.freight_max, ia.freight_std,
    ia.total_cost_sum, ia.total_cost_mean, ia.total_cost_max,
    ia.product_category_count,
    COALESCE(ia.product_category_mode, 'unknown')           AS product_category_mode,
    (ia.product_category_count > 1)                         AS is_multi_category,
    ia.product_weight_g_mean, ia.product_weight_g_max,
    ia.product_length_cm_mean, ia.product_length_cm_max,
    ia.product_height_cm_mean, ia.product_height_cm_max,
    ia.product_width_cm_mean,  ia.product_width_cm_max,
    ia.product_volume_mean,    ia.product_volume_max,
    -- seller geography
    COALESCE(ia.seller_state_mode, 'unknown')               AS seller_state_mode,
    ia.seller_state_count, ia.seller_city_count,
    COALESCE(ia.seller_zip_mode, 0)                         AS seller_zip_mode,
    -- purchase-time calendar (all from the purchase timestamp)
    EXTRACT(HOUR  FROM o.order_purchase_timestamp)::int     AS purchase_hour,
    EXTRACT(DOW   FROM o.order_purchase_timestamp)::int     AS purchase_dayofweek,
    (EXTRACT(DOW FROM o.order_purchase_timestamp) IN (0, 6))::int AS is_weekend_purchase,
    EXTRACT(MONTH   FROM o.order_purchase_timestamp)::int   AS purchase_month,
    EXTRACT(QUARTER FROM o.order_purchase_timestamp)::int   AS purchase_quarter,
    (EXTRACT(DAY FROM (date_trunc('month', o.order_purchase_timestamp)
                       + interval '1 month' - interval '1 day'))
     - EXTRACT(DAY FROM o.order_purchase_timestamp) <= 2)::int AS is_month_end,
    -- purchase-time delivery/shipping windows (commitments, not actuals)
    GREATEST(CEIL(EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - o.order_purchase_timestamp)) / 86400.0), 1)::int AS estimated_delivery_days,
    GREATEST(EXTRACT(EPOCH FROM (o.order_approved_at - o.order_purchase_timestamp)) / 3600.0, 0)::double precision       AS approval_delay_hours,
    CEIL(EXTRACT(EPOCH FROM (ia.shipping_limit_min - o.order_purchase_timestamp)) / 86400.0)::int   AS shipping_limit_min_days,
    CEIL(EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - ia.shipping_limit_min)) / 86400.0)::int AS shipping_window_days,
    CEIL(EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - ia.shipping_limit_max)) / 86400.0)::int AS seller_margin_days,
    -- target (NOT a feature) + temporal-split key (NOT a feature)
    (o.order_delivered_customer_date > o.order_estimated_delivery_date)::int AS is_late_delivery,
    o.order_purchase_timestamp                              AS purchase_ts
FROM {RAW_SCHEMA}.orders o
JOIN item_agg ia       ON ia.order_id = o.order_id
LEFT JOIN pay_agg pa   ON pa.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
  AND o.order_purchase_timestamp     IS NOT NULL
  AND o.order_approved_at            IS NOT NULL
"""

# Non-feature columns the query also emits.
_EXTRA_COLUMNS = ["order_id", "is_late_delivery", "purchase_ts"]


def _artifact_path() -> Path:
    return Path(os.getenv("FEATURESET_PATH", "artifacts/featureset_v1.csv"))


def build(write_csv: bool = True) -> pd.DataFrame:
    engine = get_engine()
    ensure_schema(FEATURES_SCHEMA)
    qualified = f'"{FEATURES_SCHEMA}"."{FEATURESET_TABLE}"'

    # Materialize as a table so Airflow / batch-predict can query it directly.
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {qualified}"))
        conn.execute(text(f"CREATE TABLE {qualified} AS {FEATURE_SQL}"))
        n = conn.execute(text(f"SELECT count(*) FROM {qualified}")).scalar_one()
    log.info("materialized table %s: %d rows", qualified, n)

    frame = pd.read_sql_table(FEATURESET_TABLE, engine, schema=FEATURES_SCHEMA)

    # The contract is authoritative: fail loudly if the produced feature set drifts.
    produced = set(frame.columns) - set(_EXTRA_COLUMNS)
    expected = set(FEATURE_COLUMNS)
    if produced != expected:
        raise AssertionError(
            "featureset columns != config.FEATURE_COLUMNS\n"
            f"  missing (in contract, not produced): {sorted(expected - produced)}\n"
            f"  extra   (produced, not in contract): {sorted(produced - expected)}"
        )
    # Reorder to contract order + extras for a stable, readable artifact.
    frame = frame[[*FEATURE_COLUMNS, *_EXTRA_COLUMNS]]

    late_rate = frame["is_late_delivery"].mean()
    log.info("label balance: is_late_delivery mean=%.4f (%d late / %d total)",
             late_rate, int(frame["is_late_delivery"].sum()), len(frame))

    if write_csv:
        path = _artifact_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        log.info("wrote %s (%d rows, %d cols)", path, len(frame), frame.shape[1])

    return frame


def main() -> None:
    build()


if __name__ == "__main__":
    main()
