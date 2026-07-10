"""Generate a small, synthetic Olist-shaped dataset for the zero-config demo.

The real Olist dataset needs a Kaggle download (see scripts/fetch_data.sh). To let the
whole stack run offline with no credentials, this script writes schema-accurate sample
CSVs into sample_data/ — the same 8 files pipeline/load_raw.py expects, with the same
columns and referential integrity (order_items reference real products/sellers, payments
and reviews reference real orders).

The data is synthetic but not random noise: late delivery is driven by a latent signal
(freight cost, seller-state distance, tight shipping window, short promised delivery),
so the three candidate models in pipeline/train.py learn something real and PR-AUC beats
a coin flip. Deterministic (seeded) so the committed CSVs are reproducible.

Usage:
    python scripts/make_sample_data.py            # writes sample_data/*.csv
"""
from __future__ import annotations

import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 1400
N_SELLERS = 60
N_PRODUCTS = 160
OUT_DIR = Path(os.getenv("SAMPLE_DATA_DIR", "sample_data"))

random.seed(SEED)
np.random.seed(SEED)

# States with a rough "distance / logistics difficulty" weight: higher => more late risk.
# São Paulo (SP) is the hub (easy); the North/Northeast are harder to reach.
STATE_DIFFICULTY = {
    "SP": 0.0, "RJ": 0.3, "MG": 0.3, "PR": 0.4, "SC": 0.4, "RS": 0.6,
    "ES": 0.5, "GO": 0.6, "DF": 0.6, "BA": 0.9, "PE": 1.0, "CE": 1.1,
}
STATES = list(STATE_DIFFICULTY)

CATEGORIES = [
    "cama_mesa_banho", "beleza_saude", "esporte_lazer", "moveis_decoracao",
    "informatica_acessorios", "utilidades_domesticas", "relogios_presentes",
    "telefonia", "automotivo", "brinquedos", "cool_stuff", "ferramentas_jardim",
    "perfumaria", "bebes", "eletronicos", "papelaria", "fashion_bolsas_e_acessorios",
    "pet_shop", "moveis_escritorio", "consoles_games",
]
CATEGORY_EN = {
    "cama_mesa_banho": "bed_bath_table", "beleza_saude": "health_beauty",
    "esporte_lazer": "sports_leisure", "moveis_decoracao": "furniture_decor",
    "informatica_acessorios": "computers_accessories", "utilidades_domesticas": "housewares",
    "relogios_presentes": "watches_gifts", "telefonia": "telephony", "automotivo": "auto",
    "brinquedos": "toys", "cool_stuff": "cool_stuff", "ferramentas_jardim": "garden_tools",
    "perfumaria": "perfumery", "bebes": "baby", "eletronicos": "electronics",
    "papelaria": "stationery", "fashion_bolsas_e_acessorios": "fashion_bags_accessories",
    "pet_shop": "pet_shop", "moveis_escritorio": "office_furniture", "consoles_games": "consoles_games",
}
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]

START = datetime(2017, 1, 1)
END = datetime(2018, 8, 1)


def _hex_id(prefix: str, i: int) -> str:
    """Stable 32-char-ish id, echoing Olist's hex ids but readable/deterministic."""
    return f"{prefix}{i:028d}"


def _rand_dt(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.random() * delta)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- sellers -------------------------------------------------------------
    sellers = []
    for i in range(N_SELLERS):
        state = random.choice(STATES)
        sellers.append({
            "seller_id": _hex_id("s", i),
            "seller_zip_code_prefix": random.randint(1000, 99000),
            "seller_city": f"city_{state.lower()}_{i % 12}",
            "seller_state": state,
        })
    sellers_df = pd.DataFrame(sellers)

    # --- products ------------------------------------------------------------
    products = []
    for i in range(N_PRODUCTS):
        cat = random.choice(CATEGORIES)
        # Physical dimensions are floats (and occasionally missing), mirroring the real
        # Olist data. Keeping them float — not int — matters: the model signature inferred
        # at training time must be `double`, or the API's float inputs fail MLflow schema
        # enforcement and every real request silently falls back to the baseline.
        def _dim(lo, hi):
            return None if random.random() < 0.02 else round(random.uniform(lo, hi), 1)
        products.append({
            "product_id": _hex_id("p", i),
            "product_category_name": cat,
            "product_name_lenght": random.randint(20, 60),
            "product_description_lenght": random.randint(100, 3000),
            "product_photos_qty": random.randint(1, 6),
            "product_weight_g": _dim(150, 9000),
            "product_length_cm": _dim(15, 60),
            "product_height_cm": _dim(5, 40),
            "product_width_cm": _dim(10, 45),
        })
    products_df = pd.DataFrame(products)

    # --- orders + children ---------------------------------------------------
    orders, items, payments, reviews, customers = [], [], [], [], []
    for i in range(N_ORDERS):
        order_id = _hex_id("o", i)
        customer_id = _hex_id("c", i)
        customers.append({
            "customer_id": customer_id,
            "customer_unique_id": _hex_id("u", i),
            "customer_zip_code_prefix": random.randint(1000, 99000),
            "customer_city": f"customer_city_{i % 50}",
            "customer_state": random.choice(STATES),
        })

        purchase = _rand_dt(START, END)
        approved = purchase + timedelta(hours=random.uniform(0.2, 20))
        seller = random.choice(sellers)
        difficulty = STATE_DIFFICULTY[seller["seller_state"]]

        n_items = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
        # shipping_limit: seller must hand to carrier within a few days of purchase.
        ship_days = random.uniform(2, 8)
        shipping_limit = purchase + timedelta(days=ship_days)
        # promised delivery window from purchase (days)
        est_days = random.randint(8, 40)
        estimated = purchase + timedelta(days=est_days)

        freight_base = 8 + difficulty * 22 + random.uniform(0, 12)
        order_freight = []
        for j in range(n_items):
            product = random.choice(products)
            price = round(random.uniform(12, 400), 2)
            freight = round(freight_base + random.uniform(-3, 3), 2)
            order_freight.append(freight)
            items.append({
                "order_id": order_id,
                "order_item_id": j + 1,
                "product_id": product["product_id"],
                "seller_id": seller["seller_id"],
                "shipping_limit_date": shipping_limit.strftime("%Y-%m-%d %H:%M:%S"),
                "price": price,
                "freight_value": freight,
            })

        # --- latent lateness signal --------------------------------------
        shipping_window = est_days - ship_days          # slack between handoff and promise
        mean_freight = float(np.mean(order_freight))
        logit = (
            -2.4                                        # base -> ~15% late
            + 1.3 * difficulty                          # far sellers run late
            + 0.045 * (mean_freight - 15)               # costly freight correlates with distance/late
            - 0.10 * (shipping_window - 12)             # tight window -> late
            - 0.05 * (est_days - 20)                    # short promise -> harder to keep
            + (0.25 if purchase.weekday() >= 5 else 0)  # weekend orders slightly worse
            + random.gauss(0, 0.5)                       # irreducible noise
        )
        is_late = random.random() < _sigmoid(logit)

        carrier = purchase + timedelta(days=random.uniform(1, min(ship_days + 2, est_days - 1)))
        if is_late:
            delivered = estimated + timedelta(days=random.uniform(1, 15))
        else:
            # on time: delivered before the promise but after carrier handoff
            earliest = carrier + timedelta(days=1)
            latest = estimated - timedelta(days=1)
            delivered = _rand_dt(earliest, latest) if latest > earliest else earliest

        # A few non-delivered orders exercise the `order_status = 'delivered'` filter.
        status = "delivered"
        if random.random() < 0.04:
            status = random.choice(["shipped", "canceled", "invoiced"])
            delivered_val = ""
        else:
            delivered_val = delivered.strftime("%Y-%m-%d %H:%M:%S")

        orders.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "order_status": status,
            "order_purchase_timestamp": purchase.strftime("%Y-%m-%d %H:%M:%S"),
            "order_approved_at": approved.strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_carrier_date": carrier.strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_customer_date": delivered_val,
            "order_estimated_delivery_date": estimated.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # payments (1-2 rows/order)
        total = sum(order_freight) + sum(random.uniform(12, 400) for _ in range(n_items))
        n_pay = random.choices([1, 2], weights=[0.85, 0.15])[0]
        ptype = random.choices(PAYMENT_TYPES, weights=[0.7, 0.2, 0.05, 0.05])[0]
        for k in range(n_pay):
            payments.append({
                "order_id": order_id,
                "payment_sequential": k + 1,
                "payment_type": ptype,
                "payment_installments": random.randint(1, 10) if ptype == "credit_card" else 1,
                "payment_value": round(total / n_pay, 2),
            })

        # review (one per order); late orders skew toward lower scores
        score = random.choices([1, 2, 3, 4, 5],
                               weights=[0.3, 0.2, 0.2, 0.15, 0.15] if is_late
                               else [0.05, 0.05, 0.1, 0.3, 0.5])[0]
        rev_created = delivered if delivered_val else estimated
        reviews.append({
            "review_id": _hex_id("r", i),
            "order_id": order_id,
            "review_score": score,
            "review_comment_title": "",
            "review_comment_message": "",
            "review_creation_date": rev_created.strftime("%Y-%m-%d %H:%M:%S"),
            "review_answer_timestamp": (rev_created + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        })

    translation = pd.DataFrame(
        [{"product_category_name": c, "product_category_name_english": CATEGORY_EN[c]} for c in CATEGORIES]
    )

    # --- write ---------------------------------------------------------------
    frames = {
        "olist_orders_dataset.csv": pd.DataFrame(orders),
        "olist_order_items_dataset.csv": pd.DataFrame(items),
        "olist_customers_dataset.csv": pd.DataFrame(customers),
        "olist_sellers_dataset.csv": sellers_df,
        "olist_products_dataset.csv": products_df,
        "olist_order_payments_dataset.csv": pd.DataFrame(payments),
        "olist_order_reviews_dataset.csv": pd.DataFrame(reviews),
        "product_category_name_translation.csv": translation,
    }
    for name, df in frames.items():
        df.to_csv(OUT_DIR / name, index=False)

    delivered = pd.DataFrame(orders)
    delivered = delivered[delivered["order_delivered_customer_date"] != ""]
    late = (
        pd.to_datetime(delivered["order_delivered_customer_date"])
        > pd.to_datetime(delivered["order_estimated_delivery_date"])
    ).mean()
    print(f"wrote {len(frames)} CSVs to {OUT_DIR}/")
    print(f"  orders={len(orders)} items={len(items)} payments={len(payments)} reviews={len(reviews)}")
    print(f"  delivered late-rate ~= {late:.3f}")


if __name__ == "__main__":
    build()
