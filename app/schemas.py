"""Pydantic request/response models for the delivery-risk API.

``PredictionInput`` is the purchase-time feature contract with ``extra='forbid'``
as the temporal-leakage firewall.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """Purchase-time features (featureset_v1), aligned to the model's inputs.

    ``extra='forbid'`` rejects any field not listed here, so delivery-outcome /
    review / actual-delivery values can never enter the model. The service aligns
    this payload to the loaded model's own ``feature_names_in_``, so registering the
    trained model needs no change here as long as it uses these features.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "order_id": "abc123", "payment_count": 1, "payment_type_mode": "credit_card",
                "max_installments": 3.0, "order_item_count": 2, "product_count": 2, "seller_count": 1,
                "price_sum": 120.0, "price_mean": 60.0, "price_max": 70.0, "price_std": 14.1,
                "freight_sum": 20.0, "freight_mean": 10.0, "freight_max": 12.0, "freight_std": 2.8,
                "total_cost_sum": 140.0, "total_cost_mean": 70.0, "total_cost_max": 82.0,
                "product_category_count": 1, "product_category_mode": "cama_mesa_banho",
                "is_multi_category": False, "product_weight_g_mean": 800.0, "product_weight_g_max": 900.0,
                "product_length_cm_mean": 30.0, "product_length_cm_max": 35.0,
                "product_height_cm_mean": 10.0, "product_height_cm_max": 12.0,
                "product_width_cm_mean": 20.0, "product_width_cm_max": 22.0,
                "product_volume_mean": 6000.0, "product_volume_max": 9240.0,
                "seller_state_mode": "SP", "seller_state_count": 1, "seller_city_count": 1,
                "seller_zip_mode": 14000, "purchase_hour": 9, "purchase_dayofweek": 1,
                "is_weekend_purchase": 0, "purchase_month": 5, "purchase_quarter": 2, "is_month_end": 0,
                "estimated_delivery_days": 8, "approval_delay_hours": 4.5,
                "shipping_limit_min_days": 3, "shipping_window_days": 5, "seller_margin_days": 2,
            }
        },
    )

    order_id: str  # identifier only; dropped before scoring, never a feature

    # payment
    payment_count: int = Field(ge=0)
    payment_type_mode: str
    max_installments: float = Field(ge=0)

    # order / items
    order_item_count: int = Field(ge=0)
    product_count: int = Field(ge=0)
    seller_count: int = Field(ge=0)

    # price / freight / cost aggregates
    price_sum: float = Field(ge=0)
    price_mean: float = Field(ge=0)
    price_max: float = Field(ge=0)
    price_std: Optional[float] = None
    freight_sum: float = Field(ge=0)
    freight_mean: float = Field(ge=0)
    freight_max: float = Field(ge=0)
    freight_std: Optional[float] = None
    total_cost_sum: float = Field(ge=0)
    total_cost_mean: float = Field(ge=0)
    total_cost_max: float = Field(ge=0)

    # product category / dimensions
    product_category_count: int = Field(ge=0)
    product_category_mode: str
    is_multi_category: bool
    product_weight_g_mean: Optional[float] = None
    product_weight_g_max: Optional[float] = None
    product_length_cm_mean: Optional[float] = None
    product_length_cm_max: Optional[float] = None
    product_height_cm_mean: Optional[float] = None
    product_height_cm_max: Optional[float] = None
    product_width_cm_mean: Optional[float] = None
    product_width_cm_max: Optional[float] = None
    product_volume_mean: Optional[float] = None
    product_volume_max: Optional[float] = None

    # seller geography
    seller_state_mode: str
    seller_state_count: int = Field(ge=0)
    seller_city_count: int = Field(ge=0)
    seller_zip_mode: int

    # purchase timing (all known at purchase)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_dayofweek: int = Field(ge=0, le=6)
    is_weekend_purchase: int = Field(ge=0, le=1)
    purchase_month: int = Field(ge=1, le=12)
    purchase_quarter: int = Field(ge=1, le=4)
    is_month_end: int = Field(ge=0, le=1)

    # estimated / shipping windows (set at purchase, not actuals)
    estimated_delivery_days: int = Field(gt=0, le=365)
    approval_delay_hours: float = Field(ge=0)
    shipping_limit_min_days: int
    shipping_window_days: int
    seller_margin_days: int


class PredictionResponse(BaseModel):
    order_id: str
    late_delivery_probability: float
    risk_level: str
    recommended_action: str
    model_version: str
    latency: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    source: str
    name: str
    version: Optional[str] = None
    stage: Optional[str] = None
    n_features: Optional[int] = None
    is_real_model: bool
    load_error: Optional[str] = None
    temporal_leakage_policy: str = "purchase-time features only"
