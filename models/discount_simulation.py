"""
Phase 6 — Discount & Price Simulation
Quantifies historical promo uplift per store/category and exposes a simple
what-if function: given a hypothetical discount %, estimate the demand
impact based on that store's own historical promo response.

This is intentionally a transparent, explainable elasticity model — not a
black box — so it can be defended in an interview: the number comes directly
from comparing real historical promo vs non-promo sales, not a hidden model.
"""

import pandas as pd
import numpy as np


def compute_promo_uplift(df: pd.DataFrame, store_id: int = None) -> dict:
    """
    Computes promo uplift % for a single store (or overall if store_id=None).
    Uplift = (avg sales during promo - avg sales without promo) / avg sales without promo
    """
    data = df if store_id is None else df[df["Store"] == store_id]

    promo_avg = data[data["Promo"] == 1]["Sales"].mean()
    no_promo_avg = data[data["Promo"] == 0]["Sales"].mean()

    if pd.isna(promo_avg) or pd.isna(no_promo_avg) or no_promo_avg == 0:
        return {"store": store_id, "uplift_pct": None, "avg_sales_promo": None,
                "avg_sales_no_promo": None, "note": "Insufficient promo history"}

    uplift_pct = (promo_avg - no_promo_avg) / no_promo_avg * 100

    return {
        "store": store_id,
        "uplift_pct": round(float(uplift_pct), 2),
        "avg_sales_promo": round(float(promo_avg), 1),
        "avg_sales_no_promo": round(float(no_promo_avg), 1),
    }


def simulate_discount_scenario(
    baseline_sales: float,
    historical_uplift_pct: float,
    discount_pct: float,
) -> dict:
    """
    What-if simulation: projects demand impact for a hypothetical discount,
    scaled from the store's OWN historical promo response.

    Assumption (explicit, stated — not hidden): uplift scales roughly
    linearly with discount depth relative to Rossmann's typical historical
    promo depth (~20%, a reasonable retail markdown benchmark). This is a
    simplifying assumption appropriate for a what-if planning tool, not a
    causal-inference-grade elasticity estimate — worth stating plainly
    rather than dressing up as more rigorous than it is.
    """
    REFERENCE_DISCOUNT_PCT = 20.0  # assumed typical depth of historical promos

    scaling_factor = discount_pct / REFERENCE_DISCOUNT_PCT
    projected_uplift_pct = historical_uplift_pct * scaling_factor
    projected_sales = baseline_sales * (1 + projected_uplift_pct / 100)

    # Margin trade-off: more discount = more volume, but lower margin per unit
    projected_revenue_change_pct = (
        (1 + projected_uplift_pct / 100) * (1 - discount_pct / 100) - 1
    ) * 100

    return {
        "discount_pct": discount_pct,
        "baseline_sales": round(baseline_sales, 1),
        "projected_uplift_pct": round(projected_uplift_pct, 2),
        "projected_sales": round(projected_sales, 1),
        "projected_revenue_change_pct": round(projected_revenue_change_pct, 2),
    }


if __name__ == "__main__":
    df = pd.read_parquet("data/clean_data.parquet")

    overall = compute_promo_uplift(df)
    print("Overall promo uplift:", overall)

    for store_id in [1, 2, 3, 85]:
        result = compute_promo_uplift(df, store_id)
        print(f"Store {store_id}:", result)

        if result["uplift_pct"] is not None:
            for discount in [10, 20, 30]:
                sim = simulate_discount_scenario(
                    result["avg_sales_no_promo"], result["uplift_pct"], discount
                )
                print(f"  what-if {discount}% discount ->", sim)
