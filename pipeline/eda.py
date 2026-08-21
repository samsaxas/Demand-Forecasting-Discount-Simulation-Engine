"""
Phase 2 — Exploratory Data Analysis
Generates the summary stats that inform feature engineering and model choice.
"""

import pandas as pd
import numpy as np


def run_eda(df: pd.DataFrame) -> dict:
    results = {}

    # Overall sales distribution
    results["sales_summary"] = df["Sales"].describe().to_dict()

    # Sales by store type
    results["sales_by_store_type"] = df.groupby("StoreType")["Sales"].mean().round(1).to_dict()

    # Promo vs non-promo uplift (this feeds directly into Phase 6 discount simulation)
    promo_avg = df[df["Promo"] == 1]["Sales"].mean()
    no_promo_avg = df[df["Promo"] == 0]["Sales"].mean()
    results["promo_uplift_pct"] = round((promo_avg - no_promo_avg) / no_promo_avg * 100, 2)
    results["avg_sales_promo"] = round(promo_avg, 1)
    results["avg_sales_no_promo"] = round(no_promo_avg, 1)

    # Day-of-week seasonality
    results["sales_by_dow"] = df.groupby("DayOfWeek")["Sales"].mean().round(1).to_dict()

    # State holiday effect
    results["sales_by_state_holiday"] = df.groupby("StateHoliday")["Sales"].mean().round(1).to_dict()

    # Correlation: Sales vs Customers, CompetitionDistance
    results["corr_sales_customers"] = round(df["Sales"].corr(df["Customers"]), 3)
    results["corr_sales_competition_distance"] = round(
        df["Sales"].corr(df["CompetitionDistance"]), 3
    )

    # Monthly seasonality
    monthly = df.set_index("Date").resample("ME")["Sales"].mean()
    results["monthly_trend"] = {str(k.date()): round(v, 1) for k, v in monthly.items()}

    return results


if __name__ == "__main__":
    df = pd.read_parquet("data/clean_data.parquet")
    results = run_eda(df)
    for k, v in results.items():
        print(f"\n--- {k} ---")
        print(v)
