"""
Phase 1 — Data Cleaning
Turns validated raw data into a clean, merged, analysis-ready dataframe.
Only the OUTPUT of this module is allowed to reach modeling or the dashboard —
raw data is never displayed or modeled directly.
"""

import pandas as pd
import numpy as np


def load_and_clean(sales_path: str, store_path: str) -> pd.DataFrame:
    """
    Loads train.csv + store.csv, merges them, and returns a single clean
    dataframe ready for EDA / feature engineering / modeling.
    """
    sales = pd.read_csv(sales_path, dtype={"StateHoliday": "str"}, low_memory=False)
    store = pd.read_csv(store_path)

    sales = _clean_sales(sales)
    store = _clean_store(store)

    df = sales.merge(store, on="Store", how="left")
    return df


def _clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop rows where the store was closed — zero sales on closed days is not
    # a demand signal, it's an operational flag. Including them would bias
    # the forecast toward "store closed" days instead of true demand.
    df = df[df["Open"] == 1].copy()

    # Drop rows with non-positive sales even while "open" — data entry noise
    df = df[df["Sales"] > 0].copy()

    # Normalize StateHoliday: '0' (string) and 0 (int) both mean "no holiday"
    df["StateHoliday"] = df["StateHoliday"].replace({0: "0", "0.0": "0"})

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def _clean_store(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Missing CompetitionDistance -> assume far away (no nearby competition)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].max()
    )
    # Missing competition-open-since fields -> fill with 0 (unknown / not applicable)
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "PromoInterval" in df.columns:
        df["PromoInterval"] = df["PromoInterval"].fillna("")
    return df


def data_quality_report(raw_sales: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Summarizes what cleaning removed/changed — for transparency in the dashboard."""
    return {
        "raw_rows": len(raw_sales),
        "clean_rows": len(clean_df),
        "rows_dropped": len(raw_sales) - len(clean_df),
        "closed_store_rows_dropped": int((raw_sales["Open"] == 0).sum()),
        "zero_or_negative_sales_dropped": int((raw_sales["Sales"] <= 0).sum()),
        "date_range": (str(clean_df["Date"].min().date()), str(clean_df["Date"].max().date())),
        "n_stores": int(clean_df["Store"].nunique()),
    }


if __name__ == "__main__":
    raw_sales = pd.read_csv("data/train.csv", dtype={"StateHoliday": "str"}, low_memory=False)
    clean = load_and_clean("data/train.csv", "data/store.csv")
    report = data_quality_report(raw_sales, clean)
    print(report)
    print(clean.head())
    clean.to_parquet("data/clean_data.parquet", index=False)
    print(f"\nSaved clean data: {clean.shape}")
