"""
Phase 3 — Feature Engineering
Adds lag, rolling-window, and calendar features on top of the cleaned data.
Used by the XGBoost comparison model; Prophet/SARIMA consume the raw
Date/Sales series directly (they don't need hand-built lag features).
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).copy()

    g = df.groupby("Store")["Sales"]

    # Lag features
    df["sales_lag_1"] = g.shift(1)
    df["sales_lag_7"] = g.shift(7)
    df["sales_lag_14"] = g.shift(14)

    # Rolling averages (shifted by 1 to avoid leaking the current day)
    df["sales_roll_7"] = g.shift(1).rolling(7).mean()
    df["sales_roll_30"] = g.shift(1).rolling(30).mean()

    # Calendar features
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # Promo features
    df["promo_active"] = df["Promo"]

    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/clean_data.parquet")
    featured = add_features(df)
    print(featured[["Store", "Date", "Sales", "sales_lag_1", "sales_lag_7",
                     "sales_roll_7", "month", "is_weekend"]].dropna().head())
    featured.to_parquet("data/featured_data.parquet", index=False)
    print(f"\nSaved featured data: {featured.shape}")
