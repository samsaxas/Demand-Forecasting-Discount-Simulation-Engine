"""
Phase 4 — Forecasting Models
Phase 5 — Model Validation

Runs Prophet and SARIMA across MULTIPLE stores/categories through the same
pipeline logic (this is what makes it "scalable" in the JD sense — one
pipeline applied across many series, not one hand-built model per store).

Validated with walk-forward validation (train on early period, test on a
held-out final window) using RMSE and MAPE — not a random train/test split,
which would leak future information into a time-series model.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

HOLDOUT_DAYS = 42  # 6 weeks — matches Rossmann's original forecast horizon


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def get_store_series(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    s = df[df["Store"] == store_id][["Date", "Sales"]].sort_values("Date")
    return s.rename(columns={"Date": "ds", "Sales": "y"})


def walk_forward_split(series: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    cutoff = series["ds"].max() - pd.Timedelta(days=holdout_days)
    train = series[series["ds"] <= cutoff]
    test = series[series["ds"] > cutoff]
    return train, test


def fit_prophet(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.fit(train)

    future = test[["ds"]].copy()
    forecast = model.predict(future)

    preds = forecast["yhat"].values
    actual = test["y"].values

    return {
        "model": "Prophet",
        "rmse": rmse(actual, preds),
        "mape": mape(actual, preds),
        "predictions": preds.tolist(),
        "actuals": actual.tolist(),
        "dates": test["ds"].dt.strftime("%Y-%m-%d").tolist(),
    }


def fit_sarima(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    # Weekly seasonal order (7-day retail cycle), kept low-order for speed
    # across many stores — this is a deliberate scalability trade-off.
    y_train = train.set_index("ds")["y"].asfreq("D").interpolate()

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    preds = fit.forecast(steps=len(test))
    actual = test["y"].values
    preds = preds.values[: len(actual)]

    return {
        "model": "SARIMA",
        "rmse": rmse(actual, preds),
        "mape": mape(actual, preds),
        "predictions": preds.tolist(),
        "actuals": actual.tolist(),
        "dates": test["ds"].dt.strftime("%Y-%m-%d").tolist(),
    }


def run_forecasting_pipeline(df: pd.DataFrame, store_ids: list) -> dict:
    """
    Runs Prophet + SARIMA for every store_id passed in, using the SAME
    pipeline logic each time. This loop is what demonstrates "scalable"
    forecasting across categories/stores rather than one bespoke model.
    """
    results = {}
    for store_id in store_ids:
        series = get_store_series(df, store_id)
        if len(series) < 100:  # not enough history to validate meaningfully
            continue

        train, test = walk_forward_split(series)
        if len(test) == 0 or len(train) < 60:
            continue

        store_result = {}
        try:
            store_result["prophet"] = fit_prophet(train, test)
        except Exception as e:
            store_result["prophet"] = {"error": str(e)}

        try:
            store_result["sarima"] = fit_sarima(train, test)
        except Exception as e:
            store_result["sarima"] = {"error": str(e)}

        # Pick best model by RMSE
        candidates = {k: v for k, v in store_result.items() if "rmse" in v}
        if candidates:
            best = min(candidates, key=lambda k: candidates[k]["rmse"])
            store_result["best_model"] = best

        results[str(store_id)] = store_result
        print(f"Store {store_id}: "
              f"Prophet RMSE={store_result.get('prophet', {}).get('rmse', 'NA'):.1f} "
              f"MAPE={store_result.get('prophet', {}).get('mape', 'NA'):.1f}% | "
              f"SARIMA RMSE={store_result.get('sarima', {}).get('rmse', 'NA'):.1f} "
              f"MAPE={store_result.get('sarima', {}).get('mape', 'NA'):.1f}% | "
              f"best={store_result.get('best_model')}")

    return results


if __name__ == "__main__":
    df = pd.read_parquet("data/clean_data.parquet")
    store_meta = pd.read_csv("data/store.csv")

    # Select a representative sample across StoreType (a, b, c, d) —
    # demonstrates the pipeline works across categories, not just one store.
    sample_stores = []
    for store_type in ["a", "b", "c", "d"]:
        candidates = store_meta[store_meta["StoreType"] == store_type]["Store"].tolist()
        sample_stores.extend(candidates[:4])  # 4 stores per type = 16 total

    print(f"Running forecasting pipeline across {len(sample_stores)} stores: {sample_stores}\n")
    results = run_forecasting_pipeline(df, sample_stores)

    with open("outputs/forecast_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results for {len(results)} stores to outputs/forecast_results.json")
