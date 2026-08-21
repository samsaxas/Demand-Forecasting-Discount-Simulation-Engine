"""
Phase 1 — Schema Validation
Validates incoming sales and store data against an expected schema BEFORE
any cleaning or modeling touches it. Fails loudly with a clear error rather
than silently letting bad data flow into forecasts.

Designed to be config-driven: column names/types are declared once, so this
same validator works if you swap in a different (but similarly-shaped)
retail dataset later without rewriting logic.
"""

import pandas as pd


class SchemaValidationError(Exception):
    """Raised when incoming data does not match the expected schema."""
    pass


# --- Config: expected schema for the two input tables -----------------
SALES_SCHEMA = {
    "Store": "int64",
    "DayOfWeek": "int64",
    "Date": "object",       # parsed to datetime during cleaning
    "Sales": "int64",
    "Customers": "int64",
    "Open": "int64",
    "Promo": "int64",
    "StateHoliday": "object",
    "SchoolHoliday": "int64",
}

STORE_SCHEMA = {
    "Store": "int64",
    "StoreType": "object",
    "Assortment": "object",
    "CompetitionDistance": "float64",
}


def _check_columns(df: pd.DataFrame, schema: dict, table_name: str):
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"[{table_name}] Missing required columns: {missing}. "
            f"Expected at least: {list(schema.keys())}"
        )


def _check_duplicates(df: pd.DataFrame, key_cols: list, table_name: str):
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes > 0:
        raise SchemaValidationError(
            f"[{table_name}] Found {dupes} duplicate rows on key {key_cols}. "
            f"Each {key_cols} combination must be unique."
        )


def validate_sales_schema(df: pd.DataFrame) -> dict:
    """
    Validates the daily sales table (train.csv-shaped data).
    Returns a report dict; raises SchemaValidationError on hard failures.
    """
    report = {"table": "sales", "rows_in": len(df), "issues": []}

    _check_columns(df, SALES_SCHEMA, "sales")
    _check_duplicates(df, ["Store", "Date"], "sales")

    # Soft checks — logged, not fatal
    if df["Sales"].lt(0).any():
        report["issues"].append("Negative Sales values found")
    if df["Date"].isna().any():
        report["issues"].append("Null Date values found")
    if df["Store"].isna().any():
        raise SchemaValidationError("[sales] Null Store IDs are not allowed")

    try:
        pd.to_datetime(df["Date"])
    except Exception as e:
        raise SchemaValidationError(f"[sales] Date column is not parseable: {e}")

    report["rows_valid"] = len(df)
    return report


def validate_store_schema(df: pd.DataFrame) -> dict:
    """
    Validates the store metadata table (store.csv-shaped data).
    """
    report = {"table": "store", "rows_in": len(df), "issues": []}

    _check_columns(df, STORE_SCHEMA, "store")
    _check_duplicates(df, ["Store"], "store")

    if df["Store"].isna().any():
        raise SchemaValidationError("[store] Null Store IDs are not allowed")

    report["rows_valid"] = len(df)
    return report


if __name__ == "__main__":
    sales = pd.read_csv("data/train.csv")
    store = pd.read_csv("data/store.csv")

    sales_report = validate_sales_schema(sales)
    store_report = validate_store_schema(store)

    print("Sales validation:", sales_report)
    print("Store validation:", store_report)
