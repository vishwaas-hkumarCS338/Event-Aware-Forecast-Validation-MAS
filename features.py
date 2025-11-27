# features.py
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

PARQUET = "data/sales.parquet"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_time_key(df: pd.DataFrame) -> pd.Series:
    # Global integer month index (e.g., 2019-12 -> 2019*12+12)
    return (df["year"].astype(int) * 12 + df["month"].astype(int)).astype(int)


def _safe_sum_units(df: pd.DataFrame) -> pd.Series:
    # Units = retail_sales + retail_transfers + warehouse_sales (all coerced to float)
    for col in ("retail_sales", "retail_transfers", "warehouse_sales"):
        if col not in df.columns:
            df[col] = 0.0
    return (
        df["retail_sales"].astype(float)
        + df["retail_transfers"].astype(float)
        + df["warehouse_sales"].astype(float)
    )


def build_product_features() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT
      LOWER(item_type) AS product,
      CAST(year AS INTEGER)  AS year,
      CAST(month AS INTEGER) AS month,
      CAST(retail_sales     AS DOUBLE) AS retail_sales,
      CAST(retail_transfers AS DOUBLE) AS retail_transfers,
      CAST(warehouse_sales  AS DOUBLE) AS warehouse_sales
    FROM '{PARQUET}'
    """
    df = con.execute(q).fetchdf()

    # Aggregate to product-month (units = sum of the three)
    df["units"] = _safe_sum_units(df)
    df = (
        df.groupby(["product", "year", "month"], as_index=False)["units"]
          .sum()
          .sort_values(["product", "year", "month"])
          .reset_index(drop=True)
    )

    df["t"] = _make_time_key(df)

    # Lags & rolling means per product (no .apply; keep columns intact)
    g = df.groupby("product", group_keys=False)
    df["lag_1"] = g["units"].shift(1)
    df["lag_12"] = g["units"].shift(12)
    df["roll_3"] = g["units"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    df["roll_12"] = g["units"].transform(lambda s: s.rolling(window=12, min_periods=1).mean())

    # Seasonality (month sin/cos)
    df["m_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["m_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Fill NA for modeling simplicity
    df = df.fillna(0.0)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_parquet(OUT_DIR / "features_product.parquet", index=False)
    return df


def build_supplier_features() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT
      LOWER(item_type) AS product,
      LOWER(supplier)  AS supplier,
      CAST(year AS INTEGER)  AS year,
      CAST(month AS INTEGER) AS month,
      CAST(retail_sales     AS DOUBLE) AS retail_sales,
      CAST(retail_transfers AS DOUBLE) AS retail_transfers,
      CAST(warehouse_sales  AS DOUBLE) AS warehouse_sales
    FROM '{PARQUET}'
    """
    df = con.execute(q).fetchdf()

    # Aggregate to product-supplier-month
    df["units"] = _safe_sum_units(df)
    df = (
        df.groupby(["product", "supplier", "year", "month"], as_index=False)["units"]
          .sum()
          .sort_values(["product", "supplier", "year", "month"])
          .reset_index(drop=True)
    )

    df["t"] = _make_time_key(df)

    # Positive MoM deltas (ramp proxy) per (product, supplier) — no .apply, preserves keys
    gs = df.groupby(["product", "supplier"], group_keys=False)
    df["delta_plus"] = (
        gs["units"].diff().clip(lower=0.0).fillna(0.0).astype(float)
    )

    df.to_parquet(OUT_DIR / "features_supplier.parquet", index=False)
    return df


if __name__ == "__main__":
    p = build_product_features()
    s = build_supplier_features()
    print("Saved data/features_product.parquet and data/features_supplier.parquet")
