#!/usr/bin/env python3
"""
scripts/08_compare_unified.py

Builds a consistent feature row for a single scenario (wine, 2020-01),
computes raw ML quantiles from evaluations/models_split.pkl, calls the
calibrated forecast function from ml_forecast_calibrated (if present),
and writes a compare_unified.json result file.

Usage:
  from repo root with venv active:
    (.venv) python scripts/08_compare_unified.py
"""

import os
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure repo root on PYTHONPATH so local modules import reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Scenario (defaults)
PRODUCT = "wine"
YEAR = 2020
MONTH = 1
FORECAST_UNITS = 150000.0
DISRUPTED_SUPPLIER = "RELIABLE CHURCHILL LLLP"

# Paths
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluations"
EVAL_DIR.mkdir(exist_ok=True)
FEATURES_PATH = DATA_DIR / "features_product.parquet"
MODEL_BUNDLE_PATH = EVAL_DIR / "models_split.pkl"   # produced by scripts/02_train_lgbm_split.py
COMPARE_OUT = EVAL_DIR / "compare_unified.json"

# Local helper: try imports for calibrated module (it's OK if not present)
try:
    import ml_forecast_calibrated as calib_mod
except Exception:
    calib_mod = None

# Try to import ml_forecast for _latest_context fallback
try:
    import ml_forecast
except Exception:
    ml_forecast = None


def load_feature_row(product: str, year: int, month: int) -> dict:
    """
    Returns a dict of feature values for the requested product/year/month.
    Priority:
     1) exact row from data/features_product.parquet (if exists)
     2) ml_forecast._latest_context(product, year, month) if available
    Ensures numeric values and returns a dict.
    """
    product = product.lower()
    # 1) Try features file
    if FEATURES_PATH.exists():
        try:
            df_all = pd.read_parquet(FEATURES_PATH)
            # standardize product column to str lower if present
            if "product" in df_all.columns:
                df_all["product"] = df_all["product"].astype(str).str.lower()
            # find exact match
            mask = (df_all["product"] == product) & (df_all["year"] == int(year)) & (df_all["month"] == int(month))
            df_prod = df_all[mask]
            if not df_prod.empty:
                row = df_prod.iloc[-1].to_dict()
                # coerce numeric types where possible
                row = {k: (float(v) if _is_number_like(v) else v) for k, v in row.items()}
                print(f"Loaded exact feature row from {FEATURES_PATH} for {product} {year}-{month:02d}")
                return row
        except Exception as e:
            print("Warning: failed to load/parquet features file:", e)

    # 2) Fallback to ml_forecast._latest_context
    if ml_forecast and hasattr(ml_forecast, "_latest_context"):
        try:
            ctx_df = ml_forecast._latest_context(product, year, month)
            if ctx_df is not None and not ctx_df.empty:
                row = ctx_df.iloc[0].to_dict()
                row = {k: (float(v) if _is_number_like(v) else v) for k, v in row.items()}
                print(f"Built feature row via ml_forecast._latest_context for {product} {year}-{month:02d}")
                return row
        except Exception as e:
            print("Warning: ml_forecast._latest_context failed:", e)

    raise RuntimeError("Unable to build feature row: no features file and ml_forecast fallback unavailable.")


def _is_number_like(v):
    try:
        if isinstance(v, (int, float, np.number)):
            return True
        float(v)
        return True
    except Exception:
        return False


def ensure_time_index(feature_row: dict, year: int, month: int, df_all=None):
    """
    Ensure 't' exists. If training features file is available (df_all),
    compute t as months since first training month; otherwise fallback to year*12+month.
    """
    if "t" in feature_row and feature_row.get("t") is not None:
        return feature_row

    # try to compute a stable base from features file if loaded
    base = None
    if df_all is not None:
        try:
            # find min year/month in df_all
            df = df_all.copy()
            if "year" in df.columns and "month" in df.columns:
                df = df.dropna(subset=["year", "month"])
                min_year = int(df["year"].min())
                min_month = int(df[df["year"] == min_year]["month"].min())
                base = (min_year, min_month)
        except Exception:
            base = None

    if base is not None:
        # months since base (0-indexed)
        months_since_base = (int(year) - base[0]) * 12 + (int(month) - base[1])
        feature_row["t"] = float(months_since_base)
    else:
        feature_row["t"] = float(int(year) * 12 + int(month))

    return feature_row


def load_model_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    features = bundle.get("features", [])
    models = bundle.get("models", {})
    if not features or not models:
        raise RuntimeError("Model bundle missing 'features' or 'models' keys.")
    return features, models


def prepare_input_df(feature_row: dict, required_features):
    # ensure all required_features present; fill missing with 0.0
    row = {}
    for f in required_features:
        v = feature_row.get(f, 0.0)
        # coerce numeric; if not numeric, attempt conversion else use 0.0
        try:
            row[f] = float(v)
        except Exception:
            row[f] = 0.0
    return pd.DataFrame([row])


def predict_raw_from_bundle(feature_row: dict, model_bundle_path: Path):
    feats, models = load_model_bundle(model_bundle_path)
    Xdf = prepare_input_df(feature_row, feats)
    X = Xdf.values
    # models should be lightgbm.Booster objects with .predict
    preds = {}
    for k in ["p10", "p50", "p90"]:
        if k not in models:
            # try q10/q50/q90 fallback
            alt = k.replace("p", "q")
            if alt in models:
                mdl = models[alt]
            else:
                raise KeyError(f"Model for quantile {k} not found in bundle.")
        else:
            mdl = models[k]
        preds[k] = float(mdl.predict(X)[0])
    return preds, feats


def call_calibrated(feature_row: dict):
    """
    Calls ml_forecast_calibrated.ml_forecast_calibrated_row(feature_row)
    if available. If not present, returns None.
    """
    if calib_mod is None:
        print("Calibrated module not available (ml_forecast_calibrated). Skipping calibrated call.")
        return None

    # attempt direct call; this function previously existed in your environment.
    if hasattr(calib_mod, "ml_forecast_calibrated_row"):
        try:
            # make sure feature_row values are numeric floats
            fr = {k: float(v) if _is_number_like(v) else v for k, v in feature_row.items()}
            out = calib_mod.ml_forecast_calibrated_row(fr)
            return out
        except Exception as e:
            print("Error calling ml_forecast_calibrated.ml_forecast_calibrated_row:", e)
            return None

    # fallback: if the module exposes a ForecastModel-like API, try that
    if hasattr(calib_mod, "MODEL") and hasattr(calib_mod, "ml_forecast_for_row"):
        try:
            fr = {k: float(v) if _is_number_like(v) else v for k, v in feature_row.items()}
            out = calib_mod.ml_forecast_for_row(fr)
            return out
        except Exception as e:
            print("Error calling calibration fallback functions:", e)
            return None

    print("No recognized entry point in ml_forecast_calibrated; skipping calibration.")
    return None


def main():
    # Build feature row (try to also keep df_all for t base computation)
    df_all = None
    if FEATURES_PATH.exists():
        try:
            df_all = pd.read_parquet(FEATURES_PATH)
        except Exception:
            df_all = None

    feature_row = load_feature_row(PRODUCT, YEAR, MONTH)

    # ensure 't' is present and consistent with training if possible
    feature_row = ensure_time_index(feature_row, YEAR, MONTH, df_all=df_all)

    # print a short diagnostic
    print("\n=== Feature row preview (trimmed) ===")
    for k, v in sorted(feature_row.items())[:20]:
        print(f" {k:15s}: {v}")
    print("... (total keys: {})\n".format(len(feature_row)))

    # Predict raw (from evaluations/models_split.pkl)
    try:
        raw_preds, model_features = predict_raw_from_bundle(feature_row, MODEL_BUNDLE_PATH)
        print("Raw model features used:", model_features)
        print("Raw ML quantiles:", raw_preds)
    except Exception as e:
        print("Failed to predict raw quantiles from model bundle:", e)
        raw_preds = None
        model_features = None

    # Call calibrated predictor if present
    calib_out = call_calibrated(feature_row)
    if calib_out is not None:
        print("\nCalibrated / API output (ml_forecast_calibrated):")
        print(json.dumps(calib_out, indent=2))
    else:
        print("\nNo calibrated output produced.")

    # Compare with external forecast
    comparison = {
        "scenario": {
            "product": PRODUCT,
            "period": f"{YEAR:04d}-{MONTH:02d}",
            "forecast_units": FORECAST_UNITS,
            "disrupted_supplier": DISRUPTED_SUPPLIER
        },
        "feature_row_snapshot": {k: feature_row.get(k) for k in (model_features if model_features else list(feature_row.keys()) )[:50]},
        "raw_ml": raw_preds,
        "calibrated": calib_out,
    }

    # If raw_preds present compute relative error vs external forecast
    if raw_preds and "p50" in raw_preds:
        p50 = float(raw_preds["p50"])
        comparison["relative_error_vs_p50"] = (FORECAST_UNITS - p50) / max(1.0, p50)
        comparison["forecast_minus_p50"] = FORECAST_UNITS - p50

    # write output
    with open(COMPARE_OUT, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved comparison to: {COMPARE_OUT.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
