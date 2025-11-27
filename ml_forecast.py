# ml_forecast.py
#
# Unified, calibrated forecast backend for the API.
# - Loads quantile LightGBM models from evaluations/models_split.pkl
# - Optionally loads calibration from evaluations/calibration.pkl + calibration.json
# - Exposes predict_product_quantiles(product, year, month) so validation_agent works unchanged.

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features_product.parquet"

EVAL_DIR = Path(os.getenv("EVAL_DIR", "evaluations"))
MODEL_PATH = EVAL_DIR / "models_split.pkl"
CALIB_PATH = EVAL_DIR / "calibration.pkl"
CALIB_JSON = EVAL_DIR / "calibration.json"


class ForecastModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {MODEL_PATH}. "
                "Run scripts/02_train_lgbm_split.py first."
            )

        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        self.features = bundle["features"]          # list of feature column names
        self.models = bundle["models"]              # dict {"p10": model, "p50": model, "p90": model}

        # Optional calibration
        self.calib = None
        self.calib_meta = {}
        if CALIB_PATH.exists():
            with open(CALIB_PATH, "rb") as f:
                self.calib = pickle.load(f)
            try:
                with open(CALIB_JSON, "r") as f:
                    self.calib_meta = json.load(f)
            except Exception:
                self.calib_meta = {}

    def predict_raw(self, X_df: pd.DataFrame):
        """Predict raw quantiles for a feature dataframe X_df."""
        X = X_df[self.features].values
        p10 = self.models["p10"].predict(X)
        p50 = self.models["p50"].predict(X)
        p90 = self.models["p90"].predict(X)
        return np.asarray(p10), np.asarray(p50), np.asarray(p90)

    def apply_calibration(self, p10, p50, p90, method_override: str | None = None):
        """
        Apply calibration if available.
        method_override: "isotonic" | "conformal" | None.
        If None, use method stored in calibration.json (if any).
        """
        if self.calib is None and method_override is None:
            return p10, p50, p90

        method = method_override or self.calib_meta.get("method", "")

        if method == "isotonic" and self.calib:
            iso10 = self.calib["iso10"]
            iso50 = self.calib["iso50"]
            iso90 = self.calib["iso90"]
            p10_c = iso10.predict(p10)
            p50_c = iso50.predict(p50)
            p90_c = iso90.predict(p90)
            return np.asarray(p10_c), np.asarray(p50_c), np.asarray(p90_c)

        if method == "conformal" and self.calib:
            q = float(self.calib.get("conformal_q", 0.0))
            p10_c = p50 - q
            p90_c = p50 + q
            return np.asarray(p10_c), np.asarray(p50), np.asarray(p90_c)

        # Fallback: no calibration
        return p10, p50, p90


MODEL = ForecastModel()


def _build_feature_row(product: str, year: int, month: int) -> pd.DataFrame | None:
    """
    Build a single-row feature dataframe for the given product/year/month,
    using the same features that the LightGBM model was trained on.
    """
    if not FEATURES_PATH.exists():
        return None

    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["product"].str.lower() == product.lower()].copy()
    if df.empty:
        return None

    # Ensure year/month are numeric
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Try exact match
    row = df[(df["year"] == int(year)) & (df["month"] == int(month))]
    if row.empty:
        # Fallback: use the latest available row for that product
        row = df.sort_values(["year", "month"]).iloc[[-1]]

    r = row.iloc[0]

    # Build feature dict using the model's feature list
    feat_dict = {}
    for c in MODEL.features:
        # use 0.0 if missing in the parquet row
        feat_dict[c] = float(r.get(c, 0.0))

    return pd.DataFrame([feat_dict])


def predict_product_quantiles(product: str, year: int, month: int, calibrate_method: str | None = None) -> dict:
    """
    Public API used by validation_agent.

    product: e.g. "wine"
    year: int, e.g. 2020
    month: int, e.g. 1

    Returns:
      {
        "ok": True/False,
        "p10": ...,
        "p50": ...,
        "p90": ...,
        "error": optional message
      }
    """
    X = _build_feature_row(product, year, month)
    if X is None or X.empty:
        return {
            "ok": False,
            "error": f"No features found for product={product}, year={year}, month={month}"
        }

    try:
        p10, p50, p90 = MODEL.predict_raw(X)
        p10, p50, p90 = MODEL.apply_calibration(p10, p50, p90, method_override=calibrate_method)
        return {
            "ok": True,
            "p10": float(p10[0]),
            "p50": float(p50[0]),
            "p90": float(p90[0]),
        }
    except Exception as e:
        return {"ok": False, "error": f"Forecast failed: {e}"}
