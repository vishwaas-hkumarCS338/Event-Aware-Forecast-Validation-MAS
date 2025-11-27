# ml_forecast_calibrated.py
import os, pickle, json
import numpy as np
import pandas as pd

EVAL_DIR = os.getenv("EVAL_DIR", "evaluations")
MODEL_PATH = os.path.join(EVAL_DIR, "models_split.pkl")
CALIB_PATH = os.path.join(EVAL_DIR, "calibration.pkl")
CALIB_JSON = os.path.join(EVAL_DIR, "calibration.json")


class ForecastModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        self.features = bundle["features"]
        self.models = bundle["models"]

        self.calib = None
        self.calib_meta = {}
        if os.path.exists(CALIB_PATH):
            with open(CALIB_PATH, "rb") as f:
                self.calib = pickle.load(f)
            try:
                with open(CALIB_JSON, "r") as f:
                    self.calib_meta = json.load(f)
            except Exception:
                self.calib_meta = {}

    def predict_raw(self, X_df: pd.DataFrame):
        X = X_df[self.features].values
        p10 = self.models["p10"].predict(X)
        p50 = self.models["p50"].predict(X)
        p90 = self.models["p90"].predict(X)
        return np.asarray(p10), np.asarray(p50), np.asarray(p90)

    def apply_calibration(self, p10, p50, p90, method_override=None):
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
        return p10, p50, p90


MODEL = ForecastModel()


def ml_forecast_calibrated_row(row_dict: dict, calibrate_method: str | None = None) -> dict:
    """
    row_dict: dict with feature columns (t, lag_1, lag_12, roll_3, roll_12, m_sin, m_cos)
    """
    df = pd.DataFrame([row_dict])
    missing = [c for c in MODEL.features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for calibrated forecast: {missing}")
    p10, p50, p90 = MODEL.predict_raw(df)
    p10, p50, p90 = MODEL.apply_calibration(p10, p50, p90, method_override=calibrate_method)
    return {"p10": float(p10[0]), "p50": float(p50[0]), "p90": float(p90[0])}
