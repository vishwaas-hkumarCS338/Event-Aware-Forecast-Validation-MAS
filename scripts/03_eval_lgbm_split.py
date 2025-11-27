# scripts/03_eval_lgbm_split.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from math import sqrt

OUT_DIR = Path("evaluations"); OUT_DIR.mkdir(exist_ok=True)

def pinball(y, qhat, alpha):
    diff = y - qhat
    return np.mean(np.maximum(alpha*diff, (alpha-1)*diff))

def winkler_interval(y, lwr, upr, alpha=0.2):
    # alpha=0.2 -> 80% PI
    width = upr - lwr
    penalty = 0.0
    below = y < lwr
    above = y > upr
    penalty += (2/alpha)*(lwr - y) * below
    penalty += (2/alpha)*(y - upr) * above
    return np.mean(width + penalty)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--test_path", default="evaluations/test.parquet")
    ap.add_argument("--models_path", default="evaluations/models_split.pkl")
    args = ap.parse_args()

    test = pd.read_parquet(args.test_path)
    with open(args.models_path, "rb") as f:
        bundle = pickle.load(f)
    feats = bundle["features"]; models = bundle["models"]

    df = test.dropna(subset=feats + ["units"]).copy()
    X = df[feats].values; y = df["units"].values

    p10 = models["p10"].predict(X)
    p50 = models["p50"].predict(X)
    p90 = models["p90"].predict(X)

    preds = df[["product","year","month","units"]].copy()
    preds["p10"] = p10; preds["p50"] = p50; preds["p90"] = p90
    preds["period"] = pd.to_datetime(preds["year"].astype(str) + "-" + preds["month"].astype(str) + "-01")
    preds.to_csv(OUT_DIR / "preds_test.csv", index=False)

    mae = float(np.mean(np.abs(y - p50)))
    rmse = float(sqrt(np.mean((y - p50)**2)))
    mape = float(np.mean(np.abs((y - p50) / np.maximum(1e-6, y))) * 100.0)

    metrics = {
        "rows": int(len(df)),
        "periods_tested": sorted(preds["period"].dt.strftime("%Y-%m").unique().tolist()),
        "pinball_p10": float(pinball(y, p10, 0.1)),
        "pinball_p50": float(pinball(y, p50, 0.5)),
        "pinball_p90": float(pinball(y, p90, 0.9)),
        "mae_p50": mae, "rmse_p50": rmse, "mape_p50": mape,
        "coverage_80": float(np.mean((y >= p10) & (y <= p90))),
        "winkler_80": float(winkler_interval(y, p10, p90, alpha=0.2))
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
