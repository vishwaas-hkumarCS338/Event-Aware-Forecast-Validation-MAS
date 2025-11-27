# scripts/04_calibrate_quantiles.py
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

OUT_DIR = Path("evaluations"); OUT_DIR.mkdir(exist_ok=True)

def select_calibration(train_df, months=3):
    # take the last K unique months as calibration
    per = sorted(train_df["period"].dt.to_period("M").unique())
    cal_months = per[-months:]
    return train_df[train_df["period"].dt.to_period("M").isin(cal_months)].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["isotonic","conformal"], required=True)
    ap.add_argument("--train_path", default="evaluations/train.parquet")
    ap.add_argument("--models_path", default="evaluations/models_split.pkl")
    ap.add_argument("--months", type=int, default=3, help="calibration months from tail of train")
    args = ap.parse_args()

    train = pd.read_parquet(args.train_path).copy()
    with open(args.models_path, "rb") as f:
        bundle = pickle.load(f)
    feats = bundle["features"]; models = bundle["models"]

    train["period"] = pd.to_datetime(train["year"].astype(str)+"-"+train["month"].astype(str)+"-01")
    cal = select_calibration(train, months=args.months).dropna(subset=feats+["units"]).copy()

    Xc = cal[feats].values; yc = cal["units"].values
    p10c = models["p10"].predict(Xc); p50c = models["p50"].predict(Xc); p90c = models["p90"].predict(Xc)

    calib = {"method": args.method, "months": args.months}

    if args.method == "isotonic":
        iso10 = IsotonicRegression(out_of_bounds="clip").fit(p10c, yc)
        iso50 = IsotonicRegression(out_of_bounds="clip").fit(p50c, yc)
        iso90 = IsotonicRegression(out_of_bounds="clip").fit(p90c, yc)
        calib["isotonic"] = True
        with open(OUT_DIR / "calibration.pkl", "wb") as f:
            pickle.dump({"iso10":iso10, "iso50":iso50, "iso90":iso90}, f)
    else:
        # Conformal additive offsets so that coverage of [p10+oL, p90+oU] hits ~80%
        resid50 = yc - p50c
        # Set symmetric offset via 90th percentile absolute residual for 80% PI (simple heuristic)
        q = np.quantile(np.abs(resid50), 0.90)
        calib["conformal_offset"] = float(q)
        with open(OUT_DIR / "calibration.pkl", "wb") as f:
            pickle.dump({"conformal_q": q}, f)

    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print(json.dumps(calib, indent=2))

if __name__ == "__main__":
    main()
