# scripts/02_train_lgbm_split.py
import argparse, json, pickle
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np

OUT_DIR = Path("evaluations"); OUT_DIR.mkdir(exist_ok=True)

def _fit_quantile(train_df, features, alpha):
    train_df = train_df.dropna(subset=features + ["units"]).copy()
    X = train_df[features].values
    y = train_df["units"].values
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "verbosity": -1,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50
    }
    dtrain = lgb.Dataset(X, label=y, free_raw_data=True)
    model = lgb.train(params, dtrain, num_boost_round=1200)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--train_path", default="evaluations/train.parquet")
    ap.add_argument("--features_path", default="data/features_product.parquet")
    args = ap.parse_args()

    train = pd.read_parquet(args.train_path)
    # Build feature list by excluding identifiers/targets
    drop_cols = {"product","year","month","units","period"}
    features = [c for c in train.columns if c not in drop_cols and train[c].dtype != "O"]

    models = {}
    for name, alpha in [("p10",0.1),("p50",0.5),("p90",0.9)]:
        print(f"Training {name} (alpha={alpha}) on {args.product}...")
        models[name] = _fit_quantile(train, features, alpha)

    with open(OUT_DIR / "models_split.pkl", "wb") as f:
        pickle.dump({"features":features, "models":models}, f)

    meta = {"product": args.product, "features": features, "n_features": len(features)}
    with open(OUT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
