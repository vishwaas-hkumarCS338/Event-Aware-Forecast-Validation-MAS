# train_forecaster.py (patched for older LightGBM versions)
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = ["month","m_sin","m_cos","lag_1","lag_12","roll_3","roll_12"]


def _train_quantile(df: pd.DataFrame, alpha: float):
    df = df.copy()
    df["ts"] = df["year"] * 12 + df["month"]
    val_cut = int(df["ts"].max()) - 6

    X = df[FEATURES]
    y = df["units"].astype(float)

    train_idx = df["ts"] <= val_cut
    valid_idx = df["ts"] > val_cut

    train_data = lgb.Dataset(X[train_idx], label=y[train_idx])

    # Older LightGBM does NOT accept valid_sets or early stopping reliably on Windows
    # So we use a simple training run
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=400  # fixed boosting rounds
    )

    return model


def train_product_models():
    fpath = DATA_DIR / "features_product.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fpath}. Run: python features.py")

    f = pd.read_parquet(fpath)

    models = {}
    for name, alpha in [("p10",0.10),("p50",0.50),("p90",0.90)]:
        print(f"Training LightGBM quantile model: {name} (alpha={alpha})...")
        models[name] = _train_quantile(f, alpha)

    bundle = {"models": models, "features": FEATURES}
    with open(MODEL_DIR / "product_quantile_models.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    print("Saved models/product_quantile_models.pkl (p10/p50/p90).")


def compute_supplier_flex():
    spath = DATA_DIR / "features_supplier.parquet"
    if not spath.exists():
        raise FileNotFoundError(f"Missing {spath}. Run: python features.py")

    fs = pd.read_parquet(spath)
    agg = (
        fs.groupby(["product","supplier"])["delta_plus"]
          .quantile(0.80)
          .reset_index()
          .rename(columns={"delta_plus": "flex_80p"})
    )

    out = DATA_DIR / "supplier_flex.parquet"
    agg.to_parquet(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    train_product_models()
    compute_supplier_flex()
