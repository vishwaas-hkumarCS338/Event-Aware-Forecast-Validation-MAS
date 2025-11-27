# scripts/01_make_splits.py
import argparse, json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("evaluations"); OUT_DIR.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--features", default="data/features_product.parquet")
    ap.add_argument("--split_end", required=True, help="YYYY-MM (last month in TRAIN)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    # Expect columns: product, year, month, units, and feature columns
    df = df[df["product"].str.lower() == args.product.lower()].copy()
    df["period"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    split_end = pd.to_datetime(args.split_end + "-01")

    df_train = df[df["period"] <= split_end].copy()
    df_test  = df[df["period"] >  split_end].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(OUT_DIR / "train.parquet", index=False)
    df_test.to_parquet(OUT_DIR / "test.parquet", index=False)

    summary = {
        "product": args.product,
        "split_end": args.split_end,
        "train_months": sorted(df_train["period"].dt.strftime("%Y-%m").unique().tolist()),
        "test_months": sorted(df_test["period"].dt.strftime("%Y-%m").unique().tolist()),
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test))
    }
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
