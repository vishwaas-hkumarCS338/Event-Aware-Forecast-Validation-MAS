# scripts/05_plot_eval.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("evaluations")

def main():
    preds = pd.read_csv(EVAL_DIR / "preds_test.csv", parse_dates=["period"])
    with open(EVAL_DIR / "metrics.json","r") as f:
        metrics = json.load(f)

    # 1) Errors over time (absolute error)
    preds["ae"] = np.abs(preds["units"] - preds["p50"])
    plt.figure()
    preds.groupby("period")["ae"].mean().plot(marker="o")
    plt.title("Mean Absolute Error over Time (p50)")
    plt.xlabel("Period"); plt.ylabel("MAE")
    plt.tight_layout(); plt.savefig(EVAL_DIR / "errors.png"); plt.close()

    # 2) Coverage over time for 80% interval
    cov = ((preds["units"] >= preds["p10"]) & (preds["units"] <= preds["p90"])).astype(int)
    cov_ts = preds.assign(cov=cov).groupby("period")["cov"].mean()
    plt.figure()
    cov_ts.plot(marker="o")
    plt.axhline(0.8, linestyle="--")
    plt.title("Coverage (80% PI) over Time")
    plt.xlabel("Period"); plt.ylabel("Coverage")
    plt.tight_layout(); plt.savefig(EVAL_DIR / "coverage.png"); plt.close()

    # 3) Scatter: True vs Predicted p50
    plt.figure()
    plt.scatter(preds["p50"], preds["units"], alpha=0.5)
    mn = min(preds["p50"].min(), preds["units"].min()); mx = max(preds["p50"].max(), preds["units"].max())
    plt.plot([mn,mx],[mn,mx])
    plt.title("True vs Predicted (p50)")
    plt.xlabel("Pred p50"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(EVAL_DIR / "p50_scatter.png"); plt.close()

    # 4) Residuals histogram
    resid = preds["units"] - preds["p50"]
    plt.figure()
    plt.hist(resid.dropna(), bins=40)
    plt.title("Residuals Histogram (y - p50)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(EVAL_DIR / "residuals.png"); plt.close()

    # Print key metrics to console
    print(json.dumps(metrics, indent=2))
    print(f"Saved plots to: {EVAL_DIR.resolve()}")

if __name__ == "__main__":
    main()
