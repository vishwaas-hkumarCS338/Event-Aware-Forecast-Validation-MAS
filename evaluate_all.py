#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_all.py
End-to-end evaluation harness for the Event-Aware Forecast Validation system.

What it does:
1) ML backtest (LightGBM quantile) on historical months
   - Pinball loss (p10/p50/p90), MAE/RMSE/MAPE (p50),
     80% coverage, Winkler score.

2) Scenario runner against the FastAPI endpoint /validate
   - Feasibility rate, remaining gap, total flex, covered units,
     allocation concentration (Herfindahl), ML-vs-LLM agreement,
     retrieval top-1 similarity, LLM parse rate.

3) API reliability
   - Success rate, latency (p50/p90/p99), average response bytes.

Outputs:
- Prints summaries to terminal
- Saves:
  - evaluations/ml_backtest_summary.json
  - evaluations/scenario_results.csv
  - evaluations/scenario_summary.json
  - evaluations/api_summary.json
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests

# ---------- CONFIG ----------
API_BASE = os.environ.get("EVAL_API_BASE", "http://127.0.0.1:8000")
PRODUCT = os.environ.get("EVAL_PRODUCT", "wine")

# Periods to test (YYYY-MM)
PERIODS = os.environ.get("EVAL_PERIODS", "2019-11,2019-12,2020-01").split(",")

# Target grid to probe feasibility
TARGETS = [int(x) for x in os.environ.get("EVAL_TARGETS", "90000,110000,150000,200000").split(",")]

# How many top suppliers to iterate as "disrupted_supplier" in scenarios
TOP_SUPPLIERS_K = int(os.environ.get("EVAL_TOP_SUPPLIERS", "5"))

# Files produced by your pipeline
FEATURES_PRODUCT = Path("data/features_product.parquet")  # contains monthly features & units
MODELS_PKL = Path("models/product_quantile_models.pkl")   # to ensure ML training exists

# Output dir
OUT_DIR = Path("evaluations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- METRIC HELPERS ----------
def pinball_loss(y, q_pred, alpha):
    diff = y - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))

def winkler_score(y, lo, hi, alpha=0.2):
    y = np.asarray(y)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    w = hi - lo
    below = (y < lo).astype(float)
    above = (y > hi).astype(float)
    return float(np.mean(w + 2 / alpha * (lo - y) * below + 2 / alpha * (y - hi) * above))

def herfindahl_from_alloc_list(alloc_list: List[Dict]) -> float:
    if not alloc_list:
        return 0.0
    df = pd.DataFrame(alloc_list)
    total = float(df["alloc_units"].sum())
    if total <= 0:
        return 0.0
    shares = df["alloc_units"] / total
    return float((shares ** 2).sum())

# ---------- ML BACKTEST ----------
def load_actuals_series(product: str) -> pd.DataFrame:
    """
    Returns a monthly actuals dataframe with columns:
    period (YYYY-MM), units (float)
    """
    if not FEATURES_PRODUCT.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PRODUCT}. Run `python features.py` first."
        )
    fp = pd.read_parquet(FEATURES_PRODUCT)
    # Expect columns: product, year, month, units (already created in features.py)
    fp = fp[fp["product"].str.lower() == product.lower()].copy()
    # For safety: coerce ints
    fp["year"] = fp["year"].astype(int)
    fp["month"] = fp["month"].astype(int)
    # Aggregate monthly total units (if multiple rows per month)
    g = fp.groupby(["year", "month"], as_index=False)["units"].sum()
    g["period"] = g["year"].astype(str) + "-" + g["month"].astype(str).str.zfill(2)
    g = g.sort_values(["year", "month"])
    return g[["period", "units"]]

def predict_quantiles(product: str, period: str) -> Dict[str, float]:
    """
    Uses your ml_forecast.predict_product_quantiles API to get p10/p50/p90.
    """
    from ml_forecast import predict_product_quantiles
    y, m = map(int, period.split("-"))
    res = predict_product_quantiles(product, y, m)
    if not res.get("ok", False):
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan}
    return {"p10": float(res["p10"]), "p50": float(res["p50"]), "p90": float(res["p90"])}

def ml_backtest(product: str, periods: List[str]) -> Dict:
    actuals = load_actuals_series(product)
    actuals = actuals[actuals["period"].isin(periods)].copy()
    if actuals.empty:
        raise RuntimeError(f"No actuals found for requested periods: {periods}")

    preds = []
    for period in actuals["period"].tolist():
        q = predict_quantiles(product, period)
        preds.append({"period": period, **q})
    qdf = pd.DataFrame(preds)

    df = actuals.merge(qdf, on="period", how="left")
    df = df.dropna(subset=["p10", "p50", "p90"])
    if df.empty:
        raise RuntimeError("No quantile predictions available—did you run `python train_forecaster.py`?")

    y = df["units"].to_numpy()
    p10 = df["p10"].to_numpy()
    p50 = df["p50"].to_numpy()
    p90 = df["p90"].to_numpy()

    # Metrics
    out = {}
    out["periods_evaluated"] = df["period"].tolist()
    out["pinball_p10"] = pinball_loss(y, p10, 0.1)
    out["pinball_p50"] = pinball_loss(y, p50, 0.5)
    out["pinball_p90"] = pinball_loss(y, p90, 0.9)
    out["mae_p50"] = float(np.mean(np.abs(y - p50)))
    out["rmse_p50"] = float(np.sqrt(np.mean((y - p50) ** 2)))
    out["mape_p50"] = float(np.mean(np.abs((y - p50) / np.maximum(1e-9, y)))) * 100.0
    cover_80 = np.mean((y >= p10) & (y <= p90))
    out["coverage_80"] = float(cover_80)
    out["winkler_80"] = winkler_score(y, p10, p90, alpha=0.2)

    # Save details too
    df.to_csv(OUT_DIR / "ml_backtest_detail.csv", index=False)
    with open(OUT_DIR / "ml_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

# ---------- API SCENARIOS ----------
def post_validate(product: str, period: str, forecast_units: int, disrupted_supplier: str) -> Tuple[int, float, Dict]:
    url = f"{API_BASE}/validate"
    payload = {
        "product": product,
        "period": period,
        "forecast_units": forecast_units,
        "disrupted_supplier": disrupted_supplier,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=60)
    latency_ms = (time.time() - t0) * 1000.0
    try:
        data = r.json()
    except Exception:
        data = {}
    return r.status_code, latency_ms, data

def seed_top_suppliers(product: str, period: str) -> List[str]:
    """
    One seed call to get top suppliers list for a period.
    """
    code, _, data = post_validate(product, period, forecast_units=100000, disrupted_supplier="RELIABLE CHURCHILL LLLP")
    tops = []
    try:
        tops = [x["supplier"] for x in data.get("supplier_analysis", {}).get("top_suppliers", [])]
    except Exception:
        pass
    return tops

def run_scenarios(product: str, periods: List[str], targets: List[int], top_k: int = 5) -> pd.DataFrame:
    rows = []
    for period in periods:
        tops = seed_top_suppliers(product, period)[:top_k] or ["RELIABLE CHURCHILL LLLP"]
        for disrupted in tops:
            for tgt in targets:
                status, latency_ms, data = post_validate(product, period, tgt, disrupted)
                sim = data.get("simulation", {}) if isinstance(data, dict) else {}
                rag = data.get("event_rag", {}) if isinstance(data, dict) else {}
                chunks = data.get("retrieved_context_chunks", []) if isinstance(data, dict) else []
                top1_sim = chunks[0]["score"] if chunks else None
                final_verdict = data.get("final_verdict")
                llm_verdict = None
                if rag.get("ok") and isinstance(rag.get("llm_json"), dict):
                    llm_verdict = rag["llm_json"].get("verdict")
                allocations = sim.get("allocations", [])
                row = {
                    "period": period,
                    "supplier": disrupted,
                    "target": tgt,
                    "status": status,
                    "latency_ms": latency_ms,
                    "remaining_gap": sim.get("remaining_gap"),
                    "covered_units": sim.get("covered_units"),
                    "total_flex": sim.get("total_flex") or sim.get("covered_units"),
                    "feasible": (sim.get("remaining_gap") is not None and sim.get("remaining_gap") <= 1e-6),
                    "feasible_target": sim.get("feasible_target"),
                    "final_verdict": final_verdict,
                    "llm_verdict": llm_verdict,
                    "event_rag_ok": bool(rag.get("ok")),
                    "top1_similarity": top1_sim,
                    "alloc_herfindahl": herfindahl_from_alloc_list(allocations),
                    "resp_bytes": len(json.dumps(data)) if isinstance(data, dict) else 0,
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT_DIR / "scenario_results.csv", index=False)
    return df

def summarize_scenarios(df: pd.DataFrame) -> Dict:
    out = {}
    if df.empty:
        return out
    out["rows"] = int(len(df))
    out["success_rate"] = float((df["status"] // 100 == 2).mean())
    out["error_rate_5xx"] = float((df["status"] // 100 == 5).mean())
    for p in (50, 90, 99):
        out[f"latency_p{p}_ms"] = float(np.percentile(df["latency_ms"], p))
    out["feas_rate"] = float(df["feasible"].mean())
    out["avg_remaining_gap"] = float(df["remaining_gap"].dropna().mean()) if "remaining_gap" in df else None
    out["avg_total_flex"] = float(df["total_flex"].dropna().mean()) if "total_flex" in df else None
    out["avg_covered_units"] = float(df["covered_units"].dropna().mean()) if "covered_units" in df else None
    if "final_verdict" in df and "llm_verdict" in df:
        agree = (df["final_verdict"] == df["llm_verdict"]).mean()
        out["ml_llm_agreement"] = float(agree)
    out["mean_top1_similarity"] = float(df["top1_similarity"].dropna().mean()) if "top1_similarity" in df else None
    out["llm_parse_rate"] = float(df["event_rag_ok"].mean()) if "event_rag_ok" in df else None
    out["median_alloc_concentration"] = float(df["alloc_herfindahl"].dropna().median()) if "alloc_herfindahl" in df else None
    out["avg_resp_bytes"] = float(df["resp_bytes"].dropna().mean()) if "resp_bytes" in df else None

    with open(OUT_DIR / "scenario_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "api_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if k.startswith("latency_") or "rate" in k or "bytes" in k}, f, indent=2)
    return out

# ---------- MAIN ----------
def main():
    print("=== Evaluate All ===")
    # Sanity checks
    if not MODELS_PKL.exists():
        print(f"[WARN] {MODELS_PKL} not found. Run: python train_forecaster.py")

    # 1) ML BACKTEST
    print("\n[1/3] ML Backtest (LightGBM quantiles)")
    try:
        ml_summary = ml_backtest(PRODUCT, PERIODS)
        print(json.dumps(ml_summary, indent=2))
    except Exception as e:
        print(f"[ML] Skipped due to error: {e}")
        ml_summary = {}

    # 2) SCENARIO GRID via API
    print("\n[2/3] Scenario evaluation via /validate")
    try:
        df = run_scenarios(PRODUCT, PERIODS, TARGETS, top_k=TOP_SUPPLIERS_K)
        if df.empty:
            print("[Scenarios] No rows produced.")
            scenario_summary = {}
        else:
            scenario_summary = summarize_scenarios(df)
            print(json.dumps(scenario_summary, indent=2))
    except Exception as e:
        print(f"[Scenarios] Skipped due to error: {e}")
        scenario_summary = {}

    # 3) API quick reliability summary is embedded in scenario_summary
    print("\n[3/3] API summary")
    print(json.dumps({k: v for k, v in scenario_summary.items() if k.startswith("latency_") or "rate" in k}, indent=2))

    print(f"\nArtifacts saved in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
