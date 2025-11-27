# reallocator.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog

DATA_DIR = Path("data")

# ---- Tunables via environment (with safe defaults)
FLEX_FLOOR_PCT = float(os.getenv("FLEX_FLOOR_PCT", "0.05"))      # at least 5% of current units
FLEX_FLOOR_CAP = float(os.getenv("FLEX_FLOOR_CAP", "2000"))      # but no more than 2,000 units per supplier
COST_LEADTIME_W = float(os.getenv("FLEX_COST_LEADTIME_WEIGHT", "0.4"))  # weight for lead time in cost (0..1)


def load_supplier_flex() -> pd.DataFrame:
    """Load supplier flex (80th pct positive MoM deltas) computed by training."""
    path = DATA_DIR / "supplier_flex.parquet"
    if not path.exists():
        # Empty frame with required columns
        return pd.DataFrame(columns=["product", "supplier", "flex_80p"])
    flex = pd.read_parquet(path)
    # normalize columns
    for col in ["product", "supplier"]:
        if col not in flex.columns:
            flex[col] = ""
        flex[col] = flex[col].astype(str).str.lower()
    if "flex_80p" not in flex.columns:
        # backward compat: if old column layout, coerce to 0
        flex["flex_80p"] = pd.to_numeric(flex.get("delta_plus", 0.0), errors="coerce").fillna(0.0)
    flex["flex_80p"] = pd.to_numeric(flex["flex_80p"], errors="coerce").fillna(0.0).astype(float)
    return flex[["product", "supplier", "flex_80p"]]


def maybe_load_supplier_attrs() -> pd.DataFrame:
    """
    Optionally load supplier attributes, e.g., data/supplier_attrs.parquet with columns:
    - supplier (str)
    - lead_time_days (float)
    """
    path = DATA_DIR / "supplier_attrs.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["supplier", "lead_time_days"])
    attrs = pd.read_parquet(path)
    if "supplier" not in attrs.columns:
        return pd.DataFrame(columns=["supplier", "lead_time_days"])
    attrs["supplier"] = attrs["supplier"].astype(str).str.lower()
    if "lead_time_days" not in attrs.columns:
        attrs["lead_time_days"] = np.nan
    attrs["lead_time_days"] = pd.to_numeric(attrs["lead_time_days"], errors="coerce")
    return attrs[["supplier", "lead_time_days"]]


def build_reallocation_inputs(
    product: str,
    year: int,
    month: int,
    duck_con,
    lost_supplier: str,
    shortfall_units: float,
):
    product = (product or "").lower()
    lost_supplier_lc = (lost_supplier or "").lower()

    # Current-month supplier units (category/product scope)
    q = f"""
    SELECT lower(supplier) AS supplier,
           SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE)) AS units
    FROM 'data/sales.parquet'
    WHERE lower(item_type)='{product}' AND year={int(year)} AND month={int(month)}
    GROUP BY supplier
    HAVING units IS NOT NULL
    ORDER BY units DESC
    """
    month_df = duck_con.execute(q).fetchdf()
    if month_df is None or month_df.empty:
        return None

    # Remove disrupted supplier
    cand = month_df[month_df["supplier"] != lost_supplier_lc].copy()
    if cand.empty:
        return None

    cand["product"] = product
    cand["supplier"] = cand["supplier"].astype(str).str.lower()
    cand["units"] = pd.to_numeric(cand["units"], errors="coerce").fillna(0.0).astype(float)

    # Merge historical flex
    flex = load_supplier_flex()
    if not flex.empty:
        flex = flex[flex["product"] == product]
        cand = cand.merge(flex, on=["product", "supplier"], how="left")
    if "flex_80p" not in cand.columns:
        cand["flex_80p"] = 0.0
    cand["flex_80p"] = pd.to_numeric(cand["flex_80p"], errors="coerce").fillna(0.0).astype(float)

    # Apply pragmatic floor: at least PCT of current units, capped
    floor = np.minimum(FLEX_FLOOR_PCT * cand["units"].to_numpy(float), FLEX_FLOOR_CAP)
    cand["flex_80p"] = np.maximum(cand["flex_80p"].to_numpy(float), floor)

    # Optional: merge attributes (lead time) to influence cost
    attrs = maybe_load_supplier_attrs()
    if not attrs.empty:
        cand = cand.merge(attrs, on="supplier", how="left")

    # Cost proxy:
    #   base = 1/(units+1)  (prefer bigger suppliers)
    #   if lead_time_days is available, blend it: cost = (1-w)*base + w*(lead_time_days/30)
    base_cost = 1.0 / (cand["units"] + 1.0)
    if "lead_time_days" in cand.columns and cand["lead_time_days"].notna().any():
        lt_norm = pd.to_numeric(cand["lead_time_days"], errors="coerce").fillna(0.0) / 30.0
        w = max(0.0, min(1.0, COST_LEADTIME_W))
        cand["cost"] = (1.0 - w) * base_cost + w * lt_norm
    else:
        cand["cost"] = base_cost

    suppliers = cand["supplier"].tolist()
    flex_vec = cand["flex_80p"].to_numpy(dtype=float)
    cost_vec = cand["cost"].to_numpy(dtype=float)
    shortfall = float(shortfall_units)
    total_flex = float(np.sum(flex_vec))

    return {
        "suppliers": suppliers,
        "flex": flex_vec,
        "cost": cost_vec,
        "shortfall": shortfall,
        "total_flex": total_flex,
    }


def solve_min_cost_cover(inputs):
    """
    Always returns a best-effort allocation.
    - If shortfall > total_flex, we cap the target to feasible_target = total_flex.
    - Minimizes c^T x subject to  sum x >= feasible_target, 0 <= x_i <= flex_i.
    """
    c = np.asarray(inputs["cost"], dtype=float)
    flex = np.asarray(inputs["flex"], dtype=float)
    shortfall = float(inputs["shortfall"])
    total_flex = float(inputs.get("total_flex", float(np.sum(flex))))
    n = len(c)

    class Res:
        pass

    res = Res()
    if n == 0:
        res.success = False
        res.x = np.zeros((0,), dtype=float)
        res.feasible_target = 0.0
        return res

    feasible_target = float(min(shortfall, max(0.0, total_flex)))

    if feasible_target <= 1e-9 or np.all(flex <= 1e-9):
        res.success = (feasible_target <= 1e-9)
        res.x = np.zeros((n,), dtype=float)
        res.feasible_target = feasible_target
        return res

    # A_ub x <= b_ub   with -1*sum(x) <= -feasible_target  (i.e., sum x >= feasible_target)
    A_ub = np.vstack([-np.ones((1, n)), np.eye(n)])
    b_ub = np.hstack([-feasible_target, flex])
    bounds = [(0.0, float(fl)) for fl in flex]

    out = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    res.success = bool(getattr(out, "success", False))
    res.x = getattr(out, "x", np.zeros((n,), dtype=float))
    res.feasible_target = feasible_target
    return res
