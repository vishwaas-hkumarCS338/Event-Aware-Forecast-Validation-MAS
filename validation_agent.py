# validation_agent.py
import os
import json
from typing import Dict, Optional, Any, List
import numpy as np
import duckdb

from dotenv import load_dotenv

# ---- Feature/ML + LP imports
from ml_forecast import predict_product_quantiles
from reallocator import build_reallocation_inputs, solve_min_cost_cover

# ---- Optional retriever + OpenAI (guarded by env flags)
try:
    from retriever_simple import Retriever  # your local retriever
except Exception:
    Retriever = None  # guard at runtime

load_dotenv()

# ---------------- Env flags & client setup ----------------
DISABLE_LLM = os.getenv("VALIDATOR_DISABLE_LLM", "0") == "1"
LLM_TIMEOUT_S = float(os.getenv("VALIDATOR_LLM_TIMEOUT_S", "6.0"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if not DISABLE_LLM and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None  # degrade gracefully


def _fmt_int(x: Optional[float]) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def _fmt_float(x: Optional[float]) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "—"


class ValidationAgent:
    def __init__(self, db_path: str = "data/sales.duckdb"):
        # DuckDB connection (works with Parquet paths as well)
        if os.path.exists(db_path):
            self.con = duckdb.connect(database=db_path)
        else:
            self.con = duckdb.connect(database=":memory:")

        # Retriever is optional; guard if module missing or disabled
        self.retriever = Retriever() if (not DISABLE_LLM and Retriever) else None

    # ---------- HISTORICAL ----------
    def _nearest_period_with_data(self, product: str, year: int, month: int):
        target = year * 12 + month
        q = f"""
        SELECT year, month,
               ABS((year*12+month) - {target}) AS dist,
               COUNT(*) AS rows
        FROM 'data/sales.parquet'
        WHERE lower(item_type)='{product.lower()}'
        GROUP BY year, month
        ORDER BY dist ASC
        LIMIT 1
        """
        row = self.con.execute(q).fetchone()
        return row  # (year, month, dist, rows) or None

    def query_historical(self, product: str, period: str) -> Dict[str, Any]:
        try:
            y, m = map(int, str(period).split("-"))
        except Exception:
            return {"found": False, "error": f"Bad period format: {period}"}

        nearest = self._nearest_period_with_data(product, y, m)
        if not nearest:
            return {"found": False}

        uy, um, dist, rows = nearest
        q = f"""
        SELECT
          SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE)) AS total_units
        FROM 'data/sales.parquet'
        WHERE lower(item_type)='{product.lower()}' AND year={int(uy)} AND month={int(um)};
        """
        total = self.con.execute(q).fetchone()[0] or 0

        q_rows = f"""
        SELECT AVG(CAST(retail_sales AS DOUBLE)) AS r,
               AVG(CAST(retail_transfers AS DOUBLE)) AS t,
               AVG(CAST(warehouse_sales AS DOUBLE)) AS w
        FROM 'data/sales.parquet'
        WHERE lower(item_type)='{product.lower()}' AND year={int(uy)} AND month={int(um)};
        """
        r, t, w = self.con.execute(q_rows).fetchone()
        mean_per_row = (r or 0) + (t or 0) + (w or 0)

        return {
            "found": True,
            "requested_period": f"{y:04d}-{m:02d}",
            "used_period": f"{int(uy):04d}-{int(um):02d}",
            "used_reason": f"nearest_match_dist_{int(dist)}_months",
            "total_month_units": int(total),
            "mean_per_row_units": float(mean_per_row),
            "rows_count": int(rows),
            "explain": [
                f"used nearest available period {int(uy):04d}-{int(um):02d} (dist={int(dist)} months)",
                "units = retail_sales + retail_transfers + warehouse_sales",
            ],
        }

    def heuristic_flag(self, forecast_units: float, hist: Dict[str, Any]) -> str:
        if not hist or not hist.get("found"):
            return "no_history"
        mean_units = hist.get("total_month_units", 0) or 0
        if mean_units == 0:
            return "no_history"
        if forecast_units > 1.3 * mean_units:
            return "forecast_higher_than_historical"
        elif forecast_units < 0.7 * mean_units:
            return "forecast_lower_than_historical"
        return "forecast_within_historical_range"

    # ---------- SUPPLIER ----------
    def supplier_analysis(self, product: str, year: int, month: int, top_n: int = 10) -> Dict[str, Any]:
        prod = product.lower()
        q_top = f"""
        SELECT lower(supplier) AS supplier,
               SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE)) AS units
        FROM 'data/sales.parquet'
        WHERE lower(item_type)='{prod}' AND year={year} AND month={month}
        GROUP BY supplier
        ORDER BY units DESC
        LIMIT {top_n}
        """
        rows = self.con.execute(q_top).fetchdf()

        q_total = f"""
        SELECT SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE)) AS units
        FROM 'data/sales.parquet'
        WHERE lower(item_type)='{prod}' AND year={year} AND month={month}
        """
        total = self.con.execute(q_total).fetchone()[0] or 0

        suppliers = []
        for _, r in rows.iterrows():
            share = (float(r["units"]) / (float(total) + 1e-9)) if total else 0.0
            suppliers.append(
                {"supplier": r["supplier"], "units": int(r["units"]), "share": round(share, 4)}
            )

        q_season = f"""
        SELECT AVG(units) FROM (
           SELECT month,
                  SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE)) AS units
           FROM 'data/sales.parquet'
           WHERE lower(item_type)='{prod}'
           GROUP BY year, month
        ) WHERE month={month}
        """
        season_mean = self.con.execute(q_season).fetchone()[0] or 0

        return {
            "total_month_units": int(total),
            "top_suppliers": suppliers,
            "season_month_mean_units": float(season_mean),
        }

    # ---------- LLM (explanation / events) ----------
    def _llm_explain(
        self, product: str, period: str, forecast_units: float, hist: Dict[str, Any], context_text: str
    ) -> Dict[str, Any]:
        if DISABLE_LLM:
            return {"ok": False, "skipped": True, "reason": "LLM disabled via VALIDATOR_DISABLE_LLM=1"}
        if client is None:
            return {"ok": False, "skipped": True, "reason": "LLM client unavailable"}

        prompt = f"""
You are an analytical assistant. Product={product}, period={period}, forecast={forecast_units}.
Historical total (nearest month): {hist.get('total_month_units')}
Context:
{context_text}

Return JSON only with keys: verdict (UNDER-ESTIMATING/OVER-ESTIMATING/REASONABLE), demand_adj_pct, supply_loss_pct, confidence (0-100), reasons[], citations[].
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Be concise and numeric."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                timeout=LLM_TIMEOUT_S,
            )
            txt = resp.choices[0].message.content.strip()

            # ---- Robust parse: strip code fences ```json ... ```
            cleaned = txt.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.split("\n", 1)[-1]  # drop language line if present
                cleaned = cleaned.strip().rstrip("`").strip()

            try:
                return {"ok": True, "llm_json": json.loads(cleaned), "raw": txt}
            except Exception:
                return {"ok": False, "raw": txt}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------- MAIN ----------
    def validate_forecast(
        self,
        product: str,
        period: str,
        forecast_units: int,
        disrupted_supplier: Optional[str] = None,
    ) -> Dict[str, Any]:
        y, m = map(int, period.split("-"))

        # 1) Historical (may fall back to nearest month)
        hist = self.query_historical(product, period)

        # Decide baseline month to use for supplier view & reallocation (align with history)
        base_year, base_month = y, m
        if hist.get("found") and isinstance(hist.get("used_period"), str):
            try:
                by, bm = map(int, hist["used_period"].split("-"))
                base_year, base_month = by, bm
            except Exception:
                pass  # keep original y, m if parsing fails

        # 2) ML quantile forecast (product-level)
        ml = predict_product_quantiles(product, y, m)  # {ok,p10,p50,p90}
        ml_ok = bool(ml.get("ok"))

        # 3) Supplier view (use baseline month so it aligns with history)
        supp_info = self.supplier_analysis(product, base_year, base_month)

        # 4) Retrieve context and LLM narrative (for explanation)
        rag: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "Disabled or unavailable"}
        chunks: List[Dict[str, Any]] = []
        if not DISABLE_LLM and self.retriever is not None:
            try:
                query_text = f"{product} {period} demand seasonality supply disruption weather"
                chunks = self.retriever.retrieve(query_text, top_k=3) or []
            except Exception as e:
                chunks = []
                rag = {"ok": False, "error": f"retriever_failed: {e}"}

            try:
                ctx = "\n\n---\n\n".join([c.get("text", c.get("chunk", "")) for c in chunks])
            except Exception:
                ctx = ""
            rag = self._llm_explain(product, period, forecast_units, hist, ctx)

        # 5) Verdict logic (prefer ML bands)
        final_verdict = "UNKNOWN"
        if ml_ok:
            p10, p50, p90 = ml["p10"], ml["p50"], ml["p90"]
            if forecast_units < p10:
                final_verdict = "UNDER-ESTIMATING"
            elif forecast_units > p90:
                final_verdict = "OVER-ESTIMATING"
            else:
                final_verdict = "REASONABLE"
        else:
            v = (rag.get("llm_json") or {}).get("verdict") if isinstance(rag, dict) else None
            if v in {"UNDER-ESTIMATING", "OVER-ESTIMATING", "REASONABLE"}:
                final_verdict = v
            else:
                final_verdict = {
                    "forecast_higher_than_historical": "OVER-ESTIMATING",
                    "forecast_lower_than_historical": "UNDER-ESTIMATING",
                    "forecast_within_historical_range": "REASONABLE",
                }.get(self.heuristic_flag(forecast_units, hist), "UNKNOWN")

        # 6) If supplier disruption is given, compute shortfall and try reallocation
        simulation: Dict[str, Any] = {}
        if disrupted_supplier:
            # Historical total for the baseline month
            base_q = f"""
            SELECT SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE))
            FROM 'data/sales.parquet'
            WHERE lower(item_type)='{product.lower()}' AND year={base_year} AND month={base_month}
            """
            total_hist = float(self.con.execute(base_q).fetchone()[0] or 0.0)

            # Lost capacity for that supplier in the baseline month
            lost_q = f"""
            SELECT SUM(CAST(retail_sales AS DOUBLE)+CAST(retail_transfers AS DOUBLE)+CAST(warehouse_sales AS DOUBLE))
            FROM 'data/sales.parquet'
            WHERE lower(item_type)='{product.lower()}'
              AND year={base_year} AND month={base_month}
              AND lower(supplier)=lower('{disrupted_supplier}')
            """
            lost_cap = float(self.con.execute(lost_q).fetchone()[0] or 0.0)

            available_without_lost = max(0.0, total_hist - lost_cap)
            shortfall = max(0.0, float(forecast_units) - available_without_lost)

            inputs = build_reallocation_inputs(product, base_year, base_month, self.con, disrupted_supplier, shortfall)
            if inputs:
                sol = solve_min_cost_cover(inputs)
                allocated = []
                covered = 0.0
                if getattr(sol, "x", None) is not None:
                    x = sol.x
                    for sup, add in zip(inputs["suppliers"], x):
                        if add > 1e-6:
                            allocated.append({"supplier": sup, "alloc_units": float(round(add, 2))})
                    covered = float(round(float(np.sum(x)), 2))

                remaining_gap = float(round(max(0.0, shortfall - covered), 2))
                simulation = {
                    "supplier_found": True,
                    "lost_supplier": disrupted_supplier,
                    "supplier_period_used": f"{base_year:04d}-{base_month:02d}",
                    "total_hist_units": int(round(total_hist)),
                    "lost_capacity_units": int(round(lost_cap)),
                    "available_without_lost": int(round(available_without_lost)),
                    "forecast_units": int(round(forecast_units)),
                    "shortfall_units": int(round(shortfall)),
                    "covered_units": covered,
                    "remaining_gap": remaining_gap,
                    "allocations": allocated,
                    "lp_success": bool(getattr(sol, "success", False)),
                    # Transparency:
                    "total_flex": float(round(float(inputs.get("total_flex", 0.0)), 2)),
                    "feasible_target": float(round(float(getattr(sol, "feasible_target", shortfall)), 2)),
                }
            else:
                simulation = {
                    "supplier_found": False,
                    "lost_supplier": disrupted_supplier,
                    "supplier_period_used": f"{base_year:04d}-{base_month:02d}",
                    "forecast_units": int(round(forecast_units)),
                    "shortfall_units": int(round(shortfall)),
                    "reason": "no alternative suppliers or flex data for reallocation",
                }

        # ---- Supply feasibility verdict (in addition to demand realism)
        supply_feasible = None
        supply_reason = None
        if simulation:
            rem = simulation.get("remaining_gap", None)
            if rem is not None:
                if rem <= 1e-6:
                    supply_feasible = True
                    supply_reason = "All demand met under flex constraints."
                else:
                    supply_feasible = False
                    supply_reason = f"Shortfall of {rem} units cannot be covered within flex limits."

        # ---- Human-friendly summary
        summary = None
        try:
            p10 = ml.get("p10") if ml else None
            p50 = ml.get("p50") if ml else None
            p90 = ml.get("p90") if ml else None
            avail = (simulation or {}).get("available_without_lost")
            tflex = (simulation or {}).get("total_flex")
            covered = (simulation or {}).get("covered_units")
            gap = (simulation or {}).get("remaining_gap")
            rec_target = None
            if avail is not None and tflex is not None:
                rec_target = int(round(float(avail) + float(tflex)))

            summary = (
                f"{product.title()} @ {period}: Forecast {_fmt_int(forecast_units)} "
                f"vs ML band [{_fmt_int(p10)} – {_fmt_int(p90)}] (p50 {_fmt_int(p50)}): "
                f"demand verdict = {final_verdict}. "
            )
            if disrupted_supplier:
                summary += (
                    f"With '{disrupted_supplier}' out: available without lost supplier {_fmt_int(avail)}, "
                    f"reallocatable flex {_fmt_float(tflex)}, covered {_fmt_float(covered)}, "
                    f"remaining gap {_fmt_float(gap)} → supply_feasible = {supply_feasible}. "
                )
                if rec_target:
                    summary += f"Recommended feasible target ≈ {_fmt_int(rec_target)}."
        except Exception:
            pass

        # ---- Policy echo (so every response shows active knobs & LP transparency)
        policy = {
            "disable_llm": DISABLE_LLM,
            "llm_timeout_s": LLM_TIMEOUT_S,
            "flex_floor_pct": float(os.getenv("FLEX_FLOOR_PCT", "0.05")),
            "flex_floor_cap": float(os.getenv("FLEX_FLOOR_CAP", "2000")),
            "leadtime_cost_weight": float(os.getenv("FLEX_COST_LEADTIME_WEIGHT", "0.4")),
            "cost_formula": "cost = (1-w)*(1/(units+1)) + w*(lead_time_days/30)  if lead_time_days present else 1/(units+1)",
            # also mirror LP transparency for quick UI use
            "lp_total_flex": (simulation or {}).get("total_flex"),
            "lp_feasible_target": (simulation or {}).get("feasible_target"),
        }

        result = {
            "historical": hist,
            "ml_forecast": ml,
            "supplier_analysis": supp_info,
            "retrieved_context_chunks": chunks,
            "event_rag": rag,
            "simulation": simulation,
            "final_verdict": final_verdict,              # demand realism
            "supply_feasible": supply_feasible,          # supply check
            "supply_feasibility_reason": supply_reason,  # reason
            "summary": summary,                          # human text
            "policy": policy,                            # active knobs
        }
        return result
