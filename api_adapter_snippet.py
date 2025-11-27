# api_adapter_snippet.py  (example snippets for FastAPI)
from fastapi import FastAPI, Body, Query
from typing import Optional
import ml_forecast            # your original module (has _latest_context & predict_product_quantiles)
import ml_forecast_calibrated as calib_mod  # the new module we added

app = FastAPI()

def build_feature_row_from_request(product: str, year: int, month: int) -> dict:
    """
    Uses your existing _latest_context(product,year,month) function to build the exact features.
    If _latest_context returns None, you may want to raise or fall back to zeros.
    """
    ctx = None
    try:
        ctx = ml_forecast._latest_context(product.lower(), year, month)
    except Exception:
        ctx = None
    if ctx is None or ctx.empty:
        raise ValueError(f"No feature context for {product} {year}-{month:02d}")
    # ctx is a pandas.DataFrame with correct column names; convert to dict
    return ctx.iloc[0].to_dict()

@app.post("/validate")
def validate_scenario(body: dict = Body(...), calibrate: Optional[bool] = Query(False)):
    """
    Example expected body:
    { "product": "wine", "period": "2020-01", "forecast_units": 150000, "disrupted_supplier":"RELIABLE CHURCHILL LLLP" }
    If calibrate=True -> call calibrated predictor
    """
    product = body.get("product")
    period = body.get("period")  # "YYYY-MM"
    year, month = map(int, period.split("-"))
    feature_row = build_feature_row_from_request(product, year, month)

    if calibrate:
        # call calibrated predictor (uses evaluations/models_split.pkl)
        result = calib_mod.ml_forecast_calibrated_row(feature_row)
    else:
        # call legacy wrapper that expects (product,year,month)
        result = ml_forecast.predict_product_quantiles(product, year, month)

    # Merge result into final response skeleton, and continue rest of pipeline
    response = {"ml_forecast": result}
    # (rest of validation pipeline: supplier_analysis, RAG, LP... )
    return response
