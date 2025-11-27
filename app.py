# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from validation_agent import ValidationAgent

app = FastAPI(title="Event-aware Forecast Validator")
AGENT = ValidationAgent(db_path="data/sales.duckdb")

class ValidateReq(BaseModel):
    product: str
    period: str     # "YYYY-MM"
    forecast_units: int
    disrupted_supplier: str | None = None

@app.post("/validate")
def validate(req: ValidateReq):
    return AGENT.validate_forecast(
        product=req.product,
        period=req.period,
        forecast_units=req.forecast_units,
        disrupted_supplier=req.disrupted_supplier
    )
