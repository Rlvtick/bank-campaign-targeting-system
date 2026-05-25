from pathlib import Path
from typing import Any

import sys
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import get_policy_details, score_single_customer


app = FastAPI(
    title="Bank Campaign Targeting API",
    description="Local API demo for customer campaign targeting using a trained machine learning model.",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    age: int = Field(..., ge=17, le=100)
    job: str
    marital: str
    education: str
    credit_default: str = Field(..., alias="default")
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(..., ge=1)
    pdays: int = Field(..., ge=0, le=999)
    previous: int = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_probability: float
    recommendation: str
    policy_used: str
    policy_reference_cutoff: float
    note: str


def model_to_payload(customer: CustomerInput) -> dict[str, Any]:
    if hasattr(customer, "model_dump"):
        return customer.model_dump(by_alias=True)

    return customer.dict(by_alias=True)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Bank Campaign Targeting API",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "bank-campaign-targeting-api",
    }


@app.get("/policy")
def get_policy() -> dict[str, Any]:
    try:
        return get_policy_details()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict_customer(customer: CustomerInput = Body(...)) -> PredictionResponse:
    try:
        payload = model_to_payload(customer)
        result = score_single_customer(payload)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
