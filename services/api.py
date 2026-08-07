from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from models.spoilage.predict import predict_spoilage
from models.shelf_life.predict import predict_shelf_life
from models.demand.predict import predict_demand
from models.price.predict import predict_price

app = FastAPI(title="ML Intelligence Service")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMAND_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "demand.csv"


def request_frame(request: BaseModel) -> pd.DataFrame:
    """Convert a Pydantic request to the single-row model input frame."""
    return pd.DataFrame([request.model_dump()])

class SpoilageRequest(BaseModel):
    crop_type: str
    harvest_age_days: int
    avg_temperature: float
    avg_humidity: float
    cold_chain: int
    transport_time_hours: float
    storage_days: float
    handling_type: str
    historical_spoilage_rate: float

@app.get("/")
def root():
    return {"status": "ML API is running"}

@app.post("/predict/spoilage")
def spoilage(req: SpoilageRequest):
    df = request_frame(req)
    return predict_spoilage(df)


class ShelfLifeRequest(BaseModel):
    crop_type: str
    harvest_age_days: int
    avg_temperature: float
    avg_humidity: float
    cold_chain: int
    handling_type: str
    storage_type: str

@app.post("/predict/shelf-life")
def shelf_life(req: ShelfLifeRequest):
    try:
        return predict_shelf_life(request_frame(req))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as e:
        print(e)
        raise


class DemandRequest(BaseModel):
    crop_type: str
    region: str

@app.post("/predict/demand")
def demand(req: DemandRequest):
    df = pd.read_csv(DEMAND_DATA_PATH)
    try:
        return predict_demand(df, req.crop_type, req.region)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class PriceRequest(BaseModel):
    crop_type: str
    region: str
    demand_tons: float
    supply_tons: float
    avg_quality_score: float
    avg_batch_age_days: float

@app.post("/predict/price")
def price(req: PriceRequest):
    df = request_frame(req)
    return predict_price(df)
