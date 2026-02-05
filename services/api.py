from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from models.spoilage.predict import predict_spoilage
from models.shelf_life.predict import predict_shelf_life
from models.demand.predict import predict_demand
from models.price.predict import predict_price

app = FastAPI(title="ML Intelligence Service")

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

@app.post("/predict/spoilage")
def spoilage(req: SpoilageRequest):
    df = pd.DataFrame([req.dict()])
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
    df = pd.DataFrame([req.dict()])
    return predict_shelf_life(df)


class DemandRequest(BaseModel):
    crop_type: str
    region: str

@app.post("/predict/demand")
def demand(req: DemandRequest):
    df = pd.read_csv("data/synthetic/demand.csv")
    return predict_demand(
        df,
        req.crop_type,
        req.region
    )


class PriceRequest(BaseModel):
    crop_type: str
    region: str
    demand_tons: float
    supply_tons: float
    avg_quality_score: float
    avg_batch_age_days: float

@app.post("/predict/price")
def price(req: PriceRequest):
    df = pd.DataFrame([req.dict()])
    return predict_price(df)
