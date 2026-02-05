import pandas as pd
from joblib import load

MODEL_VERSION = "v1.0.0"

model = load("models/price/model.pkl")

# Conservative baseline volatility (₹/kg)
PRICE_VOLATILITY_BASELINE = 1.5

def predict_price(input_data):
    price = model.predict(input_data)[0]

    return {
        "expected_price_per_kg": round(float(price), 2),
        "price_volatility": PRICE_VOLATILITY_BASELINE,
        "model_version": MODEL_VERSION
    }
