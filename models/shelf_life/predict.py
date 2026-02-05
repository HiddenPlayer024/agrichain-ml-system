import yaml
import pandas as pd
from joblib import load

MODEL_VERSION = "v1.0.0"

with open("config/shelf_life.yaml") as f:
    BASE_LIFE = yaml.safe_load(f)

model = load("models/shelf_life/model.pkl")

def predict_shelf_life(input_data):
    crop = input_data["crop_type"].iloc[0]
    age = input_data["harvest_age_days"].iloc[0]

    base_days = BASE_LIFE[crop]

    decay = model.predict(input_data.drop(
        ["crop_type", "harvest_age_days"], axis=1
    ))[0]

    expected = base_days * decay
    remaining = max(0, round(expected - age))

    return {
        "expected_shelf_life_days": round(expected),
        "remaining_days": remaining,
        "decay_factor": round(float(decay), 2),
        "model_version": MODEL_VERSION
    }
