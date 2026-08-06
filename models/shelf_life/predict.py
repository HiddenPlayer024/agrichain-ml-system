from pathlib import Path

import yaml
from joblib import load

MODEL_VERSION = "v1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with (PROJECT_ROOT / "config" / "shelf_life.yaml").open(encoding="utf-8") as f:
    BASE_LIFE = yaml.safe_load(f)

model = load(PROJECT_ROOT / "models" / "shelf_life" / "model.pkl")

def predict_shelf_life(input_data):
    crop = input_data["crop_type"].iloc[0]
    age = input_data["harvest_age_days"].iloc[0]

    if crop not in BASE_LIFE:
        supported_crops = ", ".join(sorted(BASE_LIFE))
        raise ValueError(f"Unsupported crop_type '{crop}'. Supported crops: {supported_crops}.")
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
