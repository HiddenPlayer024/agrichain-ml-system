from joblib import load
from utils.confidence import confidence_from_probability
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "spoilage" / "model.pkl"

model = load(MODEL_PATH)

MODEL_VERSION = "v1.0.0"

def predict_spoilage(input_data):
    prob = model.predict_proba(input_data)[0][1]

    return {
        "spoilage_probability": round(float(prob), 2),
        "confidence": round(confidence_from_probability(prob), 2),
        "model_version": MODEL_VERSION
    }
