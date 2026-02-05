from joblib import load
from utils.confidence import confidence_from_probability
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = load(MODEL_PATH)

MODEL_VERSION = "v1.0.0"

def predict_spoilage(input_data):
    prob = model.predict_proba(input_data)[0][1]

    return {
        "spoilage_probability": round(float(prob), 2),
        "risk_level": (
            "low" if prob < 0.3 else
            "medium" if prob < 0.6 else
            "high"
        ),
        "model_version": MODEL_VERSION
    }
