import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

def predict_spoilage(input_df):
    prob = model.predict_proba(input_df)[0][1]
    return {
        "spoilage_probability": float(prob),
        "risk_level": (
            "low" if prob < 0.3 else
            "medium" if prob < 0.6 else
            "high"
        )
    }
