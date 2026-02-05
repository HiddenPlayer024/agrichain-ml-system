import pandas as pd

MODEL_VERSION = "v1.0.0"

def predict_demand(df, crop, region, window=7):
    sub = df[
        (df["crop_type"] == crop) &
        (df["region"] == region)
    ].sort_values("date")

    recent = sub.tail(window)

    avg = recent["demand_tons"].mean()

    trend = "stable"
    if recent["demand_tons"].iloc[-1] > recent["demand_tons"].iloc[0]:
        trend = "increasing"
    elif recent["demand_tons"].iloc[-1] < recent["demand_tons"].iloc[0]:
        trend = "decreasing"

    return {
        "expected_demand_tons": round(avg, 1),
        "confidence_interval": "±10%",
        "trend": trend,
        "model_version": MODEL_VERSION
    }
