MODEL_VERSION = "v1.0.0"

def predict_demand(df, crop, region, window=7):
    """Forecast demand from recent records for a crop and region.

    Matching is case-insensitive so API clients do not need to reproduce the
    capitalization used in the source CSV.
    """
    crop = crop.strip()
    region = region.strip()
    sub = df[
        (df["crop_type"].str.casefold() == crop.casefold()) &
        (df["region"].str.casefold() == region.casefold())
    ].sort_values("date")

    recent = sub.tail(window)
    if recent.empty:
        supported_crops = ", ".join(sorted(df["crop_type"].dropna().unique()))
        supported_regions = ", ".join(sorted(df["region"].dropna().unique()))
        raise ValueError(
            f"No demand data available for crop_type '{crop}' in region '{region}'. "
            f"Supported crops: {supported_crops}. Supported regions: {supported_regions}."
        )

    avg = recent["demand_tons"].mean()

    trend = "stable"
    if recent["demand_tons"].iloc[-1] > recent["demand_tons"].iloc[0]:
        trend = "increasing"
    elif recent["demand_tons"].iloc[-1] < recent["demand_tons"].iloc[0]:
        trend = "decreasing"

    return {
        "expected_demand_tons": round(float(avg), 1),
        "confidence_interval": "±10%",
        "trend": trend,
        "model_version": MODEL_VERSION
    }
