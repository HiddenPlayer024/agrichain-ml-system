import numpy as np
import pandas as pd

np.random.seed(42)
rows = 2000

crops = ["tomato", "onion", "wheat"]
regions = ["Bangalore", "Chennai", "Hyderabad"]

base_price = {
    "tomato": 18,
    "onion": 22,
    "wheat": 26
}

data = []

for _ in range(rows):
    crop = np.random.choice(crops)
    region = np.random.choice(regions)

    demand = np.random.uniform(10, 40)
    supply = np.random.uniform(10, 45)

    quality = np.random.uniform(0.6, 1.0)
    age = np.random.uniform(1, 8)

    price = (
        base_price[crop]
        + 0.3 * (demand - supply)
        + 6 * (quality - 0.8)
        - 0.8 * age
        + np.random.normal(0, 2)
    )

    data.append({
        "crop_type": crop,
        "region": region,
        "demand_tons": round(demand, 1),
        "supply_tons": round(supply, 1),
        "avg_quality_score": round(quality, 2),
        "avg_batch_age_days": round(age, 1),
        "market_price_per_kg": round(max(5, price), 2)
    })

df = pd.DataFrame(data)
df.to_csv("data/synthetic/price.csv", index=False)

print(df.describe())
