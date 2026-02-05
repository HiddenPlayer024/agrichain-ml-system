import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2025-01-01", "2025-12-31")
crops = ["tomato", "onion", "wheat"]
regions = ["Bangalore", "Chennai", "Hyderabad"]

rows = []

for crop in crops:
    for region in regions:
        base = {
            "tomato": 18,
            "onion": 25,
            "wheat": 40
        }[crop]

        for d in dates:
            seasonal = 1.2 if d.month in [6,7,8] else 1.0
            noise = np.random.normal(0, 3)

            demand = max(
                1,
                base * seasonal + noise
            )

            rows.append({
                "date": d,
                "crop_type": crop,
                "region": region,
                "demand_tons": round(demand, 1)
            })

df = pd.DataFrame(rows)
df.to_csv("data/synthetic/demand.csv", index=False)
print("✅ Synthetic demand data generated")
