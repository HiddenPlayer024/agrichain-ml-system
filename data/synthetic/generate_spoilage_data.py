import pandas as pd
import numpy as np

np.random.seed(42)
rows = 2000

data = {
    "crop_type": np.random.choice(["tomato", "onion", "wheat"], rows),
    "harvest_age_days": np.random.randint(1, 10, rows),
    "avg_temperature": np.random.uniform(10, 40, rows),
    "avg_humidity": np.random.uniform(30, 90, rows),
    "cold_chain": np.random.choice([0, 1], rows, p=[0.4, 0.6]),
    "transport_time_hours": np.random.uniform(1, 48, rows),
    "storage_days": np.random.uniform(0, 7, rows),
    "handling_type": np.random.choice(["manual", "semi", "automated"], rows),
    "historical_spoilage_rate": np.random.uniform(0.05, 0.3, rows),
}

df = pd.DataFrame(data)

risk = (
    0.02 * df["harvest_age_days"]
    + 0.015 * df["transport_time_hours"]
    + 0.03 * df["storage_days"]
    + 0.5 * df["historical_spoilage_rate"]
    + (df["cold_chain"] == 0) * 0.25
)

threshold = np.random.uniform(0.75, 1.1, rows)
df["spoilage"] = (risk + np.random.normal(0, 0.1, rows)) > threshold
df["spoilage"] = df["spoilage"].astype(int)

print(df["spoilage"].value_counts(normalize=True))

df.to_csv("data/synthetic/spoilage.csv", index=False)
