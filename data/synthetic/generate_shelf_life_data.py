import numpy as np
import pandas as pd

np.random.seed(42)
rows = 2000

data = {
    "avg_temperature": np.random.uniform(10, 40, rows),
    "avg_humidity": np.random.uniform(30, 90, rows),
    "cold_chain": np.random.choice([0, 1], rows, p=[0.4, 0.6]),
    "handling_type": np.random.choice(
        ["manual", "semi", "automated"], rows
    ),
    "storage_type": np.random.choice(
        ["open", "covered", "cold"], rows
    )
}

df = pd.DataFrame(data)

decay = (
    1.0
    - 0.015 * (df["avg_temperature"] - 20).clip(lower=0)
    - 0.01 * (df["avg_humidity"] - 60).clip(lower=0)
    - (df["cold_chain"] == 0) * 0.15
    - (df["handling_type"] == "manual") * 0.10
    - (df["storage_type"] == "open") * 0.12
)

df["decay_factor"] = decay.clip(0.3, 1.0)

df.to_csv("data/synthetic/shelf_life.csv", index=False)
print(df["decay_factor"].describe())
