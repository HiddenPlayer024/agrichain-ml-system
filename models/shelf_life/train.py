import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

df = pd.read_csv("data/synthetic/shelf_life.csv")

X = df.drop("decay_factor", axis=1)
y = df["decay_factor"]

cat = ["handling_type", "storage_type"]
num = [c for c in X.columns if c not in cat]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num)
])

model = Pipeline([
    ("prep", preprocess),
    ("reg", RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ))
])

model.fit(X, y)

dump(model, "models/shelf_life/model.pkl")
print("✅ Shelf-life decay model trained")
