import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from joblib import dump

df = pd.read_csv("data/synthetic/price.csv")

X = df.drop("market_price_per_kg", axis=1)
y = df["market_price_per_kg"]

cat = ["crop_type", "region"]
num = [c for c in X.columns if c not in cat]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num)
])

model = Pipeline([
    ("prep", preprocess),
    ("reg", LinearRegression())
])

model.fit(X, y)

dump(model, "models/price/model.pkl")
print("✅ Price recommendation model trained")
