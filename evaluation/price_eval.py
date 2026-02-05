import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

PRICE_VOLATILITY_BASELINE = 1.5

df = pd.read_csv("data/synthetic/price.csv")

X = df.drop("market_price_per_kg", axis=1)
y = df["market_price_per_kg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = load("models/price/model.pkl")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
within_band = (abs(y_test - pred) <= PRICE_VOLATILITY_BASELINE).mean()

print("=== Price Model Evaluation ===")
print(f"MAE (₹/kg)            : {mae:.2f}")
print(f"Within ±{PRICE_VOLATILITY_BASELINE} band: {within_band*100:.1f}%")
