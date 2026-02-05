import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/synthetic/shelf_life.csv")

X = df.drop("decay_factor", axis=1)
y = df["decay_factor"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = load("models/shelf_life/model.pkl")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("=== Shelf-Life Model Evaluation ===")
print(f"MAE (decay factor): {mae:.3f}")
print(f"Prediction range : {pred.min():.2f} – {pred.max():.2f}")
