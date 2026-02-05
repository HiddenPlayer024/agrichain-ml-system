import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Load data
df = pd.read_csv("data/synthetic/spoilage.csv")

X = df.drop("spoilage", axis=1)
y = df["spoilage"]

# Feature groups
cat = ["crop_type", "handling_type"]
num = [c for c in X.columns if c not in cat]

# Preprocessing
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num)
])

# Model with proper regularization
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=0.3,                  # 🔑 stronger regularization
        class_weight="balanced",# 🔑 fixes label skew
        solver="lbfgs"
    ))
])

# Train
model.fit(X, y)

# Save
dump(model, "models/spoilage/model.pkl")
print("✅ Spoilage model v1.1.0 trained and saved")
