import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

# Load data
df = pd.read_csv("data/synthetic/spoilage.csv")

X = df.drop("spoilage", axis=1)
y = df["spoilage"]

# Train-test split (never evaluate on training data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
model = load("models/spoilage/model.pkl")

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# Metrics
roc_auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

print("=== Spoilage Model Evaluation ===")
print(f"ROC-AUC Score       : {roc_auc:.3f}")
print(f"Brier Score         : {brier:.3f}")
print(f"Mean Predicted Risk : {probs.mean():.3f}")
