# src/train.py
import pandas as pd
import json, yaml, pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]
# Load data
train = pd.read_csv("data/prepared/train.csv")
X_train = train.drop("outcome", axis=1)
y_train = train["outcome"]
# Train
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=42
)
model.fit(X_train, y_train)
# Metrics on train set
preds = model.predict(X_train)
metrics = {
    "accuracy": round(accuracy_score(y_train, preds), 4),
    "f1": round(f1_score(y_train, preds), 4),
}
# Save model & metrics
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))
json.dump(metrics, open("metrics.json", "w"))
print(f"Trained: {metrics}")