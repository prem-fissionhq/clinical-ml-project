# src/evaluate.py
import pandas as pd
import json, pickle, os
from sklearn.metrics import (accuracy_score, f1_score,
    roc_curve)
# Load model & test data
model = pickle.load(open("models/model.pkl", "rb"))
test = pd.read_csv("data/prepared/test.csv")
X_test = test.drop("outcome", axis=1)
y_test = test["outcome"]
# Predict
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
# Metrics
eval_metrics = {
    "test_accuracy": round(accuracy_score(y_test, preds), 4),
    "test_f1": round(f1_score(y_test, preds), 4),
}
json.dump(eval_metrics, open("eval_metrics.json", "w"))
# ROC curve for DVC plots
fpr, tpr, _ = roc_curve(y_test, probs)
os.makedirs("plots", exist_ok=True)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_df.to_csv("plots/roc.csv", index=False)
print(f"Eval: {eval_metrics}")