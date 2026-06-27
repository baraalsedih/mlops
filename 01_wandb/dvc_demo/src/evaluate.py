"""Stage 3 — evaluate: compute metrics and write to metrics/scores.json."""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

model = joblib.load("models/model.pkl")
preds = model.predict(X_test)

scores = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "f1_weighted": round(f1_score(y_test, preds, average="weighted"), 4),
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=2)

print("Scores:", scores)
