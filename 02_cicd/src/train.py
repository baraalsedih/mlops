"""
Training script — produces a versioned model artifact.

Usage:
    python src/train.py
    python src/train.py --n_estimators 200 --max_depth 8

Outputs:
    models/model.pkl  — trained model
    models/metrics.json — evaluation metrics (used by CI quality gate)
"""

import argparse
import json
import os
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=args.random_state,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "f1_weighted": round(f1_score(y_test, preds, average="weighted"), 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
}

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Training complete: accuracy={metrics['accuracy']}, f1={metrics['f1_weighted']}")
print("Saved: models/model.pkl, models/metrics.json")
