"""
AFTER: Same training logic — now fully tracked with W&B.

What W&B adds:
- Every run is logged with its hyperparameters, metrics, and system info
- Confusion matrix and feature importances visualised in the dashboard
- Trained model saved as a versioned Artifact
- Runs are comparable side-by-side in the W&B UI

Run:
    python train_with_wandb.py
    python train_with_wandb.py --n_estimators 200 --max_depth 10   # different config
Then compare runs at https://wandb.ai
"""

import argparse
import joblib
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init W&B run
# ---------------------------------------------------------------------------
run = wandb.init(
    project="mlops-wine-classifier",
    config=vars(args),          # hyperparameters logged automatically
    tags=["random-forest", "wine"],
)
cfg = wandb.config

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.test_size, random_state=cfg.random_state
)

wandb.log({
    "data/train_size": len(X_train),
    "data/test_size": len(X_test),
    "data/n_features": X.shape[1],
    "data/n_classes": len(np.unique(y)),
})

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=cfg.n_estimators,
    max_depth=cfg.max_depth,
    random_state=cfg.random_state,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
print(classification_report(y_test, preds, target_names=data.target_names))

wandb.log({"eval/accuracy": acc, "eval/f1_weighted": f1})

# ---------------------------------------------------------------------------
# Confusion matrix as W&B plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks(range(len(data.target_names)))
ax.set_yticks(range(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=45, ha="right")
ax.set_yticklabels(data.target_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
wandb.log({"eval/confusion_matrix": wandb.Image(fig)})
plt.close()

# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------
importances = model.feature_importances_
fig2, ax2 = plt.subplots(figsize=(8, 5))
indices = np.argsort(importances)[::-1]
ax2.bar(range(len(importances)), importances[indices])
ax2.set_xticks(range(len(importances)))
ax2.set_xticklabels(
    [data.feature_names[i] for i in indices], rotation=45, ha="right"
)
ax2.set_title("Feature Importances")
plt.tight_layout()
wandb.log({"eval/feature_importances": wandb.Image(fig2)})
plt.close()

# ---------------------------------------------------------------------------
# Save model as W&B Artifact
# ---------------------------------------------------------------------------
model_path = "model.pkl"
joblib.dump(model, model_path)

artifact = wandb.Artifact(
    name="wine-classifier",
    type="model",
    description="RandomForest trained on the Wine dataset",
    metadata={"accuracy": acc, "f1": f1, **vars(cfg)},
)
artifact.add_file(model_path)
run.log_artifact(artifact)

wandb.finish()
print("\nRun complete — view your experiment at https://wandb.ai")
