"""Stage 2 — train: fit model on training data."""

import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml"))["train"]

train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=42,
)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")
