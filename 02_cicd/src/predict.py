"""
Inference module.

Usage (CLI):
    python src/predict.py --input "13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050"

Usage (programmatic):
    from src.predict import load_model, predict
    model = load_model()
    label = predict(model, [[13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]])
"""

import argparse
import joblib
import numpy as np

MODEL_PATH = "models/model.pkl"
CLASS_NAMES = ["class_0", "class_1", "class_2"]


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(model, features: list) -> str:
    """Return the predicted class name for a single sample."""
    arr = np.array(features).reshape(1, -1)
    idx = int(model.predict(arr)[0])
    return CLASS_NAMES[idx]


def predict_proba(model, features: list) -> dict:
    arr = np.array(features).reshape(1, -1)
    probs = model.predict_proba(arr)[0]
    return {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Comma-separated feature values")
    args = parser.parse_args()

    features = [float(x) for x in args.input.split(",")]
    model = load_model()

    label = predict(model, features)
    probs = predict_proba(model, features)
    print(f"Predicted class : {label}")
    print(f"Probabilities   : {probs}")
