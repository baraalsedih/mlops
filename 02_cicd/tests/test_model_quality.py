"""
Model quality gate — the CI/CD gatekeeper.

These tests run AFTER training (models/model.pkl must exist).
If they fail, the pipeline blocks deployment.

Thresholds:
  - Accuracy  >= 0.85
  - F1 (weighted) >= 0.85
  - Inference latency < 100ms for a single sample
"""

import json
import os
import time
import pytest
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"
ACCURACY_THRESHOLD = 0.85
F1_THRESHOLD = 0.85
LATENCY_THRESHOLD_MS = 100


@pytest.fixture(scope="module")
def trained_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}. Run 'make train' first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_set():
    data = load_wine()
    _, X_test, _, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_test, y_test


def test_model_accuracy(trained_model, test_set):
    X_test, y_test = test_set
    preds = trained_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= ACCURACY_THRESHOLD, (
        f"Accuracy {acc:.4f} is below threshold {ACCURACY_THRESHOLD}. "
        "Deployment blocked."
    )


def test_model_f1(trained_model, test_set):
    X_test, y_test = test_set
    preds = trained_model.predict(X_test)
    f1 = f1_score(y_test, preds, average="weighted")
    assert f1 >= F1_THRESHOLD, (
        f"F1 score {f1:.4f} is below threshold {F1_THRESHOLD}. "
        "Deployment blocked."
    )


def test_inference_latency(trained_model, test_set):
    """Single-sample inference must be fast enough for real-time serving."""
    X_test, _ = test_set
    sample = X_test[[0]]

    start = time.perf_counter()
    trained_model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < LATENCY_THRESHOLD_MS, (
        f"Inference latency {elapsed_ms:.1f}ms exceeds {LATENCY_THRESHOLD_MS}ms threshold."
    )


def test_metrics_file_exists():
    """Training must produce a metrics.json file."""
    assert os.path.exists(METRICS_PATH), (
        f"{METRICS_PATH} not found. Ensure train.py writes metrics."
    )


def test_metrics_file_content():
    """metrics.json must contain accuracy and f1_weighted keys."""
    if not os.path.exists(METRICS_PATH):
        pytest.skip("metrics.json not found")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0


def test_model_outputs_valid_classes(trained_model, test_set):
    """Model must only output valid class indices (0, 1, 2)."""
    X_test, _ = test_set
    preds = trained_model.predict(X_test)
    assert set(preds).issubset({0, 1, 2}), f"Unexpected class values: {set(preds)}"
