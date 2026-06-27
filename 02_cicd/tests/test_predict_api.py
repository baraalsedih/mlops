"""
Smoke tests for the predict module interface.

These test the predict() and predict_proba() functions directly
without requiring a trained model on disk.
"""

import os
import pytest
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure src is importable when running from 02_cicd/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predict, predict_proba, load_model, CLASS_NAMES


@pytest.fixture(scope="module")
def dummy_model(tmp_path_factory):
    """Train a tiny model and save it for testing."""
    tmp = tmp_path_factory.mktemp("models")
    model_path = str(tmp / "model.pkl")

    data = load_wine()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf, model_path


SAMPLE_FEATURES = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]


def test_predict_returns_string(dummy_model):
    model, _ = dummy_model
    result = predict(model, SAMPLE_FEATURES)
    assert isinstance(result, str)
    assert result in CLASS_NAMES


def test_predict_proba_returns_dict(dummy_model):
    model, _ = dummy_model
    probs = predict_proba(model, SAMPLE_FEATURES)
    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(CLASS_NAMES)


def test_predict_proba_sums_to_one(dummy_model):
    model, _ = dummy_model
    probs = predict_proba(model, SAMPLE_FEATURES)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_predict_proba_all_non_negative(dummy_model):
    model, _ = dummy_model
    probs = predict_proba(model, SAMPLE_FEATURES)
    assert all(v >= 0 for v in probs.values())


def test_predict_wrong_feature_count_raises(dummy_model):
    """Passing wrong number of features should raise, not silently mispredict."""
    model, _ = dummy_model
    with pytest.raises(Exception):
        predict(model, [1.0, 2.0])  # only 2 features instead of 13


def test_load_model(dummy_model):
    _, model_path = dummy_model
    loaded = load_model(path=model_path)
    assert loaded is not None
    result = predict(loaded, SAMPLE_FEATURES)
    assert result in CLASS_NAMES
