"""
Data schema validation tests.

Ensures that the input data matches the expected schema before training.
In production, run these tests against every new data batch.
"""

import numpy as np
import pytest
from sklearn.datasets import load_wine


@pytest.fixture(scope="module")
def wine_data():
    data = load_wine()
    return data


def test_feature_count(wine_data):
    """The dataset must have exactly 13 features."""
    assert wine_data.data.shape[1] == 13


def test_sample_count(wine_data):
    """Sanity check: dataset has at least 100 samples."""
    assert wine_data.data.shape[0] >= 100


def test_no_missing_values(wine_data):
    """No NaN values anywhere in the feature matrix."""
    assert not np.isnan(wine_data.data).any(), "Found NaN values in features"


def test_no_infinite_values(wine_data):
    """No infinite values in features."""
    assert np.isfinite(wine_data.data).all(), "Found non-finite values in features"


def test_target_classes(wine_data):
    """Target must have exactly 3 classes (0, 1, 2)."""
    unique_classes = np.unique(wine_data.target)
    assert len(unique_classes) == 3
    assert set(unique_classes) == {0, 1, 2}


def test_feature_value_ranges(wine_data):
    """All feature values must be non-negative (domain constraint for wine data)."""
    assert (wine_data.data >= 0).all(), "Negative feature values detected"


def test_expected_feature_names(wine_data):
    expected = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "color_intensity", "hue",
        "od280/od315_of_diluted_wines", "proline",
    ]
    assert list(wine_data.feature_names) == expected
