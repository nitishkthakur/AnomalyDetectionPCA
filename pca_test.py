import pytest
from base import AnomalyDetectorPCA
import numpy as np
import pandas as pd


@pytest.fixture
def sample_normal_data():
    """
    Creates a dataset that represents 'normal' behavior.
    Think of this as data from a well-functioning system with slight variations.
    """
    np.random.seed(42)  # For reproducible tests
    n_samples = 100

    # Create correlated features (like temperature and pressure might be)
    feature1 = np.random.normal(10, 2, n_samples)  # Mean=10, std=2
    feature2 = feature1 * 0.8 + np.random.normal(
        0, 0.5, n_samples
    )  # Correlated with feature1
    feature3 = np.random.normal(5, 1, n_samples)  # Independent feature

    return pd.DataFrame({"sensor1": feature1, "sensor2": feature2, "sensor3": feature3})


class TestAnomalyDetectorBasics:
    """
    These tests ensure the basic building blocks work correctly.
    Think of them as testing individual Lego pieces before building the castle.
    """

    def test_initialization(self):
        """Test that our detector initializes with correct default values."""
        detector = AnomalyDetectorPCA()

        assert detector.n_components == 2, "Default components should be 2"
        assert detector.threshold == 3, "Default threshold should be 3"
        assert detector.pca is None, "PCA should be None before fitting"

    def test_initialization_with_custom_params(self):
        """Test that custom parameters are set correctly."""
        detector = AnomalyDetectorPCA(n_components=5, threshold=2.0)

        assert detector.n_components == 5
        assert detector.threshold == 2.0

    def test_fit_creates_pipeline(self, sample_normal_data):
        """Test that fitting creates and trains the PCA pipeline."""
        detector = AnomalyDetectorPCA(n_components=2)
        detector.fit(sample_normal_data)

        # After fitting, these should exist
        assert detector.pca is not None, "PCA pipeline should be created"
        assert hasattr(detector, "data_columns"), "Column names should be stored"
        assert list(detector.data_columns) == list(sample_normal_data.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
