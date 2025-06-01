import pytest
from base import AnomalyDetectorPCA


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
