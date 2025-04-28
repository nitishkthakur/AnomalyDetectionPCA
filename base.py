import numpy as np
from sklearn.decomposition import PCA

class PCAAnomalyDetector:
    def __init__(self, n_components=None, threshold=None):
        """
        n_components: Number of PCA components (None = all components)
        threshold: Reconstruction error threshold to flag anomalies
                   (None = auto-set during fitting)
        """
        self.n_components = n_components
        self.threshold = threshold
        self.pca = PCA(n_components=self.n_components)
        self.fitted = False

    def fit(self, X):
        """Fit the PCA model and set threshold if not provided."""
        self.pca.fit(X)
        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        if self.threshold is None:
            # Set threshold as mean + 3*std (you can tweak this rule)
            self.threshold = np.mean(errors) + 3 * np.std(errors)
        
        self.fitted = True

    def score_samples(self, X):
        """Compute reconstruction errors (anomaly scores)."""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        return errors

    def predict(self, X):
        """Return 1 if anomaly, 0 if normal."""
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

# Example usage:
if __name__ == "__main__":
    # Generate dummy data
    X_normal = np.random.normal(0, 1, (100, 5))
    X_anomalies = np.random.normal(5, 1, (5, 5))
    X_test = np.vstack([X_normal, X_anomalies])

    # Initialize and fit
    detector = PCAAnomalyDetector(n_components=2)
    detector.fit(X_normal)  # Only fit on normal data

    # Predict
    predictions = detector.predict(X_test)
    scores = detector.score_samples(X_test)

    print("Predictions:", predictions)
    print("Scores:", scores)
