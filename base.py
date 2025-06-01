import numpy as np  # noqa: F401
from sklearn.decomposition import PCA  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
from sklearn import preprocessing, decomposition, pipeline
import pandas as pd


class AnomalyDetectorPCA:
    def __init__(self, n_components: int = 2, threshold=3):
        self.n_components = n_components
        self.pca = None
        self.threshold = threshold

    def fit(self, X: pd.DataFrame):
        self.data_columns = X.columns
        self.pca = pipeline.make_pipeline(
            preprocessing.StandardScaler(),
            decomposition.PCA(n_components=self.n_components),
        ).fit(X)

    def transform(self, X: pd.DataFrame):
        return self.pca.transform(X)

    def inverse_transform(self, X_transformed):
        return pd.DataFrame(
            self.pca.inverse_transform(X_transformed), columns=self.data_columns
        )

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        # transform  the  existing data
        transformed_data = pd.DataFrame(
            self.transform(X),
            columns=["pc: " + str(i) for i in range(1, self.n_components + 1)],
        )

        # Now, Inverse transform the data
        inverse_transformed = self.inverse_transform(transformed_data)

        # Calculate the reconstruction error
        self.reconstruction_error_explainable = (X - inverse_transformed) ** 2
        reconstruction_error = np.mean(self.reconstruction_error_explainable, axis=1)

        # prepare the return df and reconstruction error dfs
        X["reconstruction_error"] = reconstruction_error
        self.reconstruction_error_explainable["reconstruction_error"] = (
            reconstruction_error
        )

        # Sort the DataFrames by reconstruction error
        X = X.sort_values(by="reconstruction_error", ascending=False)
        self.reconstruction_error_explainable = (
            self.reconstruction_error_explainable.sort_values(
                by="reconstruction_error", ascending=False
            )
        )

        # Create the anomaly column based on the threshold
        # Apply tukey's method to determine the threshold
        q1 = X["reconstruction_error"].quantile(0.25)
        q3 = X["reconstruction_error"].quantile(0.75)
        iqr = q3 - q1
        threshold_value = q3 + self.threshold * iqr

        # Mark anomalies
        X["anomaly"] = X["reconstruction_error"] > threshold_value
        X["Index"] = np.arange(len(X))
        self.reconstruction_error_explainable["Index"] = np.arange(
            len(self.reconstruction_error_explainable)
        )

        return X

    def explain(self, index: int):
        explainer = self.reconstruction_error_explainable.loc[
            self.reconstruction_error_explainable["Index"] == index, :
        ].drop(columns=["reconstruction_error", "Index"])
        print(explainer)
        plt.figure(figsize=(25, 6))
        plt.bar(x=explainer.columns, height=explainer.values.flatten(), color="black")
        plt.ylabel("Degree of Contribution to Reconstruction Error")
        return plt
