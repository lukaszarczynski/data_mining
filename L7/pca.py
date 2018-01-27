import numpy as np


class PCA:
    def __init__(self):
        self.data = None
        self.transformed_data = None
        self.dimension = None
        self.pca_matrix = None
        self.pca_matrix_inverse = None
        self.eigenvalues = None
        self._eigenvectors = None

    def fit_transform(self, data):
        self.data = data
        self.dimension = data.shape[1]
        data_standardized = PCA.standardize(self.data)
        covariance = np.cov(data_standardized.T)
        self.eigenvalues, self._eigenvectors = np.linalg.eig(covariance)
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self._eigenvectors = self._eigenvectors[:, idx]
        self.transformed_data = (data_standardized @ self._eigenvectors) / np.sqrt(self.eigenvalues)
        self.pca_matrix = (np.identity(self.dimension) @ self._eigenvectors) / np.sqrt(self.eigenvalues)
        self.pca_matrix_inverse = np.linalg.inv(self.pca_matrix)
        return self

    @staticmethod
    def standardize(data):
        return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
