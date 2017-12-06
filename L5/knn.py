import numpy as np
from scipy.stats import mode


class KNN:
    def __init__(self, *, k):
        self.k = k
        self.training_data = None
        self.target_values = None

    def fit(self, training_data, target_values):
        self.training_data = training_data
        self.target_values = target_values
        return self

    def predict(self, test_data):
        neighbours_indices = self.nearest_neighbours(test_data)
        neighbours_labels = self.target_values[neighbours_indices]
        best_match = mode(neighbours_labels, axis=1)[0]
        return best_match.T[0]

    def score(self, test_data, labels):
        predicted_labels = self.predict(test_data)
        return sum(labels == predicted_labels) / len(labels)

    def euclidean_distance(self, matrix_1, matrix_2):
        sum_squares_1 = np.sum(matrix_1 ** 2, axis=1)[:, np.newaxis]  # transposed
        sum_squares_2 = np.sum(matrix_2 ** 2, axis=1)
        distances = -2 * matrix_1 @ matrix_2.T
        distances += sum_squares_1 + sum_squares_2
        return distances

    def nearest_neighbours(self, test_data):
        distances = self.euclidean_distance(self.training_data, test_data)
        neighbours_indices = np.argpartition(distances, self.k, axis=0)
        neighbours_indices = neighbours_indices[:self.k]
        return neighbours_indices.T


if __name__ == "__main__":
    matrix_1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    matrix_2 = np.array([[2, 2], [3, 3], [2, 4]])
    knn = KNN(k=1)
    print(knn.euclidean_distance(matrix_1, matrix_1))
    print(matrix_1, matrix_2)
    knn.fit(matrix_1, ...)
    print(knn.nearest_neighbours(matrix_2))
    knn2 = KNN(k=2).fit(matrix_1, np.array([0, 0, 1, 1]))
    print(knn2.nearest_neighbours(matrix_2))
    print(knn2.predict(matrix_2))
    print(knn2.score(matrix_2, np.array([1, 1, 1])))
