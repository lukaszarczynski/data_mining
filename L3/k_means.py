import numpy as np

class KMeans:
    def __init__(self, *, k):
        self.k = k
        self.total_iterations = 0
        self.iterations = None
        self.group_centers = None
        self.groups = None
        self._data_length = None
        self._data_dimension = None
        self.partition_changed = None

    def _get_nth_group(self, data, partition=None, *, n):
        if partition is None:
            partition = self.groups
        return data[partition == n, :]

    def random_centers(self, data):
        random_without_repetition = np.random.choice(np.arange(self._data_length),
                                                     size=self.k,
                                                     replace=False)
        return data[random_without_repetition, :]

    def nearest_neighbors(self, data, group_centers):
        distances_from_centers = -2 * (data @ group_centers.T)
        distances_from_centers += np.sum(group_centers ** 2, axis=1, keepdims=True).T
        return np.argmin(distances_from_centers, axis=1)

    def euclidean_distance(self, data, vector):
        data -= vector
        data = data ** 2
        length = np.sqrt(np.sum(data, axis=1, keepdims=True))
        return length

    def centroids(self, data, partition):
        group_centers = np.zeros([self.k, self._data_dimension])
        for i in range(self.k):
            group = self._get_nth_group(data, partition, n=i)
            if len(group) == 0:
                group_centers[i] = np.random.choice(np.arange(self._data_length), 1)
            else:
                group_centers[i] = np.sum(group, axis=0, keepdims=True) / group.shape[0]
        return group_centers

    def fit(self, data, *, max_iterations=300):
        self._data_length = data.shape[0]
        self._data_dimension = data.shape[1]
        if self.group_centers is None:
            self.group_centers = self.random_centers(data)
            self.groups = np.zeros(self._data_dimension)
            self.partition_changed = True
        self.iterations = 0
        while self.partition_changed and self.iterations < max_iterations:
            new_partition = self.nearest_neighbors(data, self.group_centers)
            self.partition_changed = not np.array_equal(self.groups, new_partition)
            self.groups = new_partition
            self.group_centers = self.centroids(data, self.groups)
            self.iterations += 1
            self.total_iterations += 1
        return self

    def average_distances_to_center(self, data):
        average_distances = np.zeros(self.k)
        if self.group_centers is None and data is not None:
            self.fit(data)
        for group_idx, center in enumerate(self.group_centers):
            group = self._get_nth_group(data, n=group_idx)
            distances = self.euclidean_distance(group, center)
            average_distances[group_idx] = np.average(distances)
        return average_distances

    def center_to_center_average_distance(self):
        if self.group_centers is None:
            raise Exception("Run KMeans::fit(data) first")
        average_distances = np.zeros(self.k)
        for group_idx, center in enumerate(self.group_centers):
            distances = self.euclidean_distance(self.group_centers, center)
            average_distances[group_idx] = np.sum(distances) / (self.k - 1)
        return average_distances


if __name__ == "__main__":
    from matplotlib import pyplot

    data = np.random.uniform(-1, 1, [100, 2])
    print(data)
    pyplot.scatter(data[:, 0], data[:, 1])
    pyplot.show()
    data_partition = KMeans(k=3)
    for _ in range(10):
        data_partition.fit(data)
        colormap = "rgb"
        print(data_partition.iterations)
        pyplot.scatter(data[:, 0], data[:, 1], c=[colormap[i] for i in data_partition.groups])
        pyplot.scatter(data_partition.group_centers[:, 0],
                       data_partition.group_centers[:, 1], c="black")
        pyplot.show()
    print(data_partition.average_distances_to_center(data))
    print(data_partition.center_to_center_average_distance())
