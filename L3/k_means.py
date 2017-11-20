import numpy as np


class KMeans:
    def __init__(self, *, k):
        self.k = k
        self.iterations = None
        self.group_centers = None
        self.groups = None
        self._data_length = None
        self._data_dimension = None

    def random_centers(self, data):
        random_without_repetition = np.random.choice(np.arange(self._data_length),
                                                     size=self.k,
                                                     replace=False)
        return data[random_without_repetition, :]

    def nearest_neighbors(self, data, group_centers):
        distances_from_centers = -2 * (data @ group_centers.T)
        distances_from_centers += np.sum(group_centers ** 2, axis=1, keepdims=True).T
        return np.argmin(distances_from_centers, axis=1)

    def centroids(self, data, partition):
        group_centers = np.zeros([self.k, self._data_dimension])
        for i in range(self.k):
            group = data[partition==i, :]
            group_centers[i] = np.sum(group, axis=0, keepdims=True) / group.shape[0]
        return group_centers

    def fit(self, data):
        self._data_length = data.shape[0]
        self._data_dimension = data.shape[1]
        group_centers = self.random_centers(data)

        partition = np.zeros(self._data_dimension)
        partition_changed = True
        self.iterations = 0
        while partition_changed:
            self.iterations += 1
            new_partition = self.nearest_neighbors(data, group_centers)
            partition_changed = not np.array_equal(partition, new_partition)
            partition = new_partition
            group_centers = self.centroids(data, partition)

        self.group_centers = group_centers
        self.groups = partition
        return self


if __name__ == "__main__":
    from matplotlib import pyplot
    data = np.random.uniform(-1, 1, [100, 2])
    print(data)
    pyplot.scatter(data[:, 0], data[:, 1])
    pyplot.show()
    data_partition = KMeans(k=3).fit(data)
    colormap = "rgb"
    print(data_partition.iterations)
    pyplot.scatter(data[:, 0], data[:, 1], c=[colormap[i] for i in data_partition.groups])
    pyplot.scatter(data_partition.group_centers[:, 0],
                   data_partition.group_centers[:, 1], c="black")
    pyplot.show()
