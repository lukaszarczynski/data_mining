import numpy as np
import math


def euclidean_distance(data, vector):
    if len(data) == 0:
        return math.inf
    data_ = data - vector
    data_ = data_ ** 2
    length = np.sqrt(np.sum(data_, axis=1))
    return length


def mean_cluster_distance(points):
    distances_sum = 0
    points_number = points.shape[0]
    for point in points:
        distances_sum += np.sum(euclidean_distance(points, point))
    return distances_sum / (points_number * (points_number - 1))


def centroid(points):
    return np.sum(points, axis=0) / points.shape[0]


def dunn_index(data, labels):
    groups_number = np.unique(labels).size
    max_mean_distance = 0
    min_centroid_distance = math.inf
    centroids = []

    for k in range(groups_number):
        mask = (labels == k)
        cluster_points = data[mask]
        if len(cluster_points) > 1:
            max_mean_distance = max(max_mean_distance, mean_cluster_distance(cluster_points))
            centroids.append(centroid(cluster_points))

    for k in range(groups_number-1):
        mask = (labels == k)
        cluster_points = data[mask]
        if len(cluster_points) > 1:
            min_centroid_distance = min(min_centroid_distance,
                                        np.min(euclidean_distance(np.array(centroids[k+1:]), centroids[k])))

    return min_centroid_distance / max_mean_distance


if __name__ == "__main__":
    test_data = np.array([[3.25685617, 3.63843017],
                          [4.84568309, 1.18329245],
                          [3.23170205, 2.91937533],
                          [5.46490842, 0.60754080],
                          [4.52515990, 0.55875938],
                          [0.71543742, 1.21950141],
                          [5.45439583, 1.46450104],
                          [2.48937667, 2.48866518],
                          [0.31109417, 0.71295525],
                          [1.00011775, 0.39126122],
                          [2.51824272, 3.23213796],
                          [1.51435139, 0.98189949],
                          [1.15382773, 1.73467109],
                          [3.23160577, 2.32974692],
                          [3.91106355, 2.87310064],
                          [4.28310364, 1.54158741]])
    print(mean_cluster_distance(test_data))
    print(centroid(test_data))
    print(dunn_index(test_data, np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1])))

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    from sklearn.metrics.cluster.unsupervised import silhouette_score
    from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN

    import matplotlib.colors as colors
    from itertools import cycle


    def plot_clustering(data, plt_labels, marker='o', show=True):
        matplotlib.rcParams['figure.figsize'] = [6., 4.]

        plt_colors = sorted(colors.cnames.keys(), key=lambda x: len(x))
        plt_colors.remove("cyan")
        plt_colors = cycle(plt_colors)

        plt_k = np.unique(plt_labels).size

        for k in range(plt_k):
            color = next(plt_colors)
            mask = (plt_labels == k)
            plt.plot(data[mask, 0], data[mask, 1], f'k.', markerfacecolor=color, marker=marker)

        print({"Silhouette": silhouette_score(data, plt_labels),
               "Dunn index": dunn_index(data, plt_labels)})

        if show:
            plt.show()


    from sklearn import datasets

    centers_ = [[1, 1], [3, 3], [5, 1]]
    blobs_data, labels = datasets.make_blobs(n_samples=3000, n_features=2, centers=centers_, cluster_std=0.5)

    birch = Birch(threshold=0.5, n_clusters=None)
    birch.fit(blobs_data)

    plot_clustering(blobs_data, birch.labels_)

    birch = Birch(threshold=0.75, n_clusters=None)
    birch.fit(blobs_data)

    plot_clustering(blobs_data, birch.labels_)

    dbscan = DBSCAN(eps=0.25, min_samples=25)
    dbscan.fit(blobs_data)

    plot_clustering(blobs_data, dbscan.labels_)
