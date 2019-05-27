import numpy as np

from L2.zad4 import polygon_point_clouds


def vector_length(vectors):
    vectors = vectors.T**2
    length = np.sqrt(np.ones((1, vectors.shape[0])) @ vectors)
    return length


def euclidean_distance(vector_list, vector):
    vector_list = vector_list - vector
    return vector_length(vector_list)


def nearest_neighbour(points, centres):
    distances = np.zeros((centres.shape[0], points.shape[0]))
    for centre_idx, centre in enumerate(centres):
        distances[centre_idx] = euclidean_distance(points, np.array(centre))
    nearest = np.argmin(distances, axis=0)
    return nearest


def check_polygon_point_cloud_distances(vertex_number, edge_length, points_number=1000):
    points_clouds = polygon_point_clouds(vertex_number, edge_length, points_number)
    centres = np.array([cloud[0] for cloud in points_clouds])
    correct_points = np.array([]).reshape(0, 2)
    incorrect_points = np.array([]).reshape(0, 2)
    for cloud_idx, points_cloud in enumerate(points_clouds):
        center, points = points_cloud
        nn = nearest_neighbour(points, centres)
        correct = (nn == cloud_idx)
        incorrect = (nn != cloud_idx)
        correct_points = np.append(correct_points, points[correct], axis=0)
        incorrect_points = np.append(incorrect_points, (points[incorrect]), axis=0)
    return correct_points, incorrect_points


if __name__ == "__main__":
    correct_points, incorrect_points = check_polygon_point_cloud_distances(4, 5)
    print(incorrect_points)
    print(len(incorrect_points))
    print(len(correct_points))
    print(len(incorrect_points) / (len(correct_points) + len(incorrect_points)))
    print(vector_length(np.array([[2,2,2], [0,0,0]])))
    print(euclidean_distance(np.array([[2, 2], [0, 0]]), np.array([[3,3]])))
    print(nearest_neighbour(np.array([[2, 2], [0, 0]]), np.array([[3, 3], [-2, -3]])))