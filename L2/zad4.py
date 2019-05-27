import numpy as np
import math


def point_cloud(center, number=1000):
    x, y = center
    cloud_x = np.random.normal(x, 1, number)
    cloud_y = np.random.normal(y, 1, number)
    cloud = np.array(list(zip(cloud_x, cloud_y)))
    return center, cloud


def regular_polygon_points(vertex_number, edge_length):
    def distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def point(number, distance_from_centre):
        return (distance_from_centre * math.sin(number * tau / vertex_number),
                distance_from_centre * math.cos(number * tau / vertex_number))
    tau = 2 * math.pi
    initial_edge_length = distance((0, 1), (math.sin(tau / vertex_number), math.cos(tau / vertex_number)))
    distance_from_center = edge_length / initial_edge_length
    points = [point(n, distance_from_center) for n in range(vertex_number)]
    return np.array(points)


def polygon_point_clouds(points_number, edge_length, number=1000):
    polygon_points = regular_polygon_points(points_number, edge_length)
    point_clouds = [point_cloud(vertex, number) for vertex in polygon_points]
    return point_clouds


if __name__ == "__main__":
    print(point_cloud((1, 10)))
    print(regular_polygon_points(3, 2))
    print(polygon_point_clouds(2, 5, 10))
