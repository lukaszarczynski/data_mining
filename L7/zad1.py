import numpy as np


def generate_data(number, mi, sigma):
    sqrt_sigma = np.linalg.cholesky(sigma)
    data = np.random.randn(number, 2)
    for i in range(number):
        data[i, :] = (sqrt_sigma @ data[i, :]) + mi
    return data
