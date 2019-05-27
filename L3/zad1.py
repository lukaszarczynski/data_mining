import numpy as np


def generate_data(number, dimension, distributions_number, probabilities, mi, sigma):
    sqrt_sigma = [np.linalg.cholesky(matrix) for matrix in sigma]
    data = np.random.randn(number, dimension)
    chosen_distributions = np.random.choice(np.arange(distributions_number), number, p=probabilities)
    for i in range(number):
        data[i, :] = (sqrt_sigma[chosen_distributions[i]] @ data[i, :]) + mi[chosen_distributions[i], :]
    return data


if __name__ == "__main__":
    number = 5000
    dimension = 2
    distributions_number = 5
    probabilities = None
    mi = np.array([3 * k * np.ones(dimension) for k in range(distributions_number)])
    sigma = np.array([np.identity(dimension)] * distributions_number)

    data = generate_data(number, dimension, distributions_number, probabilities, mi, sigma)
