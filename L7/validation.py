import numpy as np
from random import shuffle


def cross_validation_indices(current_step, shuffled_indices):
    shuffled_indices = np.array(shuffled_indices)
    test_indices = shuffled_indices == current_step
    training_indices = test_indices == False

    return training_indices, test_indices


def shuffle_indices(data_size, total_steps):
    training_indices = []
    for current_step in range(total_steps):
        training_indices += [current_step] * (data_size // total_steps)
    missing_size = data_size - len(training_indices)
    training_indices += list(range(data_size))[:missing_size]
    shuffle(training_indices)
    return np.array(training_indices)


def cross_validation(data, labels, classifier, total_steps=10):
    data_size = len(labels)
    scores = []
    shuffled_indices = shuffle_indices(data_size, total_steps)
    for i in range(total_steps):
        training_indices, test_indices = cross_validation_indices(i, shuffled_indices)
        classifier.fit(data[training_indices], labels[training_indices])
        score = classifier.score(data[test_indices], labels[test_indices])
        scores.append(score)
    return scores


if __name__ == "__main__":
    indices = shuffle_indices(30, 10)
    print(indices, cross_validation_indices(4, indices), sep="\n")
