import random
import numpy as np
from sklearn import metrics


def test_training_indices(train_size=100, data_size=150):
    training_indices = [True] * train_size + [False] * (data_size - train_size)
    random.shuffle(training_indices)
    training_indices = np.array(training_indices)
    test_indices = training_indices == False

    return training_indices, test_indices


def cross_validation_indices(current_step, total_steps=10, data_size=150):
    training_indices = [True] * (current_step * data_size // total_steps)
    range_len = (((current_step + 1) * data_size // total_steps) -
                 (current_step * data_size // total_steps))
    training_indices += [False] * range_len
    training_indices += [True] * (data_size - len(training_indices))
    training_indices = np.array(training_indices)
    test_indices = training_indices == False

    return training_indices, test_indices


def cross_validation(data, labels, classifier, total_steps=10):
    data_size = len(labels)
    scores = []
    for i in range(total_steps):
        training_indices, test_indices = cross_validation_indices(i, total_steps, data_size)
        classifier.fit(data[training_indices], labels[training_indices])
        score = classifier.score(data[test_indices], labels[test_indices])
        scores.append(score)
    return scores


def print_scores(classifier, data, labels, tested_variable):
    print(tested_variable, end=": ")
    print(classifier.fit(data, labels).score(data, labels))
    score = cross_validation(data, labels, classifier)
    print(score, np.mean(score), "\n", sep="\n")


def confusion_matrix(classifier, data, targets):
    training_indices, test_indices = test_training_indices(
        data_size=len(targets),
        train_size=len(targets) // 3)
    classifier.fit(data[training_indices],
                   targets[training_indices])
    predicted_labels = classifier.predict(data[test_indices])
    return metrics.confusion_matrix(targets[test_indices], predicted_labels)