# Steps
# Given a data point:
# Calculate its distance from all other data points
# Get the closest k points
# Calculate the average of the values

import numpy as np
from utils import euclidean_distance
from collections import Counter

class KNNClassifier:
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, X, y) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self.__predict(x) for x in X]

    def __predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[ki] for ki in k_indices]
        majority = Counter(k_nearest_labels).most_common()

        return majority[0][0]