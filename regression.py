# Steps
# Given a data point:
# Calculate its distance from all other data points
# Get the closest k points
# Calculate the average of the values

import numpy as np
from utils import euclidean_distance

class KNNRegressor:
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X) -> None:
        return [self.__predict(x) for x in X]

    def __predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_y = [self.y_train[ki] for ki in k_indices]
        return np.average(k_nearest_y)
