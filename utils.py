import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2)**2)

def mean_squared_error(y_predicted, y_actual):
    return np.mean((y_actual - y_predicted)**2)

def r2_error(y_predicted, y_actual):
    return 1 - np.sum((y_actual - y_predicted)**2) / np.sum((y_actual - np.mean(y_actual))**2)

def calculate_accuracy(y_predicted, y_actual):
    return np.sum(y_predicted == y_actual) / len(y_actual)