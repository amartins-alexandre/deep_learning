import numpy as np


def step_function(value):
    return 1 if value >= 1 else 0


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def hyperbolic_tangent(value):
    return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value))


def relu(value):
    return 0 if value >= 0 else 0
    # return np.maximum(0, value)
    # return max(0, value)


def linear(value):
    return value


def softmax(vector):
    ex = np.exp(vector)
    return ex / ex.sum()
