import math
import numpy as np
import pyml

def predict(x, theta, mu, sigma):
    x = np.hstack((x[:, 0], (x[:, 1:] - mu) / sigma))
    return x * theta

def classify(x, theta, _type=bool):
    classification = sigmoid(x * theta) > 0.5
    return classification.astype(_type)


def sigmoid(x):
    x_arr = np.array(x)
    g = 1 / (1 + np.exp( -1*(x_arr)))
    return np.matrix(g)
