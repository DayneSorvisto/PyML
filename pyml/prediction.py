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
    # Issue #1
    try:
        g = 1 / (1 + math.e ** -(x))
    except:
        g = np.matrix([1 / (1 + math.e ** -(int(i[0]))) for i in x])
    return g
