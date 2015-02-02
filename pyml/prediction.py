import numpy as np
import pyml

def predict(x, theta, mu, sigma):
    x = np.hstack((x[:, 0], (x[:, 1:] - mu) / sigma))
    return x * theta
