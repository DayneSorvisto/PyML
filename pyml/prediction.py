import math
import numpy as np
import pyml

def predict(x, theta, mu, sigma):
    x = np.hstack((x[:, 0], (x[:, 1:] - mu) / sigma))
    return x * theta

def classification(x, theta, mu, sigma):
	prediction = predict(x, theta, mu, sigma)
	return 1 / (1 + math.e ** -prediction)
