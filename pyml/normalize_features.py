import numpy as np

def normalize_features(x):
    temp = x[:,0]
    x = x[:,1:]

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    x = (x - mu) / sigma
    x = np.hstack((temp, x))
    return (x, mu, sigma)
