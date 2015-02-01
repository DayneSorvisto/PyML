import numpy as np

def compute_cost(x, y, theta):
    if x.shape[1] != theta.shape[0] or (x * theta).shape[1] != theta.shape[1] or False:
        return float('inf')
    raw_sum = sum(np.power((x * theta) - y, 2))
    J = int(np.sum(raw_sum, axis=1)) / ( 2 * len(y))
    return J
