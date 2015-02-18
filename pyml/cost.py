import numpy as np
from pyml.prediction import sigmoid

def linear_regression_cost(x, y, theta):
    # Throw error instead of returning inf?
    
    if x.shape[1] != theta.shape[0] \
            or x.shape[0] != y.shape[0] \
            or x.shape[0] == 0 \
            or False:
        return float('inf')
    raw_sum = sum(np.power((x * theta) - y, 2))
    J = int(np.sum(raw_sum, axis=1)) / ( 2 * len(y))
    return J

def logistic_regression_cost(x, y, theta):
    m = len(y)
    J = ((np.log(sigmoid( x * theta)) * -y ) - (np.log(1-sigmoid(x * theta)) * (1 - y))) / m
    J = float(J)
    grad = (x.T * (sigmoid(x * theta) - y)) / m # VERIFY CORRECTNESS
    return (J, grad)

def regularized_logistic_regression_cost(x, y, theta, _lambda):
    m = len(y)
    J, grad = logistic_regression_cost(x, y, theta)
    temp = float(theta[0])
    theta[0] = [0]
    J = J + _lambda / (2 * m) * (theta.T * theta)
    J = float(J)
    grad = grad + _lambda * theta / m
    theta[0] = temp
    return (J, grad)
