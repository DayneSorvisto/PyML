import numpy as np
import pyml

def gradient_descent_base(x, y, theta, alpha, num_iters, want_history=True):
    m = len(y)
    cost_history = None
    if want_history:
        cost_history = np.zeros(num_iters)

    for idx in range(num_iters):
        temp = x.T * (x * theta - y);
        theta = theta - alpha / m * temp;

        if want_history:
            cost_history[idx] = pyml.linear_regression_cost(x, y, theta)
    return (theta, cost_history)

def gradient_descent(x, y, theta, alpha, num_iters):
    return gradient_descent_base(x, y, theta, alpha, num_iters, want_history=False)[0]

def gradient_descent_history(x, y, theta, alpha, num_iters):
    return gradient_descent_base(x, y, theta, alpha, num_iters, want_history=True)
