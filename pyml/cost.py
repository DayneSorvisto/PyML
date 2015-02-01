import numpy as np

def compute_cost(x, y, theta):
	J = sum((np.dot(x, theta) - y) ** 2) / ( 2 * len(y))
	return J