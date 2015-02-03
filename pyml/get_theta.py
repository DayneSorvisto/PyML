import numpy as np

def get_theta(x, y):
	theta = ((x.T * x).I) * x.T * y;
	return theta
