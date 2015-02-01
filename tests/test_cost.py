from pyml.cost import compute_cost

import nose
import numpy as np

def test_cost_1():
    list_x = [[1, 1], [1, 2], [1, 3]]
    list_theta = [[0], [0]]
    list_y = [[1], [2], [3]]

    x = np.matrix(list_x)
    theta = np.matrix(list_theta)
    y = np.matrix(list_y)

    j = compute_cost(x, y, theta)

    assert j == 2.3333333333333335
