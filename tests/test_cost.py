from pyml import compute_cost

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

def test_cost_2():
    list_x = [[1, 1], [1, 2], [1, 3]]
    list_theta = [[0], [1]]
    list_y = [[1], [2], [3]]

    x = np.matrix(list_x)
    theta = np.matrix(list_theta)
    y = np.matrix(list_y)

    j = compute_cost(x, y, theta)

    assert j == 0.0

def test_cost_3():
    list_x = [[1, 1], [1, 2], [1, 3]]
    list_theta = [[0], [1]]
    list_y = [[1], [2]]

    x = np.matrix(list_x)
    theta = np.matrix(list_theta)
    y = np.matrix(list_y)

    j = compute_cost(x, y, theta)

    assert j == float('inf')

def test_cost_4():
    list_x = [[1, 1], [1, 2], [1, 3]]
    list_theta = [[0]]
    list_y = [[1], [2], [3]]

    x = np.matrix(list_x)
    theta = np.matrix(list_theta)
    y = np.matrix(list_y)

    j = compute_cost(x, y, theta)

    assert j == float('inf')

def test_cost_5():
    list_x = []
    list_theta = []
    list_y = []

    x = np.matrix(list_x)
    theta = np.matrix(list_theta)
    y = np.matrix(list_y)

    j = compute_cost(x, y, theta)

    assert j == float('inf')
