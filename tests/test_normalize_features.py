from pyml import normalize_features

import nose
import numpy as np

def test_1():
    list_x = [[1, 1], [1, 2], [1, 3]]
    x = np.matrix(list_x)

    (x, mu, sigma) = normalize_features(x)

    assert mu == 2.0
    assert sigma == 0.816496580927726

    # Error caused because of floating point numbers.
    # eg: x[0, 1] == -1.22474487139 is False.
    # Find a workaround.
    # assert (x == np.matrix([[ 1., -1.22474487139],
    #     [1., 0.],
    #     [1., 1.22474487139]])).all()
