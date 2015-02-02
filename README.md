# PyML
[![Build Status](https://api.travis-ci.org/rohithpr/PyML.svg?branch=master)](https://api.travis-ci.org/rohithpr/PyML)
[![Latest Version](https://pypip.in/version/PyML/badge.svg)](https://pypi.python.org/pypi/PyML/)

Machine Learning algorithms in Python

### Installation

`$ pip install PyML`

### Usage

The first column of the training set x must be all 1s.

The following code is assumed in all the examples.

```python
import numpy as np
import pyml
```

#### Cost

```python
list_x = [[1, 1, 100], [1, 2, 101], [1, 3, 102]]
list_theta = [[0], [1], [0]]
list_y = [[1], [2], [3]]

x = np.matrix(list_x)
theta = np.matrix(list_theta)
y = np.matrix(list_y)

cost = pyml.compute_cost(x, y, theta)
print('Cost: ', cost)
```

```
Cost:  0.0
```

#### Feature normalization

```python
list_x = [[1, 1, 100], [1, 2, 101], [1, 3, 102]]
x = np.matrix(list_x)

(x, mu, sigma) = pyml.normalize_features(x)
print('Normalized x: ', x)
```

```
Normalized x:  [[ 1.         -1.22474487 -1.22474487]
 [ 1.          0.          0.        ]
 [ 1.          1.22474487  1.22474487]]
```
