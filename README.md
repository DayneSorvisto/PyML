# PyML
[![Build Status](https://api.travis-ci.org/rohithpr/PyML.svg?branch=master)](https://api.travis-ci.org/rohithpr/PyML)
[![Latest Version](https://pypip.in/version/PyML/badge.svg)](https://pypi.python.org/pypi/PyML/)

Machine Learning algorithms in Python

### Installation

`$ pip install PyML`

### Contribution

Feel free to raise issues or add new features. A lot of work can be done on this project!

### Usage

* [Cost](#cost)
* [Feature normalization](#feature-normalization)
* [File operations](#file-operations)
* [Get theta](#get-theta)
* [Gradient descent](#gradient-descent)
* [Prediction](#prediction)
* [Sigmoid](#sigmoid)

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

cost = pyml.linear_regression_cost(x, y, theta)
print('Cost: ', cost)
```

```
Cost:  0.0
```

#### Feature normalization

Feature Normalization scales the data and centers it around zero.
The scaled data has zero mean and unity variance. 
```python
list_x = [[2., 5., 9], [100., 200., 300.], [5000., 7000., 9000.]]


x = np.matrix(list_x)

(x, mu, sigma) = pyml.normalize_features(x)
print('Normalized x: ', x)
```

```
Normalized x: 
[[-0.72800291 -0.736872   -0.74169951]
 [-0.68600274 -0.67691788 -0.67194044]
 [ 1.41400566  1.41378988  1.41363994]]
```

#### File operations

```python
# load(file, sep=',')
# save(file, x, mode='w', sep=',')

x = pyml.load('data1')
pyml.save('data2', x)
y = pyml.load('data2')
print((x == y).all())
```

```
True
```

#### Get theta

```python
list_x = [[1, 1], [1, 2], [1, 3]]
list_y = [[10], [20], [30]]

x = np.matrix(list_x)
y = np.matrix(list_y)

(x, mu, sigma) = pyml.normalize_features(x)

theta = pyml.get_theta(x, y)
print('Theta: ', theta)
```

```
Theta: [[ 20.        ]
 [  8.16496581]]
```

#### Gradient Descent

```python
list_x = [[1, 1], [1, 2], [1, 3]]
list_theta = [[0], [0]]
list_y = [[10], [20], [30]]

x = np.matrix(list_x)
theta = np.matrix(list_theta)
y = np.matrix(list_y)

(x, mu, sigma) = pyml.normalize_features(x)

alpha = 0.03
num_iters = 2000

theta = pyml.gradient_descent(x, y, theta, alpha, num_iters)
print('Theta: ', theta)
```

```
Theta:  [[ 20.        ]
 [  8.16496581]]
```

Note: `gradient_descent_history(x, y, theta, alpha, num_iters)` returns a tuple in which the first element is theta and the second element is a list with cost history from all the iterations.

#### Prediction

```python
list_x = [[1, 1], [1, 2], [1, 3]]
list_theta = [[0], [0]]
list_y = [[10], [20], [30]]

x = np.matrix(list_x)
theta = np.matrix(list_theta)
y = np.matrix(list_y)

(x, mu, sigma) = pyml.normalize_features(x)

alpha = 0.03
num_iters = 2000

theta = pyml.gradient_descent(x, y, theta, alpha, num_iters)

list_vals = [[1, 4], [1, 5]]
vals = np.matrix(list_vals)
predictions = pyml.predict(vals, theta, mu, sigma)
print('Predictions: ', predictions)
```

```
Predictions:  [[ 40.]
 [ 50.]]
```

#### Sigmoid

```python
list_y = [[10], [20], [30]]
s = pyml.sigmoid(y)
print('Sigmoid: ', s)

print('Sigmoid: ', list(map(pyml.sigmoid, [float('-inf'), -1, 0, 1, float('inf')])))
```

```
Sigmoid:  [[ 0.9999546  1.         1.       ]]    # Returns np.matrix for iterables
Sigmoid:  [0.0, 0.2689414213699951, 0.5, 0.7310585786300049, 1.0]    # Returns float for int/float
```
