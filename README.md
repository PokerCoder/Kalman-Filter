# Kalman Filter

A Python implementation of the Kalman Filter using NumPy.

![kalman filter diagram](diagram.png)

## Description

This class provides an implementation of a Kalman Filter for state estimation and prediction.

### Attributes

- `F` (numpy.ndarray): The state-transition model matrix.
- `H` (numpy.ndarray): The observation model matrix.
- `Q` (numpy.ndarray): The covariance matrix for the process noise.
- `R` (numpy.ndarray): The covariance matrix for the observation noise.
- `x` (numpy.ndarray): The current state estimate.
- `P` (numpy.ndarray): The current state covariance matrix.

## Example Usage

```python
import numpy as np
from kalman_filter import KalmanFilter

# Initialize the system matrices
F = np.array([[1, 1], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[0.1, 0], [0, 0.1]])
R = np.array([[0.5]])
x0 = np.array([0, 0])
P0 = np.array([[1, 0], [0, 1]])

# Create a Kalman Filter object
kf = KalmanFilter(F, H, Q, R, x0, P0)

# Simulate measurements and update the filter
measurements = [0.5, 1.0, 1.5, 2.0, 2.5]
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    print("State estimate:", kf.x)
```

## Methods

### `__init__(self, F, H, Q, R, x0, P0)`

Initializes the KalmanFilter class with initial values.

#### Arguments

- `F` (numpy.ndarray): The state-transition model matrix.
- `H` (numpy.ndarray): The observation model matrix.
- `Q` (numpy.ndarray): The covariance matrix for the process noise.
- `R` (numpy.ndarray): The covariance matrix for the observation noise.
- `x0` (numpy.ndarray): The initial state estimate.
- `P0` (numpy.ndarray): The initial state covariance matrix.

### `predict(self)`

Predicts the next state and updates the state covariance matrix.

#### Returns

- `numpy.ndarray`: The predicted state.

### `update(self, z)`

Updates the state estimate and state covariance matrix based on the given observation.

#### Arguments

- `z` (numpy.ndarray): The observation vector.
