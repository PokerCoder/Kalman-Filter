import numpy as np


class KalmanFilter:
    """
    A class for implementing a Kalman Filter.

    Attributes:
        F (numpy.ndarray): The state-transition model matrix.
        H (numpy.ndarray): The observation model matrix.
        Q (numpy.ndarray): The covariance matrix for the process noise.
        R (numpy.ndarray): The covariance matrix for the observation noise.
        x (numpy.ndarray): The current state estimate.
        P (numpy.ndarray): The current state covariance matrix.

    Example usage:
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
    """

    def __init__(self, F, H, Q, R, x0, P0):
        """
        Initializes the KalmanFilter class with initial values.

        Args:
            F (numpy.ndarray): The state-transition model matrix.
            H (numpy.ndarray): The observation model matrix.
            Q (numpy.ndarray): The covariance matrix for the process noise.
            R (numpy.ndarray): The covariance matrix for the observation noise.
            x0 (numpy.ndarray): The initial state estimate.
            P0 (numpy.ndarray): The initial state covariance matrix.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        """
        Predicts the next state and updates the state covariance matrix.

        Returns:
            numpy.ndarray: The predicted state.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Updates the state estimate and state covariance matrix 
        based on the given observation.

        Args:
            z (numpy.ndarray): The observation vector.
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.identity(self.P.shape[0]) - K @ self.H) @ self.P
