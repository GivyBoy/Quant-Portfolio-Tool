from __future__ import annotations

import numpy as np


class linear_regression:
    def __init__(self, learning_rate=1e-3, iters=10_000, batch_size=32) -> None:
        assert learning_rate > 0, "Learning rate must be greater than 0"
        assert iters > 0, "Number of iterations must be greater than 0"
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

<<<<<<< HEAD
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
=======
    def fit(self, X: np.ndarray, y: np.ndarray, method = "SGD") -> None:
        if method == "Closed Form":
            ones = np.ones((len(X), 1))  # zeros column as 2D array
            sample_X = np.hstack((ones, X))
            self.weights = np.linalg.inv(sample_X.T.dot(sample_X)).dot(X.T).dot(y)
            return
>>>>>>> 44c46da (Edited the readme)
        num_samples, num_features = X.shape

        # init weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.iters):
            idx = np.random.randint(0, num_samples, self.batch_size)
            X_batch = np.take(X, idx, axis=0)
            y_batch = np.take(y, idx, axis=0)
            y_pred = self.predict(X_batch)
            error = y_batch - y_pred
            dw = -(2 * (X_batch.T).dot(error)) / num_samples
            db = -(2 * np.sum(error)) / num_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum((y_true - y_pred) ** 2) / len(y_true)
