from __future__ import annotations

import math
import time

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ml.lin_reg import linear_regression


def test_lin_reg_returns_reasonable_loss() -> None:

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=17)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    print(X_train.shape, y_train.shape)

    lin_reg = LinearRegression()
    regressor = linear_regression(learning_rate=1e-3, iters=10_000)
    lin_reg.fit(X_train, y_train)
    regressor.fit(X_train, y_train)

    lin_reg_pred = lin_reg.predict(X_test)
    regressor_pred = regressor.predict(X_test)

    lin_reg_mse = linear_regression.mse(y_test, lin_reg_pred)
    regressor_mse = linear_regression.mse(y_test, regressor_pred)

    assert math.isclose(lin_reg_mse, regressor_mse, rel_tol=0.02)


def test_lin_reg_runtime() -> None:
    X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=17)
    sgd_start_time = time.time()
    regressor = linear_regression(learning_rate=1e-3, iters=10_000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    regressor.fit(X_train, y_train)
    sgd_end_time = time.time()
    sgd_time = sgd_end_time - sgd_start_time
    closed_start_time = time.time()
    regressor = linear_regression(method="Closed Form")
    regressor.fit(X, y)
    closed_end_time = time.time()
    closed_time = closed_end_time - closed_start_time
    print(f"SGD Time: {sgd_time} \t Closed Form Time: {closed_time} \t")
