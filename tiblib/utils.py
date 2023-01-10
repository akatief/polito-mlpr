import warnings
from abc import abstractmethod, ABC

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


def covariance(X):
    assert len(X.shape) == 2, 'X is not a 2D matrix'
    # X should have samples as columns
    X_centered = X - np.mean(X, axis=1).reshape(-1, 1)
    N = X_centered.shape[1]
    cov = 1 / N * (X_centered @ X_centered.T)
    return cov


class TransformerBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError('transform method was not implemented!')

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class ClassifierBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError('fit method was not implemented!')

    def score(self, X, y, metric=accuracy_score):
        return metric(y, self.predict(X))


def train_test_split(X, y, test_size, seed=0):
    assert test_size <= 1, 'test_size is more than 100%'
    train_size = 1 - test_size
    n_samples, n_feats = X.shape
    if n_feats > n_samples:
        warnings.warn("This method expects samples as rows. Are you sure X is not transposed?")
    n_train = int(n_samples * train_size)
    np.random.seed(seed)
    idx = np.random.permutation(n_samples)
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]
    return X_train, X_test, y_train, y_test


def load_iris_binary():
    iris = load_iris()
    X, y = iris['data'], iris['target']
    X = X[y != 0]  # We remove setosa from D
    y = y[y != 0]  # We remove setosa from L
    y[y == 2] = 0  # We assign label 0 to virginica (was label 2)
    return X, y


def load_iris_multiclass():
    iris = load_iris()
    X, y = iris['data'], iris['target']
    return X, y


def GAU_logpdf(x, mu, var):
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - np.power(x - mu, 2) / (2 * var)


def logpdf_GAU_ND(x: np.ndarray, mu: np.ndarray, C: np.ndarray):
    diff = x - mu
    _, slog = np.linalg.slogdet(C)
    return - (x.shape[0] * np.log(2 * np.pi) + slog + np.diagonal(diff.T @ np.linalg.inv(C) @ diff))/2
