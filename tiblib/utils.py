from abc import abstractmethod, ABC

import numpy as np


def covariance(X):
    # X should have samples as columns
    X_centered = X - np.mean(X, axis=1).reshape(-1,1)
    N = X_centered.shape[1]
    cov = 1 / N * (X_centered @ X_centered.T)
    return cov


class TransformerBase(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError('fit method was not implemented!')

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)



class ClassifierBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError('fit method was not implemented!')