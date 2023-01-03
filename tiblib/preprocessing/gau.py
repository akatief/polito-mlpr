import numpy as np
from tiblib import TransformerBase


class GAU(TransformerBase):
    def __init__(self, mu, var):
        self.mu = mu  # 🐄
        self.var = var
        self.gau = None

    def fit(self, X, y=None):
        self.gau = X

    def transform(self, X):
        if self.gau is None:
            raise ValueError('GAU was not fitted on any data!')
        return -0.5*np.log(2*np.pi)-0.5*np.log(self.var)-np.power(self.gau-self.mu, 2)/(2*self.var)