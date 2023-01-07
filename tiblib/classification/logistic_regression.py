import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from tiblib import ClassifierBase
from scipy.special import logsumexp



class BinaryLogisticRegression(ClassifierBase):
    def __init__(self, l=1):
        self.l = l # Regularization term
        self.w = None
        self.b = None
        self.f_min = None

    def fit(self, X, y):
        labels = np.unique(y)
        labels.sort()
        assert len(labels) == 2, 'Labels are nonbinary'
        assert np.alltrue(labels == np.array([0,1])), 'Labels are not in format 0/1'
        X = X.T
        n_feats, n_samples = X.shape
        p_init = np.zeros(n_feats + 1)

        # Defined inside fit to use X, y and avoid being static
        def objective(p):
            w, b = p[:-1], p[-1]
            regularization = 0.5 * self.l * np.linalg.norm(w, 2)**2
            boundary = (w.T @ X + b)
            zi = y * 2 - 1
            # Uses logaddexp to sum 1 + exp(...) avoiding numerical issues
            summation = np.sum(np.logaddexp(0,-zi * boundary)) / n_samples
            return  regularization + summation
        p_optim, self.f_min, _ = fmin_l_bfgs_b(objective, p_init, approx_grad=True)
        self.w = p_optim[:-1]
        self.b = p_optim[-1]
        return self

    def predict(self, X, return_probs=False):
        n_feats, n_samples = X.shape

        if self.w is None or self.b is None:
            raise ValueError('Logistic regression was not fitted on any data!')
        X = X.T
        score = self.w @ X + self.b
        y_pred = score > 0

        if return_probs:
            return y_pred, score, self.f_min
        else:
            return y_pred



class LogisticRegression(ClassifierBase):
    def __init__(self, l=1):
        self.l = l # Regularization term
        self.w = None
        self.b = None
        self.f_min = None
        self.n_labels = None

    def fit(self, X, y):
        labels = np.unique(y)
        labels.sort()
        assert len(labels) > 1, 'There is less than 2 classes'
        n_labels = len(labels)
        self.n_labels = n_labels
        X = X.T
        n_feats, n_samples = X.shape
        # W, b of dimensionality KxD, K
        p_init = np.zeros(n_feats * self.n_labels + self.n_labels)
        onehot_y = np.array(y.reshape(-1, 1) == np.unique(y), dtype="int32")

        # Defined inside fit to use X, y and avoid being static
        def objective(p):
            p = p.reshape(n_labels, n_feats + 1)
            W, b = p[:, :-1], p[:, -1]
            b = b.reshape(-1, 1)
            # scores for all classes
            Scores = W @ X + b
            log_y = Scores - logsumexp(Scores).reshape(-1, 1)
            regularization = 0.5 * self.l * np.sum(W * W)

            summation = np.sum(onehot_y.T * log_y)
            return regularization - summation / n_samples
        p_optim, self.f_min, d = fmin_l_bfgs_b(objective, p_init, approx_grad=True)
        p_optim = p_optim.reshape(n_labels, n_feats + 1)

        self.w = p_optim[:, :-1]
        self.b = p_optim[:, -1].reshape(-1, 1)
        return self

    def predict(self, X, return_probs=False):
        if self.w is None or self.b is None:
            raise ValueError('Logistic regression was not fitted on any data!')
        X = X.T
        score = self.w @ X + self.b
        y_pred = np.argmax(score, axis=0)

        if return_probs:
            return y_pred, score, self.f_min
        else:
            return y_pred