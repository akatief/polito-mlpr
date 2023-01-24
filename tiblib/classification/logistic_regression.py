import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from tiblib import ClassifierBase
from scipy.special import logsumexp


class LogisticRegression(ClassifierBase):
    def __init__(self, l=1):
        self.l = l  # Regularization term
        self.w = None
        self.b = None
        self.f_min = None
        self.n_labels = None

    def __str__(self):
        return f'LogReg ($\\lambda = {self.l}$)'

    def _fit_binary(self, X, y):
        n_feats, n_samples = X.shape
        p_init = np.zeros(n_feats + 1)

        # Defined inside fit to use X, y and avoid being static
        def objective(p):
            w, b = p[:-1], p[-1]
            regularization = 0.5 * self.l * np.linalg.norm(w, 2) ** 2
            boundary = (w.T @ X + b)
            zi = y * 2 - 1
            # Uses logaddexp to sum 1 + exp(...) avoiding numerical issues
            summation = np.sum(np.logaddexp(0, -zi * boundary)) / n_samples
            return regularization + summation

        p_optim, self.f_min, _ = fmin_l_bfgs_b(objective, p_init, approx_grad=True)
        self.w = p_optim[:-1]
        self.b = p_optim[-1]

    def _fit_multiclass(self, X, y):
        n_feats, n_samples = X.shape
        # W, b of dimensionality KxD, K
        p_init = np.zeros(n_feats * self.n_labels + self.n_labels)
        onehot_y = np.array(y.reshape(-1, 1) == np.unique(y), dtype="int32")

        # Defined inside fit to use X, y and avoid being static
        def objective(p):
            p = p.reshape(self.n_labels, n_feats + 1)
            W, b = p[:, :-1], p[:, -1]
            b = b.reshape(-1, 1)
            # scores for all classes
            Scores = W @ X + b
            log_y = Scores - logsumexp(Scores).reshape(-1, 1)
            regularization = 0.5 * self.l * np.sum(W * W)

            summation = np.sum(onehot_y.T * log_y)
            return regularization - summation / n_samples

        p_optim, self.f_min, d = fmin_l_bfgs_b(objective, p_init, approx_grad=True)
        p_optim = p_optim.reshape(self.n_labels, n_feats + 1)

        self.w = p_optim[:, :-1]
        self.b = p_optim[:, -1].reshape(-1, 1)

    def fit(self, X, y):
        labels = np.unique(y)
        labels.sort()
        assert len(labels) > 1, 'There is less than 2 classes'
        n_labels = len(labels)
        self.n_labels = n_labels
        X = X.T

        if n_labels == 2:
            self._fit_binary(X, y)
        else:
            self._fit_multiclass(X, y)

        return self

    def predict_scores(self, X, get_ratio=False):
        if self.w is None or self.b is None:
            raise ValueError('Logistic regression was not fitted on any data!')
        if get_ratio:
            assert self.n_labels == 2, 'Multiclass LogisticRegression is not supposed' \
                                       'to return scores'
        X = X.T
        score = self.w @ X + self.b
        return score

    def predict(self, X):
        score = self.predict_scores(X)
        if self.n_labels == 2:
            return score > 0
        else:
            y_pred = np.argmax(score, axis=0)
            return y_pred
