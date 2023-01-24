import warnings
from copy import deepcopy

import numpy as np

from tiblib import min_detection_cost_func, detection_cost_func
from tiblib.classification import BinaryLogisticRegression


def calibrate(score, y_true, _lambda, pi=0.5):
    lr = BinaryLogisticRegression(l=_lambda)
    lr.fit(score.reshape(-1,1), y_true)
    alpha = lr.w
    beta_p = lr.b
    cal_score = alpha * score + beta_p - np.log(pi / (1 - pi))
    return cal_score



class Kfold:
    def __init__(self, num_sample=5):
        self.num_sample = num_sample

    def split(self, X):
        part = len(X) // self.num_sample
        index = np.arange(len(X))
        folds = []
        splits = np.empty(self.num_sample, dtype=np.ndarray)
        for j, i in enumerate(range(0, len(X), part)):
            splits[j] = (index[i:i + part])
        index = np.arange(self.num_sample)
        for i in range(self.num_sample):
            folds.append((np.concatenate(splits[index != i]), splits[i]))
        return folds


class CVMinDCF:
    def __init__(self, model, K=5, pi=.5):
        '''
        Runs KFold CV on a given model with min DCF as a metric.

        :param model: model to cross-validate
        :param K: number of folds
        :param pi: prior over model scores
        '''
        self.model = model
        self.K = K
        self.scores = []
        self.best_score = 1
        self.best_model = None
        self.act_score = None
        self.pi = pi

    def score(self, X, y):
        if X.shape[0] < X.shape[1]:
            warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = int(n / self.K)
        for i in range(self.K):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            self.model.fit(X_train, y_train)

            val_scores = self.model.predict_scores(X_val, get_ratio=True)
            score, _ = min_detection_cost_func(val_scores, y_val, pi=self.pi)
            act_score = detection_cost_func(val_scores, y_val, pi=self.pi)
            self.scores.append(score)
            if score < self.best_score:
                self.best_score = score
                self.act_score = act_score
                self.best_model = deepcopy(self.model)
        return self.best_score, self.best_model, self.act_score


class CVCalibration:
    def __init__(self, model, _lambda=1e-3, K=5, pi=.5):
        '''
        Runs KFold CV on a given model with min DCF as a metric.
        Calibrates the scores using a LogisticRegression with
        a given lambda

        :param model: model to cross-validate
        :param _lambda: logreg hyperparameter during calibration
        :param K: number of folds
        :param pi: prior over model scores
        '''
        self.model = model
        self.K = K
        self.scores = []
        self.best_score = 1
        self.best_model = None
        self.pi = pi
        self._lambda = _lambda

    def score(self, X, y):
        if X.shape[0] < X.shape[1]:
            warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = int(n / self.K)
        for i in range(self.K):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            self.model.fit(X_train, y_train)

            val_scores = self.model.predict_scores(X_val, get_ratio=True)
            cal_scores = calibrate(val_scores, y_val, self._lambda, self.pi)
            score, _ = min_detection_cost_func(cal_scores, y_val, pi=self.pi)
            act_score = detection_cost_func(val_scores, y_val, pi=self.pi)
            self.scores.append(score)
            if score < self.best_score:
                self.best_score = score
                self.act_score = act_score
                self.best_model = deepcopy(self.model)
        return self.best_score, self.best_model, self.act_score
