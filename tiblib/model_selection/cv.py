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


def CVMinDCF(model, X, y, K=5, pi=.5):
    if X.shape[0] < X.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    best_score = np.inf
    best_act_score = None
    scores = []
    best_model = None

    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        model.fit(X_train, y_train)

        val_scores = model.predict_scores(X_val, get_ratio=True)
        score, _ = min_detection_cost_func(val_scores, y_val, pi=pi)
        act_score = detection_cost_func(val_scores, y_val, pi=pi)
        scores.append(score)
        if score < best_score:
            best_score = score
            best_act_score = act_score
            best_model = deepcopy(model)
    return best_score, best_act_score, best_model


def CVCalibration(model, X, y, K=5, pi=.5, _lambda=1e-3):
    if X.shape[0] < X.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    best_score = np.inf
    best_act_score = None
    scores = []
    best_model = None

    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        model.fit(X_train, y_train)

        val_scores = model.predict_scores(X_val, get_ratio=True)
        cal_scores = calibrate(val_scores, y_val, _lambda, pi)

        score, _ = min_detection_cost_func(cal_scores, y_val, pi=pi)
        act_score = detection_cost_func(val_scores, y_val, pi=pi)
        scores.append(score)
        if score < best_score:
            best_score = score
            best_act_score = act_score
            best_model = deepcopy(model)
    return best_score, best_act_score, best_model
