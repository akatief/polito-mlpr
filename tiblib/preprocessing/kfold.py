import numpy as np


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
