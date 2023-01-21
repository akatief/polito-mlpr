from tiblib import ClassifierBase, covariance, logpdf_GAU_ND
import numpy as np
import scipy as sc


class GaussianClassifier(ClassifierBase):
	def __init__(self, num_class, naive=False, tied=False):
		self.tied = tied
		self.naive = naive
		self.mu = None
		self.cov = None
		self.logSJoint = None
		self.num_class = num_class
		if tied:
			self.tied_cov = None

	def fit(self, X, y):
		X = X.T
		cov_list = np.empty(self.num_class, dtype = np.ndarray)
		for i in range(self.num_class):
			data: np.ndarray = X[:, y == i]
			self.mu[i] = data.mean(1).reshape((-1, 1))
			self.cov[i] = covariance(data)
			if self.tied:
				cov_list[i] = self.cov[i] * data.shape[1]
		if self.tied:
			self.tied_cov = cov_list.sum(0) / X.shape[1]

	# TODO: understand if it needs to return scores or logscores
	def predict(self, X):
		if self.mu is None:
			raise ValueError('GaussianClassifier was not fitted on any data!')
		X = X.T
		li = []
		for i in range(self.num_class):
			C = self.cov[i]
			if self.tied:
				C = self.tied_cov
			if self.naive:
				C = np.diag(np.diag(C))
			li.append(logpdf_GAU_ND(X, self.mu[i], C) - np.log(self.num_class))
		self.logSJoint = np.vstack(li)
		logSMarginal = sc.special.logsumexp(self.logSJoint, axis = 0)
		logSPost = self.logSJoint - logSMarginal
		SPost = np.exp(logSPost)
		return np.argmax(SPost, 0).astype(int)
