import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from tiblib import load_iris_multiclass, train_test_split


class SVC:
	def __init__(self, C=1.0, K=1.0, kernel='linear', c=0, d=0, gamma=0):
		self.x = None
		self.k = None
		self.z = None
		self.alpha = None
		self.gamma = gamma
		self.d = d
		self.c = c
		self.kernel = kernel
		self.W = None
		self.b = None
		self.K = K
		self.C = C

	def fit(self, X, y):
		n_samples, n_features = X.shape
		# Initialize the primal variables
		alpha = np.zeros(n_samples)
		X = X.T
		self.x = X
		x_r = np.vstack((X, np.full(n_samples, self.K)))

		# Initialize the kernel matrix
		if self.kernel == 'linear':
			self.k = x_r.T @ x_r
		else:
			self.k = self._kern(X, X, self.kernel)
		self.z = np.where(y == 1, 1, -1)
		H = self.k * self.z.reshape((-1, 1)) * self.z

		# Define the optimization function
		def objective(alpha):
			return 0.5 * alpha.T @ H @ alpha - np.sum(alpha), (H @ alpha - 1).reshape(n_samples)

		# Define the bounds
		bounds = [(0, self.C) for _ in range(n_samples)]

		# Optimize the primal variables using scipy's fmin_l_bfgs_b function
		self.alpha = fmin_l_bfgs_b(objective, alpha, bounds=bounds, approx_grad=False, factr=1.)[0]

		res = np.sum(self.alpha * self.z * x_r, 1)
		self.b = res[-1]
		self.W = res[:-1]

	def predict(self, X):
		if self.kernel == 'linear':
			predictions = self.W.T @ X.T + self.b * self.C
		else:
			predictions = (self.alpha * self.z) @ self._kern(self.x, X.T, self.kernel)
		return np.heaviside(predictions, 0)

	def _kern(self, x1, x2, ker_type):
		if ker_type == 'poly':
			return np.power(x1.T @ x2 + self.c, self.d) + self.K ** 2
		elif ker_type == 'radial':
			# return np.exp(-self.gamma * np.square(np.linalg.norm(x1.T-x2.T))) + self.K ** 2
			kern = np.zeros([x1.shape[1], x2.shape[1]])
			for i in range(x1.shape[1]):
				for j in range(x2.shape[1]):
					norm = ((x1[:, i] - x2[:, j]) ** 2).sum()
					kern[i, j] = np.exp(-self.gamma * norm) + self.K
			return kern
		else:
			raise ValueError(f"{self.kernel} is not a valid kernel type, valid type are: 'linear', 'poly', 'radial'")
