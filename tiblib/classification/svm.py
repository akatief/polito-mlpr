import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from tiblib import load_iris_multiclass, train_test_split


class SVC:
	def __init__(self, C=1.0, K= 1.0):
		self.W = None
		self.b = None
		self.K = K
		self.C = C

	def fit(self, X, y):
		n_samples, n_features = X.shape
		# Initialize the primal variables
		alpha = np.zeros(n_samples)
		# Initialize the kernel matrix
		X = X.T
		x_r = np.vstack((X, np.full(n_samples, self.C)))
		z = np.where(y == 1, 1, -1)
		H = x_r.T @ x_r * z.reshape((-1, 1)) * z

		# Define the optimization function
		def objective(alpha):
			return 0.5 * alpha.T @ H @ alpha - np.sum(alpha), (H @ alpha - 1).reshape(n_samples)

		def obj_primal(w):
			0.5 * np.square(np.linalg.norm(w)) + self.C * np.sum(np.maximum(0, 1 - z*(w.T @ x_r))), np.linalg.norm(w) + self.C * np.sum(np.maximum(0, -z*x_r))

		# Define the bounds
		bounds = [(0, self.C) for _ in range(n_samples)]

		# Optimize the primal variables using scipy's fmin_l_bfgs_b function
		result = fmin_l_bfgs_b(objective, alpha, bounds=bounds, approx_grad = False, factr=1.)[0]

		res = np.sum(result * z * x_r, 1)
		self.b = res[-1]
		self.W = res[:-1]
		result2 = fmin_l_bfgs_b(obj_primal, res, approx_grad = False, factr=1.)[0]
		print(result2 + result)

	def predict(self, X):
		predictions = np.heaviside(self.W.T @ X.T + self.b * self.C, 0)
		return predictions


if __name__ == '__main__':
	# Create a toy dataset
	X, y = load_iris_multiclass()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
	NOT_CLASS = 2
	# Create an instance of the SVC class
	clf = SVC()
	mask = y_train != NOT_CLASS
	y_train[y_train == 2] = NOT_CLASS
	# Fit the classifier to the data
	clf.fit(X_train[mask], y_train[mask])

	# Predict the labels for a new set of inputs
	mask = y_test != NOT_CLASS
	y_test[y_test == 2] = NOT_CLASS
	predictions = clf.predict(X_test[mask])
	print(predictions)
	print((predictions == y_test[mask]).mean())

