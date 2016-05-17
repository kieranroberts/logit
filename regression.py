#  Regression module
#  We train theta using training examples X to minimise the regularised 
#  cost function J w.r.t the regularised parameter lambda using num_iter
#  of iterations.

import numpy
from scipy.optimize import fmin_cg
import sigmoid
import predict
import costFunctionReg
import featureNormalize

class logistic:
	def __init__(self, X, y, L, num_iter):
		self.X = X  
		self.y = y
		self.L = L
		self.iterations = num_iter

	def objectiveFunction(self, var):
		return costFunctionReg.costFunctionReg(var, self.X, self.y, self.L)

	def objectiveGrad(self, var):
		return costFunctionReg.grad(var, self.X, self.y, self.L)

	def solve(self):
	#========== Normalise the features and add neutral feature ================
		self.X = featureNormalize.featureNormalize(self.X)
		m = self.X.shape[0]
		n = self.X.shape[1]
		self.X = numpy.hstack((numpy.ones((m,1)), self.X))
		initialTheta = numpy.zeros(n+1).reshape(n+1,1)

		optimalTheta = fmin_cg(self.objectiveFunction, initialTheta, fprime=self.objectiveGrad,maxiter=self.iterations)
		objectiveValue = self.objectiveFunction(optimalTheta)
		prediction = numpy.dot(self.X, optimalTheta)
		return optimalTheta, objectiveValue, predict.predict(optimalTheta, self.X)

class regression:
	def __init__(self, method, X, y, L, num_iter):
		self.regressor = None
		self.method = method
		if self.method == 'logistic':
			self.regressor = logistic(X, y, L, num_iter)

	def solve(self):
		if self.method == 'logistic':
			return self.regressor.solve()
