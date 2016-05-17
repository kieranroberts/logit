import numpy
import math
import utils
import sigmoid

def costFunctionReg(theta, X, y, lam):
	dim = X.shape
	m = dim[0]
	theta = theta.reshape(theta.shape[0], 1)
	J = -(1.0/m) * ( numpy.dot(numpy.transpose(y), utils.multimap(math.log, \
	sigmoid.sigmoid(numpy.dot(X, theta)))) + numpy.dot(numpy.transpose(1-y), \
	utils.multimap(math.log, 1 - sigmoid.sigmoid(numpy.dot(X,theta)))) \
	+ lam/2.0*numpy.dot(numpy.transpose(theta[1:, :]),theta[1:,:]) )

	return float(J[0])

def grad(theta, X,y,lam):
	dim = X.shape
	theta = theta.reshape(theta.shape[0], 1)
	m = dim[0]
	grad = (1.0/m) * ( numpy.dot(numpy.transpose(X), sigmoid.sigmoid( \
		numpy.dot(X, theta)) - y) + lam * numpy.vstack([[0], theta[1:,:]]))
	return grad.reshape(theta.shape[0],)
