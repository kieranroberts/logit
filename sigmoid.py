# sigmoid function used in logistic regression.
import numpy
import math

def sigmoidFunction(z):
	return 1.0/(1.0 + math.exp(-z))

def sigmoid(z):
	dim = z.shape

	if len(dim) == 1:
		for i in range(dim[0]):
			z[i] = sigmoidFunction(z[i])
	else:
		for i in range(dim[0]):
			for j in range(dim[1]):
				z[i][j] = sigmoidFunction(z[i][j])

	return z