import numpy
import sigmoid

def predict(theta, X):
	p = sigmoid.sigmoid(numpy.dot(X, theta))
	dim = p.shape

	if len(dim) == 1:
		return numpy.array(map(lambda x: round(x), p))
	else:
		return numpy.array(map((lambda x:map(lambda y:round(y),x)), p))

# Note that in the case of logistic regression: len(dim) = 1 because p will be
# an m-by-1 vector.
