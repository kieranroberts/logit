import numpy

def featureNormalize(X):
    dim = X.shape
    
    if len(dim) == 1: # X is a single training example, a row vector
        return X
    else:
        X = numpy.array(X,dtype=float) # if the entries of X are integers, then line 14 will always produce a column of integers instead of floats.
        for i in range(dim[1]):
            if (X[:, i] == numpy.zeros((1,dim[0]))).all():
                continue
            else:
                X[:, i] = numpy.divide(X[:, i] - numpy.mean(X[:, i]),numpy.std(X[:, i]))
    return X

# Note that in line 14:
# X[:, i] - numpy.mean(X[:, i]) is actually vector subtracting a floating number
# numpy seems to treat this as subtraction element-wise, which is what we want.
# But maybe it would be more correct to write:
#
# X[:, i] - numpy.mean(X[:, i])*numpy.ones((1,dim[0]))
#