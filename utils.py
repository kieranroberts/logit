import numpy

def multimap(fnc,X):
     return numpy.array(map((lambda x:map(lambda y:fnc(y),x)), X))