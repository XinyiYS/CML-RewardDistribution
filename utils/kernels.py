import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, linear_kernel

def rbf(X, Y):
	return rbf_kernel(X, Y)

def dot(X, Y):
	return linear_kernel(X, Y)

def cos_sim(X, Y):
	return cosine_similarity(X, Y)

# X = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
# Y = np.asarray([[1,1,1],[0,1,0]])
# a = linear_kernel(X,Y)