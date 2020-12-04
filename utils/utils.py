import copy
import numpy as np
from .kernels import rbf, dot


def MMD_hat_squared(X, Y, kernel=dot):
	n, m = len(X), len(Y)
	
	X = X.reshape(len(X), -1)
	Y = Y.reshape(len(Y), -1)
	pairwise_X = kernel(X, X)
	pairwise_Y = kernel(Y, Y)
	pairwise_XY = kernel(X, Y)
	mmd = 1.0/(n*(n-1)) * (pairwise_X.sum() - pairwise_X.trace())  + 1.0/(m*(m-1))* (pairwise_Y.sum()-pairwise_Y.trace())
	mmd -= 2.0/(n*m)* pairwise_XY.sum()
	return mmd


def MMD_update(MMD, X, Y, x_new, A_t, B_t, kernel=dot):
	n, m = len(X), len(Y)

	A_t = -2.0/(n+1)*A_t + 1.0/(n*(n+1)) * kernel([x_new], X)
	B_t = (-1.0/(n+1)*B_t - 2.0/(m*(n+1)) * kernel([x_new], Y))
	updated_mmd = A_t + B_t + MMD
	
	return updated_mmd, A_t, B_t


def calculate_quantiles(D, k=1000, kernel=dot):
	mmds = []
	D = copy.deepcopy(D).reshape(len(D), -1)

	for i in range(k):
		np.random.shuffle(D)
		X = copy.deepcopy(D[:len(D)//2])
		Y = copy.deepcopy(D[len(D)//2:])
		mmds.append(MMD_hat_squared(X, Y, kernel))
	return sorted(mmds)


def get_quantile(alpha, D):
	assert 0<alpha <1, "alpha must be in range (0,1) exclusive."
	# D = np.append(X, Y, axis = 0)
	quantiles = calculate_quantiles(D)

	return quantiles[int(alpha*len(quantiles))]


def union(D, R):
	if len(R) == 0:
		return D
	else:
		return np.concatenate((D, R), axis=0)
