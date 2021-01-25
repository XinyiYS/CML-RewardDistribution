import numpy as np
import torch
from torch.linalg import norm
from tqdm.notebook import trange



'''
Deep Kernal Learning with Two Sample Test MMD implementation

'''

def Pdist2(x, y):
	"""compute the paired distance between x and y."""
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y = x
		y_norm = x_norm.view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist


def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
	"""compute value of MMD and std of MMD using kernel matrix."""
	Kxxy = torch.cat((Kx,Kxy),1)
	Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
	Kxyxy = torch.cat((Kxxy,Kyxy),0)
	nx = Kx.shape[0]
	ny = Ky.shape[0]
	is_unbiased = True
	if is_unbiased:
		xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
		yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
		# one-sample U-statistic.
		if use_1sample_U:
			xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
		else:
			xy = torch.div(torch.sum(Kxy), (nx * ny))
		mmd2 = xx - 2 * xy + yy
	else:
		xx = torch.div((torch.sum(Kx)), (nx * nx))
		yy = torch.div((torch.sum(Ky)), (ny * ny))
		# one-sample U-statistic.
		if use_1sample_U:
			xy = torch.div((torch.sum(Kxy)), (nx * ny))
		else:
			xy = torch.div(torch.sum(Kxy), (nx * ny))
		mmd2 = xx - 2 * xy + yy
	if not is_var_computed:
		return mmd2, None, Kxyxy
	hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
	V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
	V2 = (hh).sum() / (nx) / nx
	varEst = 4*(V1 - V2**2)
	if  varEst == 0.0:
		# print('error_var!!'+str(V1))
		pass
	return mmd2, varEst, Kxyxy

def kernel_forward(kernel, x1, x2):
	k11 = kernel(x1, x1).evaluate()
	k12 = kernel(x1, x2).evaluate()
	k21 = kernel(x2, x1).evaluate()
	k22 = kernel(x2, x2).evaluate()

	# mmd_2, Kxx_, Kxy, Kyy_ = mmd(x1.reshape(len(x1), -1), x2.reshape(len(x2), -1), k=kernel)
	# t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)
	return k11, k12, k21, k22

def calculate_mmd(model, X, Y):
	features_x, featurex_y = model.feature_extractor(X), model.feature_extractor(Y)
	k11, k12, k21, k22 = kernel_forward(model.K_covar_module, features_x, featurex_y)
	q11, q12, q21, q22 = kernel_forward(model.q_covar_module, X.view(len(X), -1), Y.view(len(Y), -1))
	
	L = 1

	epsilon = model.epsilon
	k11 = (1-epsilon) * torch.exp(-(k11 )**L -q11 ) + epsilon * torch.exp(-q11)
	k22 = (1-epsilon) * torch.exp(-(k22 )**L -q22 ) + epsilon * torch.exp(-q22)
	k12 = (1-epsilon) * torch.exp(-(k12 )**L -q12 ) + epsilon * torch.exp(-q12)
	# H = k11 + k22 - k12 - k21
	# H = torch.div(H, len(X)* (len(X)-1))

	nx, ny = k11.shape[0], k22.shape[0]
	
	xx = torch.div((torch.sum(k11) - torch.sum(torch.diag(k11))), (nx * (nx - 1)))
	yy = torch.div((torch.sum(k22) - torch.sum(torch.diag(k22))), (ny * (ny - 1)))
	
	# one-sample U-statistic.
	use_1sample_U=True
	if use_1sample_U:
		xy = torch.div((torch.sum(k12) - torch.sum(torch.diag(k12))), (nx * (ny - 1)))
	else:
		xy = torch.div(torch.sum(k12), (nx * ny))
	mmd2 = xx - 2 * xy + yy

	hh = k11 + k22 - k12 - k21.transpose(0,1)
	V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
	V2 = (hh).sum() / (nx) / nx
	varEst = 4*(V1 - V2**2)
	mmd_hat = -1*mmd2
	t_stat = torch.div(mmd_hat, torch.sqrt(varEst+ 10 ** (-8)))
	return mmd_hat, t_stat



from math import factorial as fac
def falling_fac(n, b):
	"""
	Return the product of n..n-b+1.

	>>> falling_factorial(4, 2)  # 4*3
	12
	>>> falling_factorial(5, 3)  # 5*4*3
	60
	>>> falling_factorial(56, 1)
	56
	>>> falling_factorial(56, 0)
	1
	
	r = 1  # Running product
	for i in range(n, n-b, -1):
		r *= i
	return r
	"""
	return fac(n) // fac(n-b)

def t_statistic(mmd_2, Kxx_, Kxy, Kyy_, _eps=1e-8):
	
	"""
	Calculates the t-statistic estimator according to "Generative models and model criticism via optimized maximum mean discrepancy by Sutherland et al 2017 ICLR"
	Equation (5)
	Kxy[ij] = k(X_i, Y_i)

	Kxx_[ij] =  0 if i == j
				k(X_i, X_j) o/w
	Kyy_[ij] is similar

	fro_norm = torch.linalg.norm(matrix, ord='fro')
	
	:param mmd_2: scalar
	:param Kxx_: matrix of shape (m, m)
	:param Kxy:  matrix of shape (m, n)
	:param Kyy_: matrix of shape (n, n)
	:_eps: a small constant in case vhat is negative and cannot be taken square root
	:return: t-statistic estimate

	"""
	m = Kxx_.size(0)
	assert m >= 4, "The size of the Kxx_ currently: <{}> must be larger than or equal to 4 to avoid factorial of negative values.".format(m)

	ex = torch.ones(Kxx_.size(0), device = Kxx_.device)
	ey = torch.ones(Kyy_.size(0), device = Kyy_.device)

	vhat = 0

	#1st term
	constant =(4/ falling_fac(m, 4))
	a = torch.square(norm(Kxx_ @ ex)) + torch.square(norm(Kyy_ @ ey))
	vhat += constant * a
	
	#2nd term
	constant = 4*(m**2 - m - 1) / (m**3 * (m - 1)**2)
	a = torch.square(norm(Kxy @ ey)) + torch.square(norm(Kxy.T @ ex))
	vhat += constant * a

	# 3rd term
	constant = -8/ (m**2 * (m**2 - 3 * m + 2))   
	a =  ex.T @ Kxx_ @ Kxy @ ey + ey.T @ Kyy_ @ Kxy.T @ ex
	vhat += constant * a

	# 4th term
	constant = 8 / (m**2 * falling_fac(m, 3))
	a = (ex.T @ Kxx_ @ ex + ey.T @ Kyy_ @ ey) * (ex.T @ Kxy @ ey) 
	vhat += constant * a

	#5th term
	constant = - 2*(2*m -3)/(falling_fac(m, 2) * falling_fac(m, 4))
	a = torch.square(ex.T @ Kxx_ @ ex) + torch.square(ey.T @ Kyy_ @ ey)
	vhat += constant * a

	#6th term
	constant = -4 * (2*m - 3) / (m**3 * (m - 1)**3)
	a = torch.square(ex.T @ Kxy @ ey)
	vhat += constant * a

	#7th term
	constant = - 2/ (m* ( m**3 - 6 * m**2 + 11*m - 6 ))
	a = torch.square(norm(Kxx_, ord='fro')) + torch.square(norm(Kyy_, ord='fro'))
	vhat += constant * a

	#8th term
	constant = 4* (m-2) / (m**2 *(m-1)**3)
	a = torch.square(norm(Kxy, ord='fro'))
	vhat += constant * a

	return torch.div(mmd_2, torch.sqrt(torch.max(vhat, torch.tensor(_eps, device=vhat.device))))

def mmd_np(X, Y, k):
	 """
	 Calculates unbiased MMD^2. A, B and C are the pairwise-XX, pairwise-XY, pairwise-YY summation terms respectively.
	 :param X: array of shape (n, d)
	 :param Y: array of shape (m, d)
	 :param k: GPyTorch kernel
	 :return: MMD^2, A, B, C
	 """
	 n = X.shape[0]
	 m = Y.shape[0]
	 X_tens = torch.tensor(X)
	 Y_tens = torch.tensor(Y)

	 A = (1 / (n * (n - 1))) * (torch.sum(k(X_tens).evaluate()) - torch.sum(torch.diag(k(X_tens).evaluate())))
	 B = -(2 / (n * m)) * torch.sum(k(X_tens, Y_tens).evaluate())
	 C = (1 / (m * (m - 1))) * (torch.sum(k(Y_tens).evaluate()) - torch.sum(torch.diag(k(Y_tens).evaluate())))

	 return (A + B + C).item(), A.item(), B.item(), C.item()


def mmd(X, Y, k):
	"""
	Calculates unbiased MMD^2. Kxx_, Kxy and Kyy_ are the pairwise-XX, pairwise-XY, pairwise-YY kernel matrices respectively.
	Kxx_ and Kyy_ have zeros on their diagonals

	:param X: array of shape (n, d)
	:param Y: array of shape (m, d)
	:param k: GPyTorch kernel
	:return: MMD^2, Kxx_, Kxy, Kyy_
	"""
	n = X.shape[0]
	m = Y.shape[0]

	if torch.is_tensor(X) and torch.is_tensor(Y):
		X_tens = X.clone().detach().requires_grad_(True)
		Y_tens = Y.clone().detach().requires_grad_(True)

		A = (1 / (n * (n - 1))) * (torch.sum(k(X_tens).evaluate()) - torch.sum(torch.diag(k(X_tens).evaluate())))
		B = -(2 / (n * m)) * torch.sum(k(X_tens, Y_tens).evaluate())
		C = (1 / (m * (m - 1))) * (torch.sum(k(Y_tens).evaluate()) - torch.sum(torch.diag(k(Y_tens).evaluate())))

		Kxy  = k(X_tens, Y_tens).evaluate()
		Kxx_ = k(X_tens, X_tens).evaluate()
		Kxx_.fill_diagonal_(0)

		Kyy_ = k(Y_tens, Y_tens).evaluate()
		Kyy_.fill_diagonal_(0)

		return (A + B + C), Kxx_, Kxy, Kyy_
	else:
		X_tens = torch.tensor(X)
		Y_tens = torch.tensor(Y)

		A = (1 / (n * (n - 1))) * (torch.sum(k(X_tens).evaluate()) - torch.sum(torch.diag(k(X_tens).evaluate())))
		B = -(2 / (n * m)) * torch.sum(k(X_tens, Y_tens).evaluate())
		C = (1 / (m * (m - 1))) * (torch.sum(k(Y_tens).evaluate()) - torch.sum(torch.diag(k(Y_tens).evaluate())))

		return (A + B + C).item(), A.item(), B.item(), C.item()



def mmd_update(x, X, Y, A, B, C, k):
	"""
	Calculates unbiased MMD^2 when we add a single point to a set with an already calculated MMD. Updating one point
	like this takes linear time instead of quadratic time by naively redoing the entire calculation. Does not return
	C because it stays the same throughout.
	:param x: vector of shape (1, d)
	:param X: array of shape (n, d)
	:param Y: array of shape (m, d)
	:param A: Pairwise-XX summation term, float
	:param B: Pairwise-XY summation term (including (-2) factor), float
	:param C: Pairwise-YY summation term, float
	:param k: GPyTorch kernel
	:return: MMD^2, A, B
	"""
	x_tens = torch.tensor(x)
	X_tens = torch.tensor(X)
	Y_tens = torch.tensor(Y)

	n = X.shape[0]
	m = Y.shape[0]
	prev_mmd = A + B + C

	A_update = (-2 / (n + 1)) * A + (2 / (n * (n + 1))) * torch.sum(k(x_tens, X_tens).evaluate())
	B_update = (-1 / (n + 1)) * B - (2 / (m * (n + 1))) * torch.sum(k(x_tens, Y_tens).evaluate())

	current_mmd = A_update.item() + B_update.item() + prev_mmd
	A_new = A + A_update.item()
	B_new = B + B_update.item()

	return current_mmd, A_new, B_new


def mmd_update_batch(x, X, Y, A, B, C, k):
	"""
	Calculates unbiased MMD^2 when we add a batch of points to a set with an already calculated MMD. Updating one point like this takes linear time instead of quadratic time by naively 
	redoing the entire calculation. Does not return C because it stays the same throughout.
	:param x: vector of shape (z, d)
	:param X: array of shape (n, d)
	:param Y: array of shape (m, d)
	:param A: Pairwise-XX summation term, float
	:param B: Pairwise-XY summation term (including (-2) factor), float
	:param C: Pairwise-YY summation term, float
	:param k: GPyTorch kernel
	:return: MMD^2, A, B, all arrays of size (z)
	"""
	x_tens = torch.tensor(x)
	X_tens = torch.tensor(X)
	Y_tens = torch.tensor(Y)

	n = X.shape[0]
	m = Y.shape[0]
	prev_mmd = A + B + C

	A_update = (-2 / (n + 1)) * A + (2 / (n * (n + 1))) * torch.sum(k(x_tens, X_tens).evaluate(), axis=1)
	B_update = (-1 / (n + 1)) * B - (2 / (m * (n + 1))) * torch.sum(k(x_tens, Y_tens).evaluate(), axis=1)
	
	A_arr = A_update.detach().numpy()
	B_arr = B_update.detach().numpy()

	current_mmd = A_arr + B_arr + prev_mmd
	A_new = A + A_arr
	B_new = B + B_arr

	return current_mmd, A_new, B_new


def perm_sampling(P, Q, k, num_perms=200, num_samples=None):
	"""
	Shuffles two datasets together, splits this mix in 2, then calculates MMD to simulate P=Q. Does this num_perms
	number of times.
	:param P: First dataset, array of shape (n, d)
	:param Q: Second dataset, array of shape (m, d)
	:param k: GPyTorch kernel
	:param num_perms: Number of permutations done to get range of MMD values.
	:param num_samples: Number of samples taken in each shuffle. The larger this parameter, the smaller the variance in the estimate. Defaults
	to 0.5*(n+m)
	:return: Sorted list of MMD values.
	"""
	mmds = []
	if num_samples is None:
		num_samples = (P.shape[0] + Q.shape[0]) // 2
	for _ in trange(num_perms, desc="Permutation sampling"):
		XY = np.concatenate((P, Q)).copy()
		np.random.shuffle(XY)
		X = XY[:num_samples]
		Y = XY[num_samples:]
		mmds.append(mmd(X, Y, k)[0])
	return sorted(mmds)

