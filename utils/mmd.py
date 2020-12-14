import numpy as np
import torch
from tqdm.notebook import trange

def t_statistic(mmd_2, Kxx_, Kxy, Kyy_):
    
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
    :return: t-statistic estimate

    """
    m = Kxx_.size(0)
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

    return torch.div(mmd_2, torch.sqrt(vhat))

# def mmd(X, Y, k):
#     """
#     Calculates unbiased MMD^2. A, B and C are the pairwise-XX, pairwise-XY, pairwise-YY summation terms respectively.
#     :param X: array of shape (n, d)
#     :param Y: array of shape (m, d)
#     :param k: GPyTorch kernel
#     :return: MMD^2, A, B, C
#     """
#     n = X.shape[0]
#     m = Y.shape[0]
#     X_tens = torch.tensor(X)
#     Y_tens = torch.tensor(Y)

#     A = (1 / (n * (n - 1))) * (torch.sum(k(X_tens).evaluate()) - torch.sum(torch.diag(k(X_tens).evaluate())))
#     B = -(2 / (n * m)) * torch.sum(k(X_tens, Y_tens).evaluate())
#     C = (1 / (m * (m - 1))) * (torch.sum(k(Y_tens).evaluate()) - torch.sum(torch.diag(k(Y_tens).evaluate())))

#     return (A + B + C).item(), A.item(), B.item(), C.item()


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


def perm_sampling(P, Q, k, num_perms=200):
    """
    Shuffles two datasets together, splits this mix in 2, then calculates MMD to simulate P=Q. Does this num_perms
    number of times.
    :param P: First dataset, array of shape (n, d)
    :param Q: Second dataset, array of shape (m, d)
    :param k: GPyTorch kernel
    :param num_perms: Number of permutations done to get range of MMD values.
    :return: Sorted list of MMD values.
    """
    mmds = []
    num_samples = (P.shape[0] + Q.shape[0]) // 2
    for _ in trange(num_perms, desc="Permutation sampling"):
        XY = np.concatenate((P, Q)).copy()
        np.random.shuffle(XY)
        X = XY[:num_samples]
        Y = XY[num_samples:]
        mmds.append(mmd(X, Y, k)[0])
    return sorted(mmds)
