import numpy as np
import torch
from tqdm.notebook import trange


def mmd(X, Y, k):
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
    Calculates unbiased MMD^2 when we add a single point to a set with an already calculated MMD. Calculates this
    for a batch of points. Updating one point like this takes linear time instead of quadratic time by naively 
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
