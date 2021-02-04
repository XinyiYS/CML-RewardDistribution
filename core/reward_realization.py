import numpy as np
import torch
from multiprocessing import Pool
from scipy.special import softmax

from core.utils import mmd_neg_biased, union


def v_update_batch(x, X, Y, S_X, S_XY, k):
    """
    Calculates v when we add a batch of points to a set with an already calculated v. Updating one point like this takes linear time instead of quadratic time by naively
    redoing the entire calculation.
    :param x: vector of shape (z, d)
    :param X: array of shape (n, d)
    :param Y: array of shape (m, d)
    :param A: Pairwise-XX summation term, float
    :param B: Pairwise-XY summation term (including (-2) factor), float
    :param k: GPyTorch kernel
    :return: MMD^2, A, B, all arrays of size (z)
    """
    x_tens = torch.tensor(x)
    X_tens = torch.tensor(X)
    Y_tens = torch.tensor(Y)

    m = X.shape[0]
    n = Y.shape[0]

    S_X_update = ((m ** 2) / ((m + 1) ** 2)) * S_X + \
                 (2 / ((m + 1) ** 2)) * torch.sum(k(x_tens, X_tens).evaluate(), axis=1) + \
                 (1 / ((m + 1) ** 2)) * torch.diag(k(x_tens).evaluate())

    S_XY_update = (m / (m + 1)) * S_XY + (2 / (n * (m + 1))) * torch.sum(k(x_tens, Y_tens).evaluate(), axis=1)

    S_X_arr = S_X_update.detach().numpy()
    S_XY_arr = S_XY_update.detach().numpy()

    current_v = S_XY_arr - S_X_arr

    return current_v, S_X_arr, S_XY_arr


def weighted_sampling(candidates, D, mu_target, Y, kernel, greed, rel_tol=1e-03):
    print("Running weighted sampling algorithm with -MMD^2 target {}".format(mu_target))
    m = candidates.shape[0]
    R = []
    deltas = []
    mus = []

    mu, S_X, S_XY, S_Y = mmd_neg_biased(D, Y, kernel)
    mus.append(mu)
    G = candidates.copy()

    for _ in range(m):
        if len(G) == 1:
            break

        DuR = union(D, R)
        neg_mmds_new, S_Xs_temp, S_XYs_temp = v_update_batch(G, DuR, Y, S_X, S_XY, kernel)
        deltas_temp = neg_mmds_new - mu
        weights = deltas_temp

        try:
            weight_max = np.amax(weights)
            weight_min = np.amin(weights)
            weights = (weights - weight_min) / (
                        weight_max - weight_min)  # Scale weights to [0, 1] because greed factor may not affect
            # sampling for very small/large weight values
            probs = softmax(greed * weights)
            idx = np.random.choice(len(G), p=probs)
        except:
            print("An exception occurred in the weighted sampling block")
            break

        x = G[idx:idx + 1]
        delta = deltas_temp[idx]
        mu += delta
        deltas.append(delta)
        mus.append(mu)
        S_X = S_Xs_temp[idx]
        S_XY = S_XYs_temp[idx]

        R.append(np.squeeze(x).copy())
        G = np.delete(G, idx, axis=0)

        if mu >= mu_target:  # Exit condition
            break

    return R, deltas, mus


def process_func(params):
    """
    Function to be passed to multiprocessing Pool.
    :param params: (D, D[i], Y, kernel, null_mmds, phi, candidates)
    :return: List of rewards
    """
    D_i, Y, kernel, mu, candidates, greed, rel_tol = params

    return weighted_sampling(candidates, D_i, mu, Y, kernel, greed, rel_tol)


def reward_realization(candidates, Y, r, D, kernel, greeds=None, rel_tol=1e-3):
    """
    Reward realization algorithm. Defaults to pure greedy algorithm
    :param candidates: Candidate points from generator distribution, one for each party. array of shape (k, m, d)
    :param Y: Reference points to measure MMD against. array of shape (l, d)
    :param r: reward vector. array of shape (k)
    :param D: Parties data. array of shape (k, n, d)
    :param kernel: kernel to measure MMD
    :param greeds: list of floats in range [0, inf) of size (k). 0 corresponds to pure random sampling, inf
    corresponds to pure greedy. Leave as None to set all to pure greedy. Set individual values to -1 for pure greedy
    for specific parties
    """
    k = D.shape[0]
    if greeds is None:
        greeds = [-1] * k

    # Construct params list
    params = [(D[i], Y, kernel, r[i], candidates[i], greeds[i], rel_tol) for i in range(k)]

    # Each party's reward can be computed in parallel
    with Pool(k) as p:
        R = p.map(process_func, params)

    rewards = []
    deltas = []
    mus = []
    for i in range(len(R)):
        rewards.append(R[i][0])
        deltas.append(R[i][1])
        mus.append(R[i][2])

    return rewards, deltas, mus
