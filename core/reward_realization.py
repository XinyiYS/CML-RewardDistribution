import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm

from core.utils import union
from core.mmd import mmd_neg_biased_batched


def v_update_batch(x, X, Y, S_X, S_XY, k):
    """
    Calculates v when we add a batch of points to a set with an already calculated v. Updating one point like this takes
    linear time instead of quadratic time by naively redoing the entire calculation.
    :param x: vector of shape (z, d)
    :param X: array of shape (n, d)
    :param Y: array of shape (m, d)
    :param S_X: Pairwise-XX summation term (NOT including minus sign), float
    :param S_XY: Pairwise-XY summation term, float
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


def v_update_batch_iter(x, X, Y, S_X, S_XY, k, device, batch_size=2048):
    """
    Calculates v when we add a batch of points to a set with an already calculated v. Updating one point like this takes
    linear time instead of quadratic time by naively redoing the entire calculation.
    :param x: vector of shape (z, d)
    :param X: array of shape (n, d)
    :param Y: array of shape (m, d)
    :param S_X: Pairwise-XX summation term (NOT including minus sign), float
    :param S_XY: Pairwise-XY summation term, float
    :param k: GPyTorch kernel
    :return: MMD^2, A, B, all arrays of size (z)
    """
    with torch.no_grad():
        x_tens = torch.tensor(x, device=device)
        X_tens = torch.tensor(X, device=device)
        Y_tens = torch.tensor(Y, device=device)

        z = x.shape[0]
        m = X.shape[0]
        n = Y.shape[0]

        S_X_arr = np.zeros(z)
        S_XY_arr = np.zeros(z)

        for i in range(int(np.ceil(z / batch_size))):
            start = i * batch_size
            end = (i + 1) * batch_size

            S_X_update = ((m ** 2) / ((m + 1) ** 2)) * S_X + \
                         (2 / ((m + 1) ** 2)) * torch.sum(k(x_tens[start:end], X_tens).evaluate(), axis=1) + (1 / ((m + 1) ** 2)) * torch.diag(k(x_tens[start:end]).evaluate())

            S_XY_update = (m / (m + 1)) * S_XY + (2 / (n * (m + 1))) * torch.sum(k(x_tens[start:end], Y_tens).evaluate(), axis=1)

            S_X_arr[start:end] = S_X_update.cpu().detach().numpy()
            S_XY_arr[start:end] = S_XY_update.cpu().detach().numpy()

        current_v = S_XY_arr - S_X_arr

    return current_v, S_X_arr, S_XY_arr


def weighted_sampling(candidates, D, mu_target, Y, kernel, inv_temp, device='cpu', batch_size=2048):
    print("Running weighted sampling algorithm with -MMD^2 target {}".format(mu_target))
    m = candidates.shape[0]
    R = []
    deltas = []
    mus = []

    mu, S_X, S_XY = mmd_neg_biased_batched(D, Y, kernel, device)
    mus.append(mu)
    G = candidates.copy()
    
    mu_max = mu
    time_since_mu_max_update = 0

    for _ in tqdm(range(m)):
        if len(G) == 1:
            break

        DuR = union(D, R)
        neg_mmds_new, S_Xs_temp, S_XYs_temp = v_update_batch_iter(G, DuR, Y, S_X, S_XY, kernel, device, batch_size)
        deltas_temp = neg_mmds_new - mu
        weights = deltas_temp

        weight_max = np.amax(weights)
        weight_min = np.amin(weights)
        weights = (weights - weight_min) / (weight_max - weight_min)  # Scale weights to [0, 1] because
        # inv_temp factor may not affect sampling for very small/large weight values
        probs = softmax(inv_temp * weights)
        idx = np.random.choice(len(G), p=probs)

        x = G[idx:idx + 1]
        delta = deltas_temp[idx]
        mu += delta
        deltas.append(delta)
        mus.append(mu)
        S_X = S_Xs_temp[idx]
        S_XY = S_XYs_temp[idx]

        R.append(np.squeeze(x).copy())
        G = np.delete(G, idx, axis=0)
        
        if mu > mu_max:
            mu_max = mu
            time_since_mu_max_update = 0
        else:
            time_since_mu_max_update += 1
            if time_since_mu_max_update >= 0.1 * len(candidates):
                print("Early stopping, no increment for a long time")
                break

        if mu >= mu_target:  # Exit condition
            break

    return R, deltas, mus


def reward_realization(candidates, Y, r, D, kernel, inv_temps=None, device='cpu', batch_size=2048):
    """
    Reward realization algorithm. Defaults to pure greedy algorithm
    :param candidates: Candidate points from generator distribution, one for each party. array of shape (k, m, d)
    :param Y: Reference points to measure MMD against. array of shape (l, d)
    :param r: reward vector. array of shape (k)
    :param D: Parties data. array of shape (k, n, d)
    :param kernel: kernel to measure MMD
    :param inv_temps: list of floats in range [0, inf) of size (k). 0 corresponds to pure random sampling, inf
    corresponds to pure greedy. Leave as None to set all to pure greedy. Set individual values to -1 for pure greedy
    for specific parties
    :param device:
    :param batch_size:
    """
    k = D.shape[0]
    if inv_temps is None:
        inv_temps = [1] * k

    rewards = []
    deltas = []
    mus = []
    for i in range(k):
        print("Running weighted sampling for party {}".format(i+1))
        reward, delta, mu = weighted_sampling(candidates=candidates[i],
                                              D=D[i],
                                              mu_target=r[i],
                                              Y=Y,
                                              kernel=kernel,
                                              inv_temp=inv_temps[i],
                                              device=device,
                                              batch_size=batch_size)
        rewards.append(reward)
        deltas.append(delta)
        mus.append(mu)
        print("Finished weight sampling for party {} with reward size {}".format(i+1, len(reward)))

    return rewards, deltas, mus
