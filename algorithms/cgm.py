import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from tqdm.notebook import trange
import heapq
import math
import pickle
import itertools
from matplotlib import cm
from multiprocessing import Pool
from scipy.special import softmax
from utils.mmd import mmd, mmd_update, mmd_update_batch, perm_sampling
from utils.utils import union
from utils.maxheap import MaxHeap


def split_proportions(dataset, proportions):
    """
    :param dataset: array of shape (num_classes, N, d).
    :param proportions: array of probability simplices of shape (num_classes, num_classes). Must sum to 1 along
    all rows and columns
    """
    num_classes, N, d = dataset.shape
    split_datasets = [[] for i in range(num_classes)]
    dataset_idx = [0 for i in range(num_classes)]
    
    for i in range(num_classes):
        for j in range(num_classes):
            prop = proportions[i, j]
            for k in range(int(prop * N)):
                split_datasets[i].append(dataset[j, dataset_idx[j]])
                dataset_idx[j] += 1
    
    return np.array(split_datasets)


def mmd_neg_biased(X, Y, k):
    """
    Calculates biased MMD^2. S_X, S_XY and S_YY are the pairwise-XX, pairwise-XY, pairwise-YY 
    summation terms respectively.
    :param X: array of shape (n, d)
    :param Y: array of shape (m, d)
    :param k: GPyTorch kernel
    :return: MMD^2, S_X, S_XY, S_YY
    """
    m = X.shape[0]
    n = Y.shape[0]
    X_tens = torch.tensor(X, dtype=torch.float32)
    Y_tens = torch.tensor(Y, dtype=torch.float32)

    S_X = (1 / (m ** 2)) * torch.sum(k(X_tens).evaluate())
    S_XY = (2 / (m * n)) * torch.sum(k(X_tens, Y_tens).evaluate())
    S_Y = (1 / (n ** 2)) * torch.sum(k(Y_tens).evaluate())

    return (S_XY - S_X).item(), S_X.item(), S_XY.item(), S_Y.item()


def get_v(parties_datasets, reference_dataset, kernel):
    """
    Returns a dictionary with keys as repr(set(C)), e.g. v[repr(set((4,)))] = 10, v[repr(set((1,2)))] = 18 etc.,
    for all coalitions
    :param parties_datasets: array of shape (num_parties, n, d)
    :param reference_dataset: array of shape (m, d)
    :param kernel: GPyTorch kernel
    """
    num_parties = parties_datasets.shape[0]
    party_list = list(range(1, num_parties+1))
    v = dict()
    
    for coalition_size in range(1, num_parties+1):
        for coalition in itertools.combinations(party_list, coalition_size):
            coalition_dataset = np.concatenate([parties_datasets[i-1] for i in coalition], axis=0)
            v[repr(set(coalition))] = mmd_neg_biased(coalition_dataset, reference_dataset, kernel)[0]
    
    return v


def shapley(v, num_parties):
    """
    :param v: Dictionary with keys as repr(set(C)), e.g. v[repr(set((4,)))] = 10, v[repr(set((1,2)))] = 18 etc.,
    for all permutations
    """
    sums = [0 for i in range(num_parties)]
    for perm in itertools.permutations(list(range(1, num_parties+1))):
        current_val = 0
        coalition = set()
        for party in perm:
            coalition.add(party)
            marginal = v[repr(coalition)] - current_val
            sums[party-1] += marginal
            current_val = v[repr(coalition)]
    return list(map(lambda x: (1/math.factorial(num_parties)) * x, sums))


def norm(lst):
    max_val = max(lst)
    return list(map(lambda x: (x)/(max_val), lst))


def perm_sampling_neg_biased_variant(P, Q, k, num_perms=200, eta=1.0):
    """
    Shuffles two datasets together, splits this mix in 2, then calculates MMD to simulate P=Q. Does this num_perms
    number of times.
    :param P: First dataset, array of shape (n, d)
    :param Q: Second dataset, array of shape (m, d)
    :param k: GPyTorch kernel
    :param num_perms: Number of permutations done to get range of MMD values.
    :param eta: Fraction of samples taken in each shuffle. The larger this parameter, the smaller the variance in the estimate. Defaults
    to 0.5*(n+m)
    :return: Sorted list of MMD values.
    """
    mmds = []
    num_samples = int(eta * (P.shape[0] + Q.shape[0]) // 2)
    XY = np.concatenate((P, Q))

    for _ in trange(num_perms, desc="Permutation sampling"):
        p = np.random.permutation(len(XY))
        X = XY[p[:num_samples]]
        Y = XY[p[num_samples:num_samples*2]]
        mmds.append(mmd_neg_biased(X, Y, k)[0])
    return sorted(mmds)


def perm_sampling_neg_biased(P, Q, k, num_perms=200, eta=1.0):
    """
    Shuffles two datasets together, splits this mix in 2, then calculates MMD to simulate P=Q. Does this num_perms
    number of times.
    :param P: First dataset, array of shape (n, d)
    :param Q: Second dataset, array of shape (m, d)
    :param k: GPyTorch kernel
    :param num_perms: Number of permutations done to get range of MMD values.
    :param eta: Fraction of samples taken in each shuffle. The larger this parameter, the smaller the variance in the estimate. Defaults
    to 0.5*(n+m)
    :return: Sorted list of MMD values.
    """
    mmds = []
    num_samples = int(eta * P.shape[0])

    for _ in trange(num_perms, desc="Permutation sampling"):
        p = np.random.permutation(len(P))
        X = P[p[:num_samples]]
        Y = Q
        mmds.append(mmd_neg_biased(X, Y, k)[0])
    return sorted(mmds)


def get_q(sorted_vX, vN):
    """
    :param sorted_vX: list of sorted v(X)
    :param vN: upper bound to truncate sorted_vX
    """
    
    truncated_sorted_vX = []
    for val in sorted_vX:
        if val <= vN:
            truncated_sorted_vX.append(val)
    
    def q(alpha):
        if alpha == 1:
            return vN
        else:
            return truncated_sorted_vX[math.ceil(alpha * (len(truncated_sorted_vX) - 1))]
    
    return q


def get_vN(v, num_parties):
    return v[repr(set(range(1, num_parties+1)))]


def get_alpha_min(alpha):
    if min(alpha) == 0:
        return sorted(alpha)[1]
    else:
        return min(alpha)


def get_v_is(v, num_parties):
    return [v[repr(set([i]))] for i in range(1, num_parties+1)]


def get_eta_q(vN, alpha, v_is, perm_samp_dataset, reference_dataset, kernel, low=0.001, high=1., num_iters=10, mode="all"):
    """
    Binary search for lowest value of eta that satisfies desired condition
    alpha_i: list of N alpha values
    v_i: list of N v(i) values
    """
    def all_condition(q):
        """
        For all i, q(alpha_i) > v_i
        """
        return all([q(alpha[i]) > v_is[i] for i in range(len(alpha))])
    
    def max_condition(q):
        """
        q(alpha^+_{min})> max(v(i))
        """
        return q(get_alpha_min(alpha)) > max(v_is)
    
    if mode == "all":
        condition = all_condition
    elif mode == "max":
        condition = max_condition
    
    # Check high
    eta = high 
    sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta)
    q = get_q(sorted_vX, vN)
    if not condition(q):
        raise ValueError("High value of eta already violates {} condition".format(mode))
    
    # Check low
    eta = low
    sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta)
    q = get_q(sorted_vX, vN)
    if condition(q):
        print("Low value of eta already satisfies {} condition".format(mode))
        return eta, q
    
    current_low = low
    current_high = high
    current_high_q = q
    for i in range(num_iters):
        print("Iteration {}".format(i))
        print("current_high={}, current_low={}".format(current_high, current_low))
        eta = (current_high + current_low) / 2
        print("Evaluating for eta = {}".format(eta))
        sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta)
        q = get_q(sorted_vX, vN)
        
        if condition(q):
            print("{} condition satisfied, setting current_high to {}".format(mode, eta))
            current_high = eta
            current_high_q = q
        else:
            print("{} condition not satisfied, setting current_low to {}".format(mode, eta))
            current_low = eta
        
        if current_low >= current_high:
            print("Low greater than or equal to high, terminating")
            break
            
    return current_high, current_high_q


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
                    (2 / ((m+1) ** 2)) * torch.sum(k(x_tens, X_tens).evaluate(), axis=1) + \
                    (1 / ((m+1) ** 2)) * torch.diag(k(x_tens).evaluate())
            
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

    with trange(m) as t1:
        for _ in t1:
            t1.set_description("Additions with greed {}".format(greed))
            DuR = union(D, R)
                        
            neg_mmds_new, S_Xs_temp, S_XYs_temp = v_update_batch(G, DuR, Y, S_X, S_XY, kernel)
            deltas_temp = neg_mmds_new - mu
            weights = deltas_temp
            
            weight_max = np.amax(weights)
            weight_min = np.amin(weights)
            weights = (weights - weight_min) / (weight_max - weight_min)  # Scale weights to [0, 1] because greed factor may not affect
            # sampling for very small/large weight values

            probs = softmax(greed * weights)
            idx = np.random.choice(len(G), p=probs)
            
            x = G[idx:idx+1]
            delta = deltas_temp[idx]
            mu += delta
            deltas.append(delta)
            mus.append(mu)
            S_X = S_Xs_temp[idx]
            S_XY = S_XYs_temp[idx]
            
            R.append(np.squeeze(x).copy())
            G = np.delete(G, idx, axis=0)
            t1.set_postfix(delta=str(delta), mu=str(mu), target=str(mu_target))

            if mu > (1-rel_tol) * mu_target:  # Exit condition
                break

    return R, deltas, mus
            

def process_func(params):
    """
    Function to be passed to multiprocessing Pool.
    :param params: (D, D[i], Y, kernel, null_mmds, phi, candidates)
    :return: List of rewards
    """
    D_i, Y, kernel, mu, candidates, greed = params

    return weighted_sampling(candidates, D_i, mu, Y, kernel, greed)


def reward_realization(candidates, Y, r, D, kernel, greeds=None):
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
    params = [(D[i], Y, kernel, r[i], candidates[i], greeds[i]) for i in range(k)]

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

