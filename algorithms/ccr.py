import numpy as np
from tqdm.notebook import trange
from utils.mmd import mmd, mmd_update_batch
from utils.utils import union
import math
from multiprocessing import Pool


def process_func(params):
    D, Y, kernel, phi, candidates = params
    m = candidates.shape[0]
    R = []
    deltas = []
    mus = []

    mmd_init, A, B, C = mmd(D, Y, kernel)
    mu = -mmd_init
    mus.append(mu)
    G = candidates.copy()

    with trange(m) as t1:
        for loop0 in t1:
            t1.set_description("Additions")
            G_temp = []
            DuR = union(D, R)

            mmds_new, As_temp, Bs_temp = mmd_update_batch(G, DuR, Y, A, B, C, kernel)
            for j in range(len(G)):
                x = G[j:j + 1]
                delta = -mmds_new[j] - mu
                G_temp.append((delta, x, As_temp[j], Bs_temp[j], j))  # Track index j so we can remove it from G_i later

            G_temp.sort()
            (delta, x, A_temp, B_temp, idx_to_remove) = G_temp[
                math.ceil(phi * (len(G_temp) - 1))]  # Get point at percentile of reward vector
            mu += delta
            deltas.append(delta)
            mus.append(mu)
            A = A_temp
            B = B_temp
            R.append(np.squeeze(x.copy()))
            G = np.delete(G, idx_to_remove, axis=0)
            t1.set_postfix(point=x, delta=delta, mu=mu)

    return R, deltas, mus


def con_conv_rate(candidates, Y, phi, D, kernel):
    """
    Controlled convergence rate algorithm.
    :param candidates: Candidate points from generator distribution, one for each party. array of shape (k, m, d)
    :param Y: Reference points to measure MMD against. array of shape (l, d)
    :param phi: reward vector of values in [0, 1]. array of shape (k)
    :param D: Parties data. array of shape (k, n, d)
    :param kernel: kernel to measure MMD
    """
    k = D.shape[0]

    # Construct params list
    params = [(D[i], Y, kernel, phi[i], candidates[i]) for i in range(k)]

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

