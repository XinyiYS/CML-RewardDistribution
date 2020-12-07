import numpy as np
from tqdm.notebook import trange
from utils.mmd import mmd, mmd_update, perm_sampling
from utils.utils import union
from utils.maxheap import MaxHeap
from multiprocessing import Pool


def greedy(G_i, D_i, mu_target, Y, kernel):
    print("Running greedy algorithm with a -MMD^2 target of {}".format(mu_target))
    m = G_i.shape[0]
    maxheap = MaxHeap()
    R_i = []
    deltas = []
    mus = []

    mmd_init, A, B, C = mmd(D_i, Y, kernel)
    mu = -mmd_init
    for i in trange(m, desc="Initial delta computation"):
        x = G_i[i:i + 1]
        delta = -mmd_update(x, D_i, Y, A, B, C, kernel)[0] - mu
        maxheap.heappush((delta, x))

    with trange(m) as t:
        for i in t:
            t.set_description('Adding points')
            added = False

            while added is False:
                _, x = maxheap.heappop()
                mmd_new, A_temp, B_temp = mmd_update(x, union(D_i, R_i), Y, A, B, C, kernel)
                delta = -mmd_new - mu
                if len(maxheap.h) == 0 or delta >= maxheap[0][0]:
                    R_i.append(np.squeeze(x))
                    mu += delta
                    deltas.append(delta)
                    mus.append(mu)
                    A = A_temp
                    B = B_temp
                    added = True
                    t.set_postfix(point=x, delta=delta, mu=mu)
                    # print("Added {} with delta of {}, current -MMD^2 of {}".format(x, delta, mu))
                else:
                    maxheap.heappush((delta, x))

            if mu > mu_target:  # Exit condition
                break
            if len(maxheap.h) == 0:
                raise Exception('Max-heap is empty!')

    return R_i, deltas, mus


def process_func(params):
    """
    Function to be passed to multiprocessing Pool.
    :param params: (D, D[i], Y, kernel, num_perms, phi, candidates)
    :return: List of rewards
    """
    D, D_i, Y, kernel, num_perms, phi, candidates = params

    # Work with negative mmds from here on, so larger is better
    mmds = np.array(perm_sampling(np.concatenate(D), Y, kernel, num_perms)) * -1
    M_min, M_max = (min(mmds), max(mmds))

    mu = M_min + phi * (M_max - M_min)
    return greedy(candidates, D_i, mu, Y, kernel)


def con_div(candidates, Y, phi, D, kernel, num_perms=100):
    """
    Controlled divergence algorithm.
    :param candidates: Candidate points from generator distribution, one for each party. array of shape (k, m, d)
    :param Y: Reference points to measure MMD against. array of shape (l, d)
    :param phi: reward vector of values in [0, 1]. array of shape (k)
    :param D: Parties data. array of shape (k, n, d)
    :param kernel: kernel to measure MMD
    """
    k = D.shape[0]

    # Construct params list
    params = [(D, D[i], Y, kernel, num_perms, phi[i], candidates[i]) for i in range(k)]

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
