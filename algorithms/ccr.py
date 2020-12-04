import numpy as np
from tqdm.notebook import trange
from utils.mmd import mmd, mmd_update
from utils.utils import union
import math


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
    m = candidates.shape[1]
    R = []
    deltas = [[] for i in range(k)]
    mus = [[] for i in range(k)]

    with trange(k) as t0:
        for i in t0:
            t0.set_description("Party loop")
            R_i = []
            mmd_init, A, B, C = mmd(D[i], Y, kernel)
            mu = -mmd_init
            mus[i].append(mu)
            G_i = candidates[i].copy()

            with trange(m) as t1:
                for loop0 in t1:
                    t1.set_description("Additions")
                    G_temp = []
                    DuR = union(D[i], R_i)
                    for j in range(len(G_i)):
                        x = G_i[j:j + 1]
                        mmd_new, A_temp, B_temp = mmd_update(x, DuR, Y, A, B, C, kernel)
                        delta = -mmd_new - mu
                        G_temp.append((delta, x, A_temp, B_temp, j))  # Track index j so we can remove it from G_i later

                    G_temp.sort()
                    (delta, x, A_temp, B_temp, idx_to_remove) = G_temp[
                        math.ceil(phi[i] * (len(G_temp) - 1))]  # Get point at percentile of reward vector
                    mu += delta
                    deltas[i].append(delta)
                    mus[i].append(mu)
                    A = A_temp
                    B = B_temp
                    R_i.append(np.squeeze(x.copy()))
                    G_i = np.delete(G_i, idx_to_remove, axis=0)
                    t1.set_postfix(point=x, delta=delta, mu=mu)
            R.append(R_i)

    return R, deltas, mus
