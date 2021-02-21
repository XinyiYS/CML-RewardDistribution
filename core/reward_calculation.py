import numpy as np
import math
import itertools
import scipy.stats

from core.mmd import mmd_neg_biased_batched


def get_v(parties_datasets, reference_dataset, kernel, device, batch_size=32):
    """
    Returns a dictionary with keys as repr(set(C)), e.g. v[repr(set((4,)))] = 10, v[repr(set((1,2)))] = 18 etc.,
    for all coalitions
    :param parties_datasets: array of shape (num_parties, n, d)
    :param reference_dataset: array of shape (m, d)
    :param kernel: GPyTorch kernel
    """
    num_parties = parties_datasets.shape[0]
    party_list = list(range(1, num_parties + 1))
    v = dict()

    for coalition_size in range(1, num_parties + 1):
        for coalition in itertools.combinations(party_list, coalition_size):
            coalition_dataset = np.concatenate([parties_datasets[i - 1] for i in coalition], axis=0)
            v[repr(set(coalition))] = mmd_neg_biased_batched(coalition_dataset,
                                                             reference_dataset,
                                                             kernel,
                                                             device=device,
                                                             batch_size=batch_size)[0]

    return v


def shapley(v, num_parties):
    """
    :param v: Dictionary with keys as repr(set(C)), e.g. v[repr(set((4,)))] = 10, v[repr(set((1,2)))] = 18 etc.,
    for all permutations
    """
    sums = [0 for i in range(num_parties)]
    for perm in itertools.permutations(list(range(1, num_parties + 1))):
        current_val = 0
        coalition = set()
        for party in perm:
            coalition.add(party)
            marginal = v[repr(coalition)] - current_val
            sums[party - 1] += marginal
            current_val = v[repr(coalition)]
    return list(map(lambda x: (1 / math.factorial(num_parties)) * x, sums))


def get_alpha_min(alpha):
    if min(alpha) == 0:
        return sorted(alpha)[1]
    else:
        return min(alpha)


def get_vN(v, num_parties):
    return v[repr(set(range(1, num_parties + 1)))]


def get_v_is(v, num_parties):
    return [v[repr(set([i]))] for i in range(1, num_parties + 1)]


def get_vCi(i, phi, v):
    lower_phi = []
    for j in range(len(phi)):
        if phi[j] <= phi[i - 1]:
            lower_phi.append(j + 1)
    return v[repr(set(lower_phi))]


def perm_sampling_neg_biased(P, Q, k, num_perms=200, eta=1.0, device='cpu'):
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

    for _ in range(num_perms):
        p = np.random.permutation(len(P))
        X = P[p[:num_samples]]
        Y = Q
        mmds.append(mmd_neg_biased_batched(X, Y, k, device)[0])
    return sorted(mmds)


def get_q(sorted_vX, vN, dist="skewnormal"):
    """
    :param sorted_vX: list of sorted v(X)
    :param vN: upper bound to truncate sorted_vX
    """

    if dist == "discrete":
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

    elif dist == "skewnormal":
        params = scipy.stats.skewnorm.fit(sorted_vX)
        skewnormal_dist = scipy.stats.skewnorm(*params)
        p_min = skewnormal_dist.cdf(0)
        if p_min < 1e-10:
            p_min = 1e-10
        p_max = skewnormal_dist.cdf(vN)

        def q(alpha):
            if alpha == 0:
                return 0
            else:
                return skewnormal_dist.ppf((p_max - p_min) * alpha + p_min)

        return q


def get_eta_q(vN, alpha, v_is, phi, v, perm_samp_dataset, reference_dataset, kernel, low=0.001, high=1., num_iters=10,
              mode="all", device='cpu'):
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

    def stable_condition(q):
        return all([q(alpha[i]) >= get_vCi(i + 1, phi, v) for i in range(len(alpha))])

    if mode == "all":
        condition = all_condition
    elif mode == "max":
        condition = max_condition
    elif mode == "stable":
        condition = stable_condition

    print("Checking high value of eta")
    eta = high
    sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta, device=device)
    q = get_q(sorted_vX, vN)
    if not condition(q):
        raise ValueError("High value of eta already violates {} condition".format(mode))

    print("Checking low value of eta")
    eta = low
    sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta, device=device)
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
        sorted_vX = perm_sampling_neg_biased(perm_samp_dataset, reference_dataset, kernel, num_perms=200, eta=eta, device=device)
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


def get_q_rho(alpha, v_is, vN, phi, v, cond='stable'):
    rho = 1

    if cond == 'all':
        for i in range(len(alpha)):
            if alpha[i] == 1:
                continue
            else:
                if (np.log(v_is[i]) - np.log(vN)) / np.log(alpha[i]) < rho:
                    rho = (np.log(v_is[i]) - np.log(vN)) / np.log(alpha[i])
    elif cond == 'stable':
        for i in range(len(alpha)):
            if alpha[i] == 1:
                continue
            else:
                vCi = get_vCi(i + 1, phi, v)
                if (np.log(vCi) - np.log(vN)) / np.log(alpha[i]) < rho:
                    rho = (np.log(vCi) - np.log(vN)) / np.log(alpha[i])
    else:
        raise Exception("cond must be either all or stable")
    return lambda x: x ** rho * vN, rho
