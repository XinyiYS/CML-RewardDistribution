import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


def dkl(P, Q, k=2):
    """
    Calculates an estimate of the Kullback-Leibler divergence between P and Q based on finite samples from P and Q
    using k-nearest neighbours (Perez-Cruz, 2008). WARNING: hard coded for k < 8
    :param P: array of size (n, d)
    :param Q: array of size(m, d)
    :param k: int, kth-nearest neighbour
    :return: KL(P||Q) estimate
    """

    (n, d) = P.shape
    m = Q.shape[0]

    P_nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(P) ## WARNING: Hard-coded for k < 8
    P_distances, indices = P_nbrs.kneighbors(P)

    PQ_distances = distance.cdist(P, Q, metric='euclidean')

    kl = 0
    for i in range(n):
        P_distance = P_distances[i, k]
        new_k = k
        while P_distance == 0:
            new_k += 1
            P_distance = P_distances[i, new_k]

        PQ_distance = 0
        new_k = k
        while PQ_distance == 0:
            partition = np.argpartition(PQ_distances[i], new_k)
            idx = partition[new_k]
            PQ_distance = PQ_distances[i, idx]
            new_k += 1
            if new_k > 8:
                raise Exception("num_k > 8")
        kl += np.log(PQ_distance) - np.log(P_distance)

    kl *= (d/n)
    kl += np.log(m) - np.log(n-1)
    print('kl: {}'.format(kl))
    return kl


def average_dkl(party_datasets, reference_dataset, min_k=2, max_k=6):
    num_k = max_k - min_k + 1
    num_parties = len(party_datasets)
    dkl_sum = np.zeros(num_parties)

    for k in range(min_k, max_k + 1):
        # print("k: {}".format(k))
        dkl_after = np.array([dkl(party_datasets[i],
                                  reference_dataset,
                                  k=k) for i in range(num_parties)])
        print("DKL for k = {}: {}".format(k, dkl_after))
        dkl_sum += dkl_after

    avg_dkl = dkl_sum / num_k

    return avg_dkl
