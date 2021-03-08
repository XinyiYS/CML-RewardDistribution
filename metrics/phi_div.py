import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


def dkl(P, Q, k=2):
    """
    Calculates an estimate of the Kullback-Leibler divergence between P and Q based on finite samples from P and Q
    using k-nearest neighbours (Perez-Cruz, 2008)
    :param P: array of size (n, d)
    :param Q: array of size(m, d)
    :param k: int, kth-nearest neighbour
    :return: KL(P||Q) estimate
    """

    (n, d) = P.shape
    m = Q.shape[0]

    P_nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(P)
    P_distances, indices = P_nbrs.kneighbors(P)

    PQ_distances = distance.cdist(P, Q, metric='euclidean')

    kl = 0
    for i in range(n):
        P_idx = indices[i, k]
        P_distance = P_distances[i, k]

        partition = np.argpartition(PQ_distances[i], k)
        idx = partition[k-1]
        PQ_distance = PQ_distances[i, idx]
        kl += np.log(PQ_distance) - np.log(P_distance)

    kl *= (d/n)
    kl += np.log(m) - np.log(n-1)

    return kl
