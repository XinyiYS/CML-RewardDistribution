import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from tqdm import tqdm


def dkl(P, Q, k=2):
    """
    Calculates an estimate of the Kullback-Leibler divergence between P and Q based on finite samples from P and Q
    using k-nearest neighbours (Perez-Cruz, 2008)
    :param P: array of size (n, d)
    :param Q: array of size(m, d)
    :param k: int, kth-nearest neighbour
    :return: KL(P||Q) estimate
    """
    print("Calculating DKL for k={}".format(k))
    (n, d) = P.shape
    m = Q.shape[0]
    
    #print("Calculating PQ_distances")
    PQ_distances = np.zeros(n)
    min_PQ_dist = np.inf

    print("Calculating P_distances using NN")
    P_nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(P)
    P_distances, indices = P_nbrs.kneighbors(P)

    kl = 0
    print("Calculating PQ_distances")
    for i in tqdm(range(n)):
        dist_arr = np.linalg.norm(Q - P[i], axis=1)
        
        partition = np.argpartition(dist_arr, k)
        idx = partition[k]
        PQ_distances[i] = dist_arr[idx]
        
        curr_min = np.min(dist_arr[np.nonzero(dist_arr)])
        if curr_min < min_PQ_dist:
            min_PQ_dist = curr_min

    log_P_dist_sum = 0
    log_PQ_dist_sum = 0
    num_skips = 0
    num_replacements = 0
    
    print("Updating KL divergence")
    for i in range(n):
        P_distance = P_distances[i, k]

        if P_distance == 0:  # If P distance is 0, PQ distance will also be 0 so just skip
            num_skips += 1
            continue
        
        PQ_distance = PQ_distances[i]
        if PQ_distance == 0:  # If PQ distance is 0, replace with smallest distance observed for numerical reasons
            num_replacements += 1
            PQ_distance = min_PQ_dist
    
        #kl += np.log(PQ_distance) - np.log(P_distance)
        log_PQ_dist_sum += np.log(PQ_distance)
        log_P_dist_sum += np.log(P_distance)
    
    print("num_skips: {}".format(num_skips))
    print("num_replacements: {}".format(num_replacements))
    print("log_P_dist_sum: {}".format(log_P_dist_sum))
    print("log_PQ_dist_sum: {}".format(log_PQ_dist_sum))
    kl += log_PQ_dist_sum - log_P_dist_sum
    print("KL before factor and adding logs: {}".format(kl))
    factor = d/n
    print("factor: {}".format(factor))
    kl *= factor
    const = np.log(m) - np.log(n-1)
    print(const)
    kl += const
    print("final KL: {}".format(kl))
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
