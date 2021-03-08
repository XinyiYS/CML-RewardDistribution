import numpy as np


def sample_GMM(means, covs, num_samples):
    """
    Samples equally from clusters of normal distributions.
    """
    assert (means.shape[0] == covs.shape[0])
    assert (means.shape[1] == covs.shape[1])
    assert (covs.shape[1] == covs.shape[2])

    n = means.shape[0]
    d = means.shape[1]
    samples = np.zeros((num_samples, d))
    clusters = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        cluster = np.random.randint(n)
        samples[i] = np.random.multivariate_normal(means[cluster], covs[cluster], check_valid='raise')
        clusters[i] = cluster

    return samples, clusters
