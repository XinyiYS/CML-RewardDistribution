import numpy as np
from scipy.linalg import sqrtm


def wasserstein_2(P, Q):
    """

    :param P: (m, d) matrix. Reference data
    :param Q: (n, d) matrix. Distribution we are evaluating
    :return: float
    """

    P_mean, P_cov = np.mean(P, axis=0), np.cov(P, rowvar=False)
    Q_mean, Q_cov = np.mean(Q, axis=0), np.cov(Q, rowvar=False)

    QP_diff = Q_mean - P_mean
    return np.inner(QP_diff, QP_diff) + np.trace(Q_cov + P_cov - 2 * sqrtm(Q_cov @ P_cov))
