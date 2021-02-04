import numpy as np
import torch


def union(D, R):
    if len(R) == 0:
        return D
    else:
        return np.concatenate((D, R), axis=0)


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


def norm(lst):
    max_val = max(lst)
    return list(map(lambda x: (x)/(max_val), lst))
