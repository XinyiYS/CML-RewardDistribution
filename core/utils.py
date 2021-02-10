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
    :param proportions: array of probability simplices of shape (num_parties, num_classes). Must sum to 1 along
    all rows and columns
    :return: party_datasets array of shape (num_parties, N, d), party_labels array of shape(num_parties, N)
    """
    num_classes, N, d = dataset.shape
    num_parties = proportions.shape[0]
    split_datasets = [[] for _ in range(num_parties)]
    split_labels = [[] for _ in range(num_parties)]
    dataset_idx = [0 for _ in range(num_classes)]

    for i in range(num_parties):
        for j in range(num_classes):
            prop = proportions[i, j]
            for _ in range(int(prop * N)):
                split_datasets[i].append(dataset[j, dataset_idx[j]])
                split_labels[i].append(j)
                dataset_idx[j] += 1
                
    # Constrain all datasets to have the same length            
    min_length = min(len(ds) for ds in split_datasets)
    for i in range(num_parties): 
        split_datasets[i] = split_datasets[i][:min_length]  
        split_labels[i] = split_labels[i][:min_length]
        
    return np.array(split_datasets), np.array(split_labels)


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
