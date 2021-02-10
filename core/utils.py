import numpy as np
import torch


def union(D, R):
    if len(R) == 0:
        return D
    else:
        return np.concatenate((D, R), axis=0)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def split_proportions(dataset, proportions, party_data_size=None):
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
            for _ in range(int(prop * (num_classes/num_parties) * N)):
                split_datasets[i].append(dataset[j, dataset_idx[j]])
                split_labels[i].append(j)
                dataset_idx[j] += 1
                
    # Constrain all datasets to have the same length
    if party_data_size is None:
        party_data_size = min(len(ds) for ds in split_datasets)
    for i in range(num_parties):
        current_dataset, current_labels = unison_shuffled_copies(np.array(split_datasets[i]),
                                                                 np.array(split_labels[i]))
        split_datasets[i] = current_dataset[:party_data_size]
        split_labels[i] = current_labels[:party_data_size]
        
    return np.array(split_datasets), np.array(split_labels)


def split_data_into_classes(dataset, labels, num_classes):
    """
    :param dataset: array of shape (N * num_classes, d)
    :param labels: array of shape (N)
    :param num_classes: int
    :return: array of shape (num_classes, N, d)
    """
    N = len(labels)
    dataset_in_classes = [[] for _ in range(num_classes)]
    for i in range(N):
        dataset_in_classes[labels[i]].append(dataset[i])

    # Constrain all classes to have the same length
    min_length = min(len(ds) for ds in dataset_in_classes)
    for i in range(num_classes):
        dataset_in_classes[i] = dataset_in_classes[i][:min_length]

    return np.array(dataset_in_classes)


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
