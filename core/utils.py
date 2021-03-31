import numpy as np
import heapq


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
    :param dataset: array of shape (num_classes, N, ..).
    :param proportions: array of probability simplices of shape (num_parties, num_classes). Must sum to 1 along
    all rows and columns
    :return: party_datasets array of shape (num_parties, N, ..), party_labels array of shape(num_parties, N)
    """
    num_classes, N = dataset.shape[0], dataset.shape[1]
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


def norm(lst):
    max_val = max(lst)
    return list(map(lambda x: x / max_val, lst))


class MaxHeapObj(object):
    def __init__(self, val): self.val = val
    def __lt__(self, other): return self.val > other.val
    def __eq__(self, other): return self.val == other.val
    def __str__(self): return str(self.val)


class MinHeap(object):
    def __init__(self): self.h = []
    def heappush(self, x): heapq.heappush(self.h, x)
    def heappop(self): return heapq.heappop(self.h)
    def __getitem__(self, i): return self.h[i]
    def __len__(self): return len(self.h)


class MaxHeap(MinHeap):
    def heappush(self, x): heapq.heappush(self.h, MaxHeapObj(x))
    def heappop(self): return heapq.heappop(self.h).val
    def __getitem__(self, i): return self.h[i].val
