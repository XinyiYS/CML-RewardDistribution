import numpy as np
from data.GMM import sample_GMM
from core.utils import split_proportions, split_data_into_classes


def get_proportions(split, dataset):
    if dataset == 'gmm' or dataset == 'creditcard':  # WARNING: implicitly assumes 5 parties and 5 classes
        if split == 'equaldisjoint':
            return np.array([[0.96, 0.01, 0.01, 0.01, 0.01],
                             [0.01, 0.96, 0.01, 0.01, 0.01],
                             [0.01, 0.01, 0.96, 0.01, 0.01],
                             [0.01, 0.01, 0.01, 0.96, 0.01],
                             [0.01, 0.01, 0.01, 0.01, 0.96]])
        elif split == 'unequal':
            return np.array([[0.20, 0.20, 0.20, 0.20, 0.20],
                             [0.20, 0.20, 0.20, 0.20, 0.20],
                             [0.58, 0.39, 0.01, 0.01, 0.01],
                             [0.01, 0.20, 0.58, 0.20, 0.01],
                             [0.01, 0.01, 0.01, 0.39, 0.58]])
    elif dataset == 'mnist' or dataset == 'cifar':  # WARNING: implicitly assumes 5 parties and 10 classes
        if split == 'equaldisjoint':
            return np.array([[0.480, 0.480, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                             [0.005, 0.005, 0.480, 0.480, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                             [0.005, 0.005, 0.005, 0.005, 0.480, 0.480, 0.005, 0.005, 0.005, 0.005],
                             [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.480, 0.480, 0.005, 0.005],
                             [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.480, 0.480]])
        elif split == 'unequal':
            return np.array([[0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100],
                             [0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100],
                             [0.290, 0.290, 0.195, 0.195, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                             [0.005, 0.005, 0.100, 0.100, 0.290, 0.290, 0.100, 0.100, 0.005, 0.005],
                             [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.195, 0.195, 0.290, 0.290]])


def get_data_features(dataset, num_classes, d, num_parties, party_data_size, candidate_data_size, split):
    prop = get_proportions(split, dataset)

    if dataset == 'gmm':
        np.random.seed(2)
        means = np.random.uniform(size=(num_classes, d))
        covs = np.zeros((num_classes, d, d))
        for i in range(num_classes):
            covs[i] = np.eye(d) / 200

        # Party datasets
        num_samples = (party_data_size * num_parties) // num_classes
        data_in_classes = np.zeros((num_classes, num_samples, d))
        for i in range(num_classes):
            data_in_classes[i] = np.random.multivariate_normal(means[i], covs[i], size=(num_samples), check_valid='raise')
        party_datasets, party_labels = split_proportions(data_in_classes, prop)

        # Candidate datasets
        gmm_points, candidate_labels = sample_GMM(means, covs, candidate_data_size)
        candidate_datasets = np.array([gmm_points] * num_parties)

    elif dataset == 'mnist' or dataset == 'cifar' or dataset == 'creditcard':
        np.random.seed(0)
        party_datasets = np.load("data/{}/{}-party_features.npy".format(dataset, split))
        party_labels = np.load("data/{}/{}-party_labels.npy".format(dataset, split))
        candidate_dataset = np.load("data/{}/{}-cand_features.npy".format(dataset, split))
        candidate_labels = np.load("data/{}/{}-cand_labels.npy".format(dataset, split))

        candidate_datasets = np.array([candidate_dataset] * num_parties)

    else:
        raise Exception("Parameter dataset must be 'gmm', 'creditcard', 'mnist', 'cifar'")

    # If party_data_size or candidate_data_size are lower than actual, trim data
    party_datasets = party_datasets[:, :party_data_size, :]
    candidate_datasets = candidate_datasets[:, :candidate_data_size, :]

    # Reference dataset
    all_parties_dataset = np.concatenate(party_datasets)
    reference_dataset = np.concatenate([all_parties_dataset, candidate_datasets[0]], axis=0)

    return party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels


def get_data_raw(dataset, num_classes, party_data_size, candidate_data_size, split):
    prop = get_proportions(split, dataset)
    np.random.seed(0)

    train_images = np.load("data/{}/{}_train_images.npy".format(dataset, dataset))
    train_labels = np.load("data/{}/{}_train_labels.npy".format(dataset, dataset))
    candidate_images = np.load("data/{}/{}_samples.npy".format(dataset, dataset))
    candidate_labels = np.load("data/{}/{}_samples_labels.npy".format(dataset, dataset))

    # Party datasets
    data_in_classes = split_data_into_classes(train_images, train_labels, num_classes)
    party_datasets, party_labels = split_proportions(data_in_classes, prop, party_data_size)

    # Candidate dataset
    candidate_dataset = candidate_images[:candidate_data_size]
    candidate_labels = candidate_labels[:candidate_data_size]

    return party_datasets, party_labels, candidate_dataset, candidate_labels
