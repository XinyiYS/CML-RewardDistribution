import numpy as np
from data.GMM import sample_GMM
from core.utils import split_proportions


def get_proportions(split, num_parties, unequal_prop):
    if split == 'equaldisjoint':
        return np.eye(num_parties)
    elif split == 'unequal':
        return unequal_prop


def get_data(dataset, num_classes, d, num_parties, party_data_size, candidate_data_size, split, unequal_prop):
    if dataset == 'GMM':
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
        prop = get_proportions(split, num_parties, unequal_prop)
        party_datasets, party_labels = split_proportions(data_in_classes, prop)

        # Candidate datasets
        gmm_points, candidate_labels = sample_GMM(means, covs, candidate_data_size)
        candidate_datasets = np.array([gmm_points]*num_parties)

        # Reference dataset
        alL_parties_dataset = np.concatenate(party_datasets)
        reference_dataset = np.concatenate([alL_parties_dataset, candidate_datasets[0]], axis=0)

    elif dataset == 'MNIST':
        party_datasets = []
        reference_dataset = []
        candidate_datasets = []
        candidate_labels = []

    elif dataset == 'CIFAR':
        party_datasets = []
        reference_dataset = []
        candidate_datasets = []
        candidate_labels = []

    else:
        raise Exception("Parameter dataset must be 'GMM', 'MNIST', or 'CIFAR'")

    return party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels
