import numpy as np
import pickle
import torchvision

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.svm import SVC

from data.pipeline import get_data_raw

ex = Experiment("classifier")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def mnist():
    dataset = 'mnist'
    split = 'equaldisjoint'
    greed = 2
    num_classes = 10
    party_data_size = 10000
    candidate_data_size = 40000
    condition = 'stable'
    num_parties = 5


@ex.automain
def main(dataset, split, greed, num_classes, party_data_size, candidate_data_size, condition, num_parties):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))

    # Get data
    (party_datasets, party_labels, reference_dataset, candidate_datasets,
     candidate_labels, rewards, deltas, mus) = pickle.load(open('data/{}/cgm-results/CGM-{}-{}-greed{}-{}.p'.format(
        dataset,
        dataset,
        split,
        greed,
        condition), 'rb'))

    # Truncate rewards to maximum achieved v
    for i in range(num_parties):
        rewards[i] = rewards[i][:np.argmax(mus[i]) + 1]

    candidates = candidate_datasets[0]
    reward_idxs = [[] for _ in range(num_parties)]
    # Get index in candidate set of each reward point
    for party in range(num_parties):
        rews = rewards[party]
        n = len(rews)
        for i in range(n):
            rows, cols = np.where(candidates == rews[i])
            index_found = False
            for row in rows:
                if np.allclose(rews[i], candidates[row]):
                    reward_idxs[party].append(row)
                    index_found = True
                    break
            if not index_found:
                raise Exception("Index not found for point {}".format(i))

    # Load original data
    party_datasets, party_labels, candidate_dataset, candidate_labels = get_data_raw(dataset=dataset,
                                                                                     num_classes=num_classes,
                                                                                     party_data_size=party_data_size,
                                                                                     candidate_data_size=candidate_data_size,
                                                                                     split=split)

    party_datasets_rewards = []
    party_labels_rewards = []
    for i in range(num_parties):
        party_datasets_rewards.append(np.concatenate([party_datasets[i], candidate_dataset[reward_idxs[i]]], axis=0))
        party_labels_rewards.append(np.concatenate([party_labels[i], candidate_labels[reward_idxs[i]]], axis=0))

    # Get test data
    if dataset == 'mnist':
        test_ds = torchvision.datasets.MNIST('data/mnist', train=False, download=True)
        test_images = np.reshape(test_ds.data.numpy(), [len(test_ds.data), -1]) / 255
        test_labels = test_ds.targets.numpy()
    elif dataset == 'cifar5':
        raise Exception("Unimplemented you lazy bastard")

    # Classification with SVM
    scores_before = []
    scores_with_reward = []
    for i in range(num_parties):
        train_images = np.reshape(party_datasets[i], [len(party_datasets[i]), -1])
        train_labels = party_labels[i]
        clf = SVC()
        clf.fit(train_images, train_labels)
        scores_before.append(clf.score(test_images, test_labels))

        train_images_with_reward = np.reshape(party_datasets_rewards[i], [len(party_datasets_rewards[i]), -1])
        train_labels_with_reward = party_labels_rewards[i]
        clf_with_reward = SVC()
        clf_with_reward.fit(train_images_with_reward, train_labels_with_reward)
        scores_with_reward.append(clf_with_reward.score(test_images, test_labels))

    print("Classification accuracy before adding rewards:\n{}".format(scores_before))
    print("Classification accuracy after adding rewards:\n{}".format(scores_with_reward))
