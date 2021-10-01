import numpy as np
import pickle

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.svm import SVC

ex = Experiment("supervised")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def creditcard():
    ds = 'creditcard'
    num_classes = 5
    party_data_size = 5000
    condition = 'stable'
    num_parties = 5
    d = 4


@ex.named_config
def mnist():
    ds = 'mnist'
    num_classes = 10
    party_data_size = 5000
    condition = 'stable'
    num_parties = 5
    d = 8


@ex.named_config
def cifar():
    ds = 'cifar'
    num_classes = 10
    party_data_size = 5000
    condition = 'stable'
    num_parties = 5
    d = 8


@ex.automain
def main(ds, num_classes, party_data_size, condition, num_parties, d):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))
    candidate_data_size = 100000
    splits = ['equaldisjoint', 'unequal']
    inv_temps = [1, 2, 4, 8]
    result_dir = "data/{}/cgm-results/".format(ds)

    for split in splits:
        print("===========")
        print(split)
        corrs_before = []
        corrs_with_reward = []

        for inv_temp in inv_temps:
            print("-----")
            print("inv_temp = {}:".format(inv_temp))
            file_name = "CGM-{}-{}-invtemp{}-{}.p".format(ds,
                                                          split,
                                                          inv_temp,
                                                          condition)
            (
                party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels, rewards, deltas,
                mus,
                alpha) = pickle.load(open(result_dir + file_name, "rb"))

            # Trim party_labels to party_data_size
            party_labels = [party_labels[i][:party_data_size] for i in range(num_parties)]

            print("alpha = {}".format(alpha))
            # Get index in candidate set of each reward point
            candidates = candidate_datasets[0]
            reward_idxs = [[] for _ in range(num_parties)]

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

            # Concatenate rewards and labels
            party_datasets_rewards = []
            party_labels_rewards = []
            for i in range(num_parties):
                party_datasets_rewards.append(
                    np.concatenate([party_datasets[i], rewards[i]], axis=0))
                party_labels_rewards.append(np.concatenate([party_labels[i], candidate_labels[reward_idxs[i]]], axis=0))

            # Classification with SVM
            scores_before = []
            scores_with_reward = []
            for i in range(num_parties):
                # Construct test set as all real data in the system except for parties own (unseen data)
                other_party_datasets = []
                other_party_labels = []
                for j in range(num_parties):
                    if j == i:
                        continue
                    else:
                        other_party_datasets.append(party_datasets[j])
                        other_party_labels.append(party_labels[j])
                test_ds = np.concatenate(other_party_datasets)
                test_labels = np.concatenate(other_party_labels)

                # Without reward
                train_ds = party_datasets[i]
                train_labels = party_labels[i]
                clf = SVC()
                clf.fit(train_ds, train_labels)
                scores_before.append(clf.score(test_ds, test_labels))

                # With reward
                train_with_reward_ds = party_datasets_rewards[i]
                train_with_reward_labels = party_labels_rewards[i]
                clf_with_reward = SVC()
                clf_with_reward.fit(train_with_reward_ds, train_with_reward_labels)
                scores_with_reward.append(clf_with_reward.score(test_ds, test_labels))

            print("Classification accuracy before adding rewards:\n{}".format(scores_before))
            print("Classification accuracy after adding rewards:\n{}".format(scores_with_reward))

            corr_before = np.corrcoef(alpha, scores_before)[0, 1]
            print("Correlation with alpha before rewards: {}".format(corr_before))
            corr_with_reward = np.corrcoef(alpha, scores_with_reward)[0, 1]
            print("Correlation with alpha after rewards: {}".format(corr_with_reward))
            corrs_before.append(corr_before)
            corrs_with_reward.append(corr_with_reward)
        print("Average correlation with alpha before rewards for {} split: {}".format(split,
                                                                                      np.mean(corrs_before)))
        print("Average correlation with alpha after rewards for {} split: {}".format(split,
                                                                                     np.mean(corrs_with_reward)))
