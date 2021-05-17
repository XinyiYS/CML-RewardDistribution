import numpy as np
import pickle
import os
from pathlib import Path
from metrics.class_imbalance import get_classes, class_proportion
from metrics.phi_div import average_dkl
from metrics.wasserstein import wasserstein_2


def compute_metrics(ds,
                    split,
                    greed,
                    num_parties,
                    num_classes,
                    alpha,
                    lengthscale,
                    party_datasets,
                    party_labels,
                    reference_dataset,
                    candidate_datasets,
                    candidate_labels,
                    rewards,
                    deltas,
                    mus):
    print("Computing metrics")
    party_datasets_with_rewards = []
    for i in range(num_parties):
        party_datasets_with_rewards.append(np.concatenate([party_datasets[i], rewards[i]], axis=0))

    print("Length of rewards: {}".format([len(r) for r in rewards]))

    print("alpha:\n{}".format(alpha))

    print("Calculating average DKLs before")
    dkls_before = average_dkl(party_datasets, reference_dataset)
    print(dkls_before)

    print("Calculating average DKLs after")
    dkls_after = average_dkl(party_datasets_with_rewards, reference_dataset)
    print(dkls_after)
    print("Correlation coefficient with alpha: \n{}".format(np.corrcoef(alpha, dkls_after)[0, 1]))

    class_props = []
    for result in rewards:
        class_props.append(
            class_proportion(get_classes(np.array(result), candidate_datasets[0], candidate_labels), num_classes))
    print("Class proportions and class imbalance of rewards: {}".format(class_props))

    print("Calculating Wasserstein-2 before")
    wass_before = [wasserstein_2(party_datasets[i], reference_dataset) for i in range(num_parties)]
    wass_after = [wasserstein_2(np.concatenate([party_datasets[i], np.array(rewards[i])], axis=0), reference_dataset)
                  for i in range(num_parties)]
    print("Wasserstein-2 before: \n{}".format(wass_before))
    print("Wasserstein-2 after: \n{}".format(wass_after))
    print("Correlation coefficient with alpha: \n{}".format(np.corrcoef(alpha, wass_after)[0, 1]))

    #Save metrics
    Path(os.getcwd() + '/data/metrics').mkdir(parents=True, exist_ok=True)
    pickle.dump((party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels,
                 rewards, deltas, mus, alpha, lengthscale, class_props, wass_before, wass_after, dkls_before, dkls_after),
                open("data/metrics/metrics-{}-{}-{}.p".format(ds, split, greed), 'wb'))
