from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import pickle
from metrics.phi_div import average_dkl
from metrics.wasserstein import wasserstein_2
from metrics.class_imbalance import get_classes, class_proportion


ex = Experiment("metrics")
ex.observers.append(FileStorageObserver('runs'))


@ex.automain
def main(ds, split, greed):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))
    run_id = ex.current_run._id

    if ds == 'diabetes':
        num_parties = 3
    else:
        num_parties = 5

    if ds == 'gmm':
        num_classes = 5
    elif ds == 'diabetes':
        num_classes = 3
    else:
        num_classes = 10

    results_dict = pickle.load(open('data/results_dict.p', 'rb'))
    party_datasets = results_dict[ds][split][greed]['party_datasets']
    rewards = results_dict[ds][split][greed]['rewards']
    reference_dataset = results_dict[ds][split][greed]['reference_dataset']
    alpha = results_dict[ds][split][greed]['alpha']
    candidate_datasets = results_dict[ds][split][greed]['candidate_datasets']
    candidate_labels = results_dict[ds][split][greed]['candidate_labels']

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

    #Save
    pickle.dump((class_props, wass_before, wass_after, dkls_before, dkls_after),
                open("data/metrics-{}-{}-{}.p".format(ds, split, greed), 'wb'))
