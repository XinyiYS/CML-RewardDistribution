import numpy as np
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data_features
from core.kernel import get_kernel, median_heuristic
from core.reward_calculation import get_v, shapley, get_vN, get_v_is, get_eta_q, get_q_rho
from core.reward_realization import reward_realization
from core.utils import norm
from metrics.class_imbalance import get_classes, class_proportion

ex = Experiment("CGM")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def gmm():
    dataset = "gmm"
    split = "unequal"  # "equaldisjoint" or "unequal"
    mode = "rho_shapley"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 5
    d = 2  # Only for GMM
    party_data_size = 1000
    candidate_data_size = 10000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gamma = '0'
    gpu = True


@ex.named_config
def mnist():
    dataset = "mnist"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    mode = "rho_shapley"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 16
    party_data_size = 10000
    candidate_data_size = 40000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gamma = '0'
    gpu = True


@ex.named_config
def cifar():
    dataset = "cifar"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    mode = "rho_shapley"  # "perm_samp" or "rho_shapley"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 64
    party_data_size = 10000
    candidate_data_size = 40000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gamma = '0'
    gpu = True


@ex.automain
def main(dataset, split, mode, greed, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, perm_samp_high, perm_samp_low, perm_samp_iters, kernel, gamma, gpu):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))
    run_id = ex.current_run._id
    if gpu is True:
        device = 'cuda:0'  # single GPU
    else:
        device = 'cpu'

    # Setup data and kernel
    party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels = get_data_features(dataset,
                                                                                            num_classes,
                                                                                            d,
                                                                                            num_parties,
                                                                                            party_data_size,
                                                                                            candidate_data_size,
                                                                                            split,
                                                                                            gamma)
    print("Calculating median heuristic")
    lengthscale = median_heuristic(reference_dataset)
    kernel = get_kernel(kernel, d, lengthscale)

    # Reward calculation
    v = get_v(party_datasets, reference_dataset, kernel, device=device, batch_size=128)
    print("Coalition values:\n{}".format(v))
    phi = shapley(v, num_parties)
    print("Shapley values:\n{}".format(phi))
    alpha = norm(phi)
    print("alpha:\n{}".format(alpha))
    vN = get_vN(v, num_parties)
    v_is = get_v_is(v, num_parties)

    if mode == 'perm_samp':
        print("Using permutation sampling to calculate reward vector")
        best_eta, q = get_eta_q(vN, alpha, v_is, phi, v,
                                reference_dataset,
                                reference_dataset,
                                kernel,
                                perm_samp_low,
                                perm_samp_high,
                                perm_samp_iters,
                                mode=condition,
                                device=device)

        print("Best eta value: {}".format(best_eta))

    if mode == 'rho_shapley':
        print("Using rho-Shapley to calculate reward vector")
        q, rho = get_q_rho(alpha, v_is, vN, phi, v)
        print("rho: {}".format(rho))

    r = list(map(q, alpha))
    print("Reward values: \n{}".format(r))

    # Reward realization
    greeds = np.ones(num_parties) * greed
    rewards, deltas, mus = reward_realization(candidate_datasets,
                                              reference_dataset,
                                              r,
                                              party_datasets,
                                              kernel,
                                              greeds=greeds,
                                              rel_tol=1e-10,
                                              device=device)

    # Save results
    pickle.dump((party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels, rewards, deltas, mus),
                open("runs/{}/CGM-{}-{}-greed{}-{}.p".format(run_id,
                                                             dataset,
                                                             split,
                                                             greed,
                                                             condition), "wb"))

    print("Results saved successfully")
    print("Length of rewards: {}".format([len(r) for r in rewards]))

    class_props = []
    for result in rewards:
        class_props.append(class_proportion(get_classes(np.array(result), candidate_datasets[0], candidate_labels), num_classes))
    print("Class proportions and class imbalance: {}".format(class_props))
