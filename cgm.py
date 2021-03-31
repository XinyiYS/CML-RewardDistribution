import numpy as np
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data_features
from core.kernel import get_kernel, median_heuristic, optimize_kernel
from core.reward_calculation import get_v, shapley, get_vN, get_v_is, get_eta_q, get_q_rho, opt_vstar, get_v_maxs, \
    get_vCi
from core.reward_realization import reward_realization
from core.utils import norm
from metrics.class_imbalance import get_classes, class_proportion
from metrics.phi_div import dkl
from metrics.wasserstein import wasserstein_2


ex = Experiment("CGM")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def gmm():
    dataset = "gmm"
    split = "unequal"  # "equaldisjoint" or "unequal"
    mode = "opt_vstar"
    greed = 1
    condition = "stable"
    num_parties = 5
    num_classes = 5
    d = 2  # Only for GMM
    party_data_size = 1000
    candidate_data_size = 50000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gpu = True
    batch_size = 2048
    optimize_kernel_params = False


@ex.named_config
def mnist():
    dataset = "mnist"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    mode = "opt_vstar"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 8
    party_data_size = 10000
    candidate_data_size = 100000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = False


@ex.named_config
def cifar():
    dataset = "cifar"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    mode = "opt_vstar"  # "perm_samp" or "rho_shapley"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 8
    party_data_size = 10000
    candidate_data_size = 100000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = False


@ex.automain
def main(dataset, split, mode, greed, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, perm_samp_high, perm_samp_low, perm_samp_iters, kernel, gpu,
         batch_size, optimize_kernel_params):
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
                                                                                            split)

    kernel = get_kernel(kernel, d, 1., device)
    if optimize_kernel_params:
        print("Optimizing kernel parameters")
        # WARNING DO NOT USE THIS IF KERNEL IS NOT CUSTOM INVERSE LENGTHSCALE SQUARED KERNEL
        # kernel = optimize_kernel(kernel, device, party_datasets, reference_dataset)
    else:  # Use precomputed inverse lengthscales to save time
        if dataset == 'gmm':
            if split == 'unequal':
                kernel.lengthscale = np.sqrt(1 / np.array([2.2049, 1.0718]))
            elif split == 'equaldisjoint':
                kernel.lengthscale = np.sqrt(1 / np.array([2.1063, 1.0573]))
        elif dataset == 'mnist':
            if split == 'unequal':
                kernel.lengthscale = np.sqrt(1 / np.array([0.0027, 0.0202, 0.0017, 0.0277, 0.0121, 0.0010, 0.0513, 0.0071]))
            elif split == 'equaldisjoint':
                kernel.lengthscale = np.sqrt(1 / np.array([0.0047, 0.0149, 0.0069, 0.0216, 0.0261, 0.0211, 0.0538, 0.0410]))
        elif dataset == 'cifar':
            if split == 'unequal':
                kernel.lengthscale = np.sqrt(1 / np.array([0.0166, 0.0093, 0.0027, 0.0005, 0.0888, 0.0289, 0.2482, 0.1091]))
            elif split == 'equaldisjoint':
                kernel.lengthscale = np.sqrt(1 / np.array([2.5984e-02, 9.1558e-05, 1.1350e-06, 3.9977e-04, 9.2788e-02, 1.7913e-02,
                                           2.2021e-03, 6.0790e-02]))
        else:
            raise Exception("Tried to use precomputed inverse lengthscales but received invalid dataset")

    # Reward calculation
    v = get_v(party_datasets, reference_dataset, kernel, device=device, batch_size=batch_size)
    print("Coalition values:\n{}".format(v))
    phi = shapley(v, num_parties)
    print("Shapley values:\n{}".format(phi))
    alpha = norm(phi)
    print("alpha:\n{}".format(alpha))
    vN = get_vN(v, num_parties)
    v_is = get_v_is(v, num_parties)
    v_Cis = [get_vCi(i, phi, v) for i in range(1, num_parties + 1)]
    print("V_Cis:\n{}".format(v_Cis))
    v_maxs = get_v_maxs(party_datasets, reference_dataset, candidate_datasets[0], kernel, device, batch_size)
    print("v_maxs:\n{}".format(v_maxs))

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
    elif mode == 'rho_shapley':
        print("Using rho-Shapley to calculate reward vector")
        q, rho = get_q_rho(alpha, v_is, vN, phi, v, cond=condition)
        print("rho: {}".format(rho))
    elif mode == 'opt_vstar':
        print("Using opt_vstar to calculate reward vector")
        q, v_star, v_star_frac, rho = opt_vstar(alpha, v_is, v_maxs, v_Cis, cond=condition, rho_penalty=-0.001)
        print("v*: {}".format(v_star))
        print("Fraction of maximum possible v*: {}".format(v_star_frac))
        print("rho: {}".format(rho))
    else:
        raise Exception("mode must be perm_samp, rho_shapley or opt_vstar")

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
                                              device=device,
                                              batch_size=batch_size)

    # Save results
    pickle.dump((party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels, rewards, deltas, mus),
                open("runs/{}/CGM-{}-{}-greed{}-{}.p".format(run_id,
                                                             dataset,
                                                             split,
                                                             greed,
                                                             condition), "wb"))
    print("Results saved successfully")

    # Metrics
    print("Length of rewards: {}".format([len(r) for r in rewards]))

    print("alpha:\n{}".format(alpha))

    class_props = []
    for result in rewards:
        class_props.append(class_proportion(get_classes(np.array(result), candidate_datasets[0], candidate_labels), num_classes))
    print("Class proportions and class imbalance: {}".format(class_props))

    dkl_before = [dkl(party_datasets[i], reference_dataset) for i in range(num_parties)]
    dkl_after = [dkl(np.concatenate([party_datasets[i], np.array(rewards[i])], axis=0), reference_dataset) for i in range(num_parties)]

    print("Reverse KL before: \n{}".format(dkl_before))
    print("Reverse KL after: \n{}".format(dkl_after))
    print("Correlation coefficient with alpha: \n{}".format(np.corrcoef(alpha, dkl_after)))

    wass_before = [wasserstein_2(party_datasets[i], reference_dataset) for i in range(num_parties)]
    wass_after = [wasserstein_2(np.concatenate([party_datasets[i], np.array(rewards[i])], axis=0), reference_dataset) for i in range(num_parties)]
    print("Wasserstein-2 before: \n{}".format(wass_before))
    print("Wasserstein-2 after: \n{}".format(wass_after))
    print("Correlation coefficient with alpha: \n{}".format(np.corrcoef(alpha, wass_after)))

    # Save results and metrics in convenient place
    pickle.dump((party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels,
                 rewards, deltas, mus, alpha, class_props, dkl_before, dkl_after, wass_before, wass_after),
                open("data/{}/cgm-results/CGM-{}-{}-greed{}-{}-run{}.p".format(dataset,
                                                                               dataset,
                                                                               split,
                                                                               greed,
                                                                               condition,
                                                                               run_id), "wb"))
