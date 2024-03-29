import numpy as np
import pickle
import os
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data_features
from core.kernel import get_kernel, optimize_kernel_binsearch_only
from core.reward_calculation import get_v, shapley, get_vN, get_v_is, opt_vstar, get_v_maxs, get_vCi
from core.reward_realization import reward_realization
from core.utils import norm
from metrics.compute import compute_metrics

ex = Experiment("CGM")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def creditratings():
    dataset = "creditratings"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    inv_temp = 1
    condition = "stable"
    num_parties = 5
    num_classes = 5
    d = 2
    party_data_size = 1000
    candidate_data_size = 100000
    kernel = 'se'
    gpu = True
    batch_size = 2048
    optimize_kernel_params = True


@ex.named_config
def creditcard():
    dataset = "creditcard"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    inv_temp = 1
    condition = "stable"
    num_parties = 5
    num_classes = 5
    d = 4
    party_data_size = 5000
    candidate_data_size = 100000
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = True


@ex.named_config
def mnist():
    dataset = "mnist"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    inv_temp = 1
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 8
    party_data_size = 5000
    candidate_data_size = 100000
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = True


@ex.named_config
def cifar():
    dataset = "cifar"
    split = "equaldisjoint"  # "equaldisjoint" or "unequal"
    inv_temp = 1
    condition = "stable"
    num_parties = 5
    num_classes = 10
    d = 8
    party_data_size = 5000
    candidate_data_size = 100000
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = True


@ex.automain
def main(dataset, split, inv_temp, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, kernel, gpu, batch_size, optimize_kernel_params):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))
    run_id = ex.current_run._id
    if gpu is True:
        device = 'cuda:0'  # single GPU
    else:
        device = 'cpu'
    result_dir = "data/{}/cgm-results/".format(dataset)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    file_name = "CGM-{}-{}-invtemp{}-{}.p".format(dataset,
                                                  split,
                                                  inv_temp,
                                                  condition)

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
        kernel, lengthscale = optimize_kernel_binsearch_only(kernel, device, party_datasets, reference_dataset)

    print("Kernel lengthscale: {}".format(kernel.lengthscale))
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

    print("Using opt_vstar to calculate reward vector")
    q, v_star, v_star_frac, rho = opt_vstar(alpha, v_is, v_maxs, v_Cis, cond=condition, rho_penalty=-0.001)
    print("v*: {}".format(v_star))
    print("Fraction of maximum possible v*: {}".format(v_star_frac))
    print("rho: {}".format(rho))

    r = list(map(q, alpha))
    print("Reward values: \n{}".format(r))

    # Reward realization
    inv_temps = np.ones(num_parties) * inv_temp
    rewards, deltas, mus = reward_realization(candidate_datasets,
                                              reference_dataset,
                                              r,
                                              party_datasets,
                                              kernel,
                                              inv_temps=inv_temps,
                                              device=device,
                                              batch_size=batch_size)

    # Save results
    pickle.dump(
        (party_datasets, party_labels, reference_dataset, candidate_datasets, candidate_labels, rewards, deltas, mus,
         alpha),
        open(result_dir + file_name, "wb"))
    print("Results saved successfully")

    # Metrics
    compute_metrics(dataset,
                    split,
                    inv_temp,
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
                    mus)
