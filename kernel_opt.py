from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data_features
from core.kernel import get_kernel, optimize_kernel

ex = Experiment("kernel_opt")
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
    gamma = '0'
    gpu = True
    batch_size = 2048
    optimize_kernel_params = True
    num_pareto_val_points = 2000


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
    gamma = '0'
    gpu = True
    batch_size = 2048
    optimize_kernel_params = True
    num_pareto_val_points = 250


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
    gamma = '0'
    gpu = True
    batch_size = 2048
    optimize_kernel_params = True
    num_pareto_val_points = 250


@ex.automain
def main(dataset, split, mode, greed, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, perm_samp_high, perm_samp_low, perm_samp_iters, kernel, gamma, gpu,
         batch_size, optimize_kernel_params, num_pareto_val_points):
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
    kernel = optimize_kernel(kernel, device, party_datasets, reference_dataset,
                             batch_size=64, num_val_points=num_pareto_val_points)
