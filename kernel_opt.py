from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data_features
from core.kernel import get_kernel, optimize_kernel
from core.reward_calculation import get_v, shapley, get_vN, get_v_is, get_eta_q, get_q_rho, opt_vstar, get_v_maxs, \
    get_vCi
from core.utils import norm

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
    gpu = True
    batch_size = 2048
    optimize_kernel_params = True
    patience = 8


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
    party_data_size = 5000
    candidate_data_size = 100000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = True
    patience = 8


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
    party_data_size = 5000
    candidate_data_size = 100000
    perm_samp_high = 0.4
    perm_samp_low = 0.001
    perm_samp_iters = 8
    kernel = 'se'
    gpu = True
    batch_size = 256
    optimize_kernel_params = True
    patience=8


@ex.automain
def main(dataset, split, mode, greed, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, perm_samp_high, perm_samp_low, perm_samp_iters, kernel, gpu,
         batch_size, optimize_kernel_params, patience):
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
    kernel = optimize_kernel(kernel, device, party_datasets, reference_dataset, patience=patience)

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
