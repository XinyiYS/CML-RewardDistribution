import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.pipeline import get_data
from core.kernel import get_kernel
from core.reward_calculation import get_v, shapley, get_vN, get_v_is
from core.utils import norm

ex = Experiment("CGM")
ex.observers.append(FileStorageObserver('runs'))


@ex.config
def params():
    dataset = "GMM"
    split = "unequal"  # "equaldisjoint" or "unequal"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 5
    d = 2
    party_data_size = 1000
    candidate_data_size = 10000
    unequal_prop = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.6, 0.4, 0.0, 0.0, 0.0],
                             [0.0, 0.2, 0.6, 0.2, 0.0],
                             [0.0, 0.0, 0.0, 0.4, 0.6]])


@ex.automain
def main(dataset, split, greed, condition, num_parties, num_classes, d, party_data_size,
         candidate_data_size, unequal_prop):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))

    party_datasets, reference_dataset, candidate_datasets, candidate_labels = get_data(dataset,
                                                                                       num_classes,
                                                                                       d,
                                                                                       num_parties,
                                                                                       party_data_size,
                                                                                       candidate_data_size,
                                                                                       split,
                                                                                       unequal_prop)

    kernel = get_kernel(dataset, d)

    v = get_v(party_datasets, reference_dataset, kernel)
    print("Coalition values:\n{}".format(v))
    phi = shapley(v, num_parties)
    print("Shapley values:\n{}".format(phi))
    alpha = norm(phi)
    print("alpha:\n{}".format(alpha))
    vN = get_vN(v, num_parties)
    v_is = get_v_is(v, num_parties)
