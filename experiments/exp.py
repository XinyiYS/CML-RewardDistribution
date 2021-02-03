from sacred import Experiment
import numpy as np

ex = Experiment("CGM")


@ex.config
def params():
    dataset = "GMM"
    split = "unequal"
    greed = 2
    condition = "stable"
    num_parties = 5
    num_classes = 10
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
