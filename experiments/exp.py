from sacred import Experiment

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
    params = {"dataset": dataset,
              "split": split,
              "greed": greed,
              "condition": condition,
              "num_parties": num_parties,
              "num_classes": num_classes,
              "d": d,
              "party_data_size": party_data_size}
    message = "Running with parameters {}".format(params)


@ex.automain
def main(message, dataset, split, greed, condition, num_parties, num_classes, d, party_data_size):
    print(message)
