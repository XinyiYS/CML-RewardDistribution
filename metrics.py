from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import torch
import pickle
from scipy import stats
from core.kernel import get_kernel
from core.mmd import mmd_neg_unbiased_batched
from metrics.class_imbalance import get_classes, class_proportion

ex = Experiment("metrics")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def creditratings():
    ds = "creditratings"
    num_classes = 5
    d = 2
    party_data_size = 1000


@ex.named_config
def creditcard():
    ds = "creditcard"
    num_classes = 5
    d = 4
    party_data_size = 5000


@ex.named_config
def mnist():
    ds = "mnist"
    num_classes = 10
    d = 8
    party_data_size = 5000


@ex.named_config
def cifar():
    ds = "cifar"
    num_classes = 10
    d = 8
    party_data_size = 5000


@ex.automain
def main(ds, num_classes, d, party_data_size):
    args = dict(sorted(locals().items()))
    print("Running with parameters {}".format(args))

    # Parameters
    device = 'cuda:0'
    num_parties = 5
    splits = ['equaldisjoint', 'unequal']
    inv_temps = [1, 2, 4, 8]
    condition = 'stable'
    keys = ['party_datasets', 'party_labels', 'reference_dataset', 'candidate_datasets', 'candidate_labels',
            'rewards', 'deltas', 'mus', 'alpha', 'lengthscale', 'class_props', 'wass_before', 'wass_after',
            'dkl_before', 'dkl_after']

    # Create empty dicts
    results_dict = {}
    for split in splits:
        results_dict[split] = {}
        for inv_temp in inv_temps:
            results_dict[split][inv_temp] = {}

    # Load data
    for split in splits:
        for inv_temp in inv_temps:
            tup = pickle.load(open("data/metrics/metrics-{}-{}-{}.p".format(ds,
                                                                            split,
                                                                            inv_temp), "rb"))
            for i in range(len(keys)):
                results_dict[split][inv_temp][keys[i]] = tup[i]

    # Cut down rewards at maximum v(S) attained if stopped early
    for split in splits:
        for inv_temp in inv_temps:
            dic = results_dict[split][inv_temp]
            for party in range(num_parties):
                mus = dic['mus'][party]
                max_mu_idx = np.argmax(mus)
                if max_mu_idx != len(mus) - 1:
                    print('{}-{}-{} party {}: max at {}, total length is {}'.format(ds, split, inv_temp, party + 1,
                                                                                    max_mu_idx, len(mus)))

    # MMD unbiased
    print("Calculating MMD unbiased")
    for split in splits:
        for inv_temp in inv_temps:
            rewards = results_dict[split][inv_temp]['rewards']
            reference_dataset = results_dict[split][inv_temp]['reference_dataset']
            party_datasets = results_dict[split][inv_temp]['party_datasets']
            reference_dataset_tens = torch.tensor(reference_dataset, device=device, dtype=torch.float32)
            ls = results_dict[split][inv_temp]['lengthscale']
            k = get_kernel('se', d, ls, device)
            mmd_unbiased_before = [0 for i in range(num_parties)]
            mmd_unbiased_after = [0 for i in range(num_parties)]
            for i in range(num_parties):
                party_dataset_tens = torch.tensor(party_datasets[i], device=device, dtype=torch.float32)
                party_dataset_with_rewards_tens = torch.tensor(np.concatenate([party_datasets[i], rewards[i]], axis=0),
                                                               device=device,
                                                               dtype=torch.float32)
                mmd_unbiased_before[i] = -mmd_neg_unbiased_batched(party_dataset_tens,
                                                                   reference_dataset_tens,
                                                                   k).item()
                mmd_unbiased_after[i] = -mmd_neg_unbiased_batched(party_dataset_with_rewards_tens,
                                                                  reference_dataset_tens,
                                                                  k).item()
            results_dict[split][inv_temp]['mmd_unbiased_before'] = mmd_unbiased_before
            results_dict[split][inv_temp]['mmd_unbiased_after'] = mmd_unbiased_after
    
    # Class imbalance
    print("Calculating class imbalance")
    for split in splits:  # Reduce all party_labels to party_data_size in case its larger
        for inv_temp in inv_temps:
            results_dict[split][inv_temp]['party_labels'] = [labels[:party_data_size] for labels in
                                                              results_dict[split][inv_temp]['party_labels']]

    for split in splits:
        for inv_temp in inv_temps:
            party_datasets = results_dict[split][inv_temp]['party_datasets']
            party_labels = results_dict[split][inv_temp]['party_labels']
            rewards = results_dict[split][inv_temp]['rewards']
            candidate_dataset = results_dict[split][inv_temp]['candidate_datasets'][0]
            candidate_labels = results_dict[split][inv_temp]['candidate_labels']

            imba_after = []
            for i in range(num_parties):
                party_dataset = party_datasets[i]
                party_label = party_labels[i]
                party_ds_with_rewards = np.concatenate([party_dataset, rewards[i]], axis=0)
                all_dataset = np.concatenate([party_dataset, candidate_dataset], axis=0)
                all_labels = np.concatenate([party_label, candidate_labels], axis=0)
                imba_after.append(class_proportion(get_classes(party_ds_with_rewards,
                                                               all_dataset,
                                                               all_labels), num_classes)[1])

            results_dict[split][inv_temp]['imba_after'] = imba_after

    # Number of rewards
    for split in splits:
        for inv_temp in inv_temps:
            rewards = results_dict[split][inv_temp]['rewards']
            results_dict[split][inv_temp]['num_rewards'] = [len(rewards[i]) for i in range(len(rewards))]

    # Show alpha
    print("======= Alpha =======")
    for split in splits:
        print("Split: {}".format(split))
        print("Alpha: {}".format(results_dict[split][inv_temps[0]]['alpha']))

    # Show mean and standard deviation of correlations across inv_temps
    print("======= Mean and standard deviation of correlations across inv_temps =======")
    for split in splits:
        print("Split: {}".format(split))
        all_wass = []
        all_dkl = []
        all_num_rewards = []
        all_mmd_u = []
        all_imba = []
        for inv_temp in inv_temps:
            alpha = results_dict[split][inv_temp]['alpha']
            wass = results_dict[split][inv_temp]['wass_after']
            dkl = results_dict[split][inv_temp]['dkl_after']
            num_rewards = results_dict[split][inv_temp]['num_rewards']
            mmd_u = results_dict[split][inv_temp]['mmd_unbiased_after']
            imba = results_dict[split][inv_temp]['imba_after']

            all_wass.append(np.corrcoef(alpha, wass)[0, 1])
            all_dkl.append(np.corrcoef(alpha, dkl)[0, 1])
            all_num_rewards.append(np.corrcoef(alpha, num_rewards)[0, 1])
            all_mmd_u.append(np.corrcoef(alpha, mmd_u)[0, 1])
            all_imba.append(np.corrcoef(alpha, imba)[0, 1])

        results_dict[split]['all_wass'] = all_wass
        results_dict[split]['all_dkl'] = all_dkl
        results_dict[split]['all_num_rewards'] = all_num_rewards
        results_dict[split]['all_mmd_u'] = all_mmd_u
        results_dict[split]['all_imba'] = all_imba

        print("MMD: {} +/- {} \n DKL: {} +/- {} \n Wass: {} +/- {} \n Class imbalance: {} +/- {} \n num rewards: {} +/- {} |".format(
            np.mean(all_mmd_u),
            stats.sem(all_mmd_u),
            np.mean(all_dkl),
            stats.sem(all_dkl),
            np.mean(all_wass),
            stats.sem(all_wass),
            np.mean(all_imba),
            stats.sem(all_imba),
            np.mean(all_num_rewards),
            stats.sem(all_num_rewards)))
        print("========")

    # Show correlation between inv_temp and number of rewards / MMD unbiased
    print("======= Correlation between inv_temp and number of rewards / MMD unbiased =======")
    for split in splits:
        print("Split: {}".format(split))
        corrs_num_rewards = []
        corrs_mmd = []
        for i in range(num_parties):
            # print("Party {}".format(i))
            num_rewards_greeds = []
            mmd_unbiased_greeds = []
            for inv_temp in inv_temps:
                num_rewards_greeds.append(results_dict[split][inv_temp]['num_rewards'][i])
                mmd_unbiased_greeds.append(results_dict[split][inv_temp]['mmd_unbiased_after'][i])
            corrs_num_rewards.append(np.corrcoef(inv_temps, num_rewards_greeds)[0, 1])
            corrs_mmd.append(np.corrcoef(inv_temps, mmd_unbiased_greeds)[0, 1])
        print("Correlation between inv_temps and num_rewards mean and std err: {}, {}".format(np.mean(corrs_num_rewards),
                                                                                           stats.sem(
                                                                                               corrs_num_rewards)))
        print("Correlation between inv_temps and mmd_unbiased mean and std err: {}, {}".format(np.mean(corrs_mmd),
                                                                                            stats.sem(corrs_mmd)))
        print("========")
