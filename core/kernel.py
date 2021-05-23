import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from core.mmd import mmd_neg_unbiased, mmd_neg_unbiased_batched
from core.reward_calculation import get_v, get_v_is


class SEKernel:
    """
    Custom squared exponential kernel parameterized by inverse lengthscale, with variance term = 1
    """
    def __init__(self, ard_num_dims, inv_lengthscale_squared, device):
        self.inv_ls_squared = torch.tensor([inv_lengthscale_squared for _ in range(ard_num_dims)], device=device,
                                           requires_grad=True, dtype=torch.float32)
        self.ard_num_dims = ard_num_dims
        self.device = device

    def __call__(self, X, Y=None):
        """
        :param X: torch tensor of size (m, d)
        :param Y: torch tensor of size (n, d)
        :return: lazy tensor of size (m, n)
        """
        if Y is None:
            Y = X

        diff_squared = torch.square(torch.unsqueeze(X, 1) - Y)  # tensor of shape (m, n, d)
        exponent = torch.matmul(diff_squared, self.inv_ls_squared)  # tensor of shape (m, n)
        return torch.exp(-0.5 * exponent)

    def parameters(self):
        return [self.inv_ls_squared]

    def set_inv_ls_squared_scalar(self, inv_ls):
        self.inv_ls_squared = torch.tensor([inv_ls for _ in range(self.ard_num_dims)], device=self.device,
                                           requires_grad=True, dtype=torch.float32)

    def set_inv_ls_squared(self, inv_ls_squared):
        self.inv_ls_squared = torch.tensor(inv_ls_squared, device=self.device, requires_grad=True, dtype=torch.float32)


def median_heuristic(input, num_samples=5000):
    """
    :param input: array of shape (n, d)
    """
    idxs = np.random.permutation(len(input))[:num_samples]
    data = input[idxs]

    n, d = data.shape
    norms = np.zeros(n*(n-1)//2)
    idx = 0
    for i in tqdm(range(len(data))):
        current_norms = np.linalg.norm(data[i:i+1] - data[i+1:], axis=1)
        norms[idx:idx+len(current_norms)] = current_norms
        idx = idx + len(current_norms)
    return np.median(norms)


def nonneg_lb(n, S, k, eta=None, tol=1e-04):
    """
    Lower bound required for non-negativity of v(S).
    :param n: size of reference dataset Y
    :param S: minimum party dataset size
    :param k: value of diagonal terms (usually 1)
    :param eta: upper bound (if none, set to k)
    :return: scalar
    """
    if eta is None:
        eta = k

    return (n - 2 * S) / (2 * S * (n - S)) * (k + (S - 1) * eta) - tol


def is_all_above_lb(k, val_points, lb):
    num_above = (k(val_points).cpu().detach().numpy() > lb).sum()
    return num_above == len(val_points) ** 2


def is_pareto_efficient(costs, return_mask=True):
    """
    Find pareto-efficient points. Defined in terms of costs, so returns lowest values. From accepted answer in
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    with tqdm(total=len(costs)) as pbar:
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
            pbar.set_postfix(num_is_efficient=len(is_efficient))
            pbar.update(1)
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_kernel(kernel_name, d, lengthscale, device):
    if kernel_name == 'se':
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        kernel.lengthscale = [lengthscale for _ in range(d)]
        kernel.to(device)
    elif kernel_name == 'se_sum':
        kernel01 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel01.base_kernel.lengthscale = [0.1*lengthscale for _ in range(d)]
        kernel01.outputscale = 1
        kernel1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel1.base_kernel.lengthscale = [1 * lengthscale for _ in range(d)]
        kernel1.outputscale = 1
        kernel10 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel10.base_kernel.lengthscale = [10 * lengthscale for _ in range(d)]
        kernel10.outputscale = 1
        kernel = kernel01 + kernel1
        kernel.kernels.append(kernel10)
    elif kernel_name == 'rq':
        rq01 = gpytorch.kernels.RQKernel()
        rq01.alpha = 0.1
        rq1 = gpytorch.kernels.RQKernel()
        rq1.alpha = 1.
        rq10 = gpytorch.kernels.RQKernel()
        rq10.alpha = 10.
        kernel = rq01 + rq1
        kernel.kernels.append(rq10)
    else:
        raise Exception("Kernel name must be 'se' or 'se_sum' or 'rq'")

    return kernel


def is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
    v = get_v(party_datasets, reference_dataset, k, device=device, batch_size=batch_size)
    v_is = get_v_is(v, len(party_datasets))
    return np.all(np.array(v_is) >= 0)


def optimize_kernel_binsearch_only(k, device, party_datasets, reference_dataset, low=0.01, high=1000, num_iters=20, batch_size=256):
    """
    Does a binary search over lengthscales for minimum value that gives non-negative party dataset values
    :param k:
    :param device:
    :param party_datasets:
    :param reference_dataset:
    :return:
    """
    d = reference_dataset.shape[-1]

    k.lengthscale = [high for _ in range(d)]
    if not is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
        raise Exception("High value of lengthscale is already invalid")

    k.lengthscale = [low for _ in range(d)]
    if is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
        raise Exception("Low value of lengthscale is still valid, can be pushed lower")

    for _ in tqdm(range(num_iters)):
        mid = (high + low) / 2
        k.lengthscale = [mid for _ in range(d)]
        if is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
            high = mid
        else:
            low = mid

    k.lengthscale = [high for _ in range(d)]
    print("Optimal lengthscale: {}".format(high))

    return k, high


def binary_search_ls(lengthscales, device, party_datasets, reference_dataset, high=100, low=1, num_iters=10, batch_size=128):
    """
    Searches for minimum factor to multiply lengthscales by in order to be valid
    :param lengthscale: d length numpy array of lengthscales
    :param device:
    :param party_datasets:
    :param reference_dataset:
    :return:
    """
    d = reference_dataset.shape[-1]
    k = gpytorch.kernels.RBFKernel(ard_num_dims=d)
    k.to(device)

    k.lengthscale = lengthscales * high
    if not is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
        raise Exception("High value of lengthscale is already invalid")

    for _ in tqdm(range(num_iters)):
        mid = (high + low) / 2
        k.lengthscale = lengthscales * mid
        if is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
            high = mid
        else:
            low = mid

    return lengthscales * high


def optimize_kernel(k, device, party_datasets, reference_dataset, num_epochs=30, batch_size=128, patience=8):
    """

    :param k:
    :param device:
    :param party_datasets:
    :param reference_dataset:
    :return:
    """
    # Data setup
    S = np.min([len(ds) for ds in party_datasets])
    train_test_split_idx = int(0.6 * S)
    party_ds_size = train_test_split_idx
    num_parties = len(party_datasets)

    party_datasets_tens = torch.tensor(party_datasets[:, :train_test_split_idx], device=device, dtype=torch.float32)
    reference_dataset_tens = torch.tensor(reference_dataset, device=device, dtype=torch.float32)
    party_datasets_test = torch.tensor(party_datasets[:, train_test_split_idx:], device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(k.parameters(), lr=0.1)
    averages = []
    best_idx = 0

    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        print("========= Test -MMD unbiased ===========")
        stats = []
        for i in range(num_parties):
            stat = mmd_neg_unbiased_batched(party_datasets_test[i], reference_dataset_tens, k).cpu().detach().numpy()
            print("Party {}: {}".format(i + 1, stat))
            stats.append(stat)
        avg = np.mean(stats)
        print("Average: {}".format(avg))

        # Code for early termination if no improvement after patience number of epochs
        averages.append(avg)
        if avg <= averages[best_idx]:
            best_idx = epoch  # Low is better for this
        elif avg > averages[best_idx] and epoch - best_idx >= patience:
            print("No improvement for {} epochs, terminating early".format(patience))
            break

        print("========= Kernel parameters ===========")
        print("lengthscale: {}".format(k.lengthscale))

        for i in range(party_ds_size // batch_size):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            loss = 0

            idx = (i + 1)
            next_m = np.min([idx * batch_size, party_ds_size])
            m = i * batch_size

            ref_idx = np.random.randint(0, len(reference_dataset) - batch_size)
            next_ref_idx = ref_idx + batch_size

            for party in range(num_parties):
                loss += mmd_neg_unbiased(party_datasets_tens[party][m:next_m],
                                         reference_dataset_tens[ref_idx:next_ref_idx],
                                         k)

            # Calc loss and backprop gradients
            loss.backward()
            optimizer.step()

        if not is_all_lb_positive(k, party_datasets, reference_dataset, device, batch_size):
            # "Project" lengthscale back to valid range
            print("Projecting lengthscales back to valid range")
            valid_lengthscales = binary_search_ls(k.lengthscale.cpu().detach().numpy(), device, party_datasets, reference_dataset)
            print("Found valid lengthscales: {}".format(valid_lengthscales))
            k.lengthscale = valid_lengthscales
        else:
            print("All lower bounds still positive")

    return k
