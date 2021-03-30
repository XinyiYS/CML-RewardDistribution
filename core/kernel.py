import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from core.mmd import mmd_neg_unbiased, mmd_neg_unbiased_batched
from scipy.optimize import linprog


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
        kernel = SEKernel(d, lengthscale, device)
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
        # for k in kernel.kernels:
        #     k.raw_alpha.requires_grad = False
        #     k.raw_lengthscale.requires_grad = False
    else:
        raise Exception("Kernel name must be 'se' or 'se_sum' or 'rq'")

    return kernel


def optimize_kernel(k, device, party_datasets, reference_dataset, num_epochs=50,
                    batch_size=128, num_val_points=2000):
    # Data setup
    n = len(reference_dataset)
    S = np.min([len(ds) for ds in party_datasets])
    d = reference_dataset.shape[-1]
    train_test_split_idx = int(0.6 * S)
    party_ds_size = train_test_split_idx
    num_parties = len(party_datasets)

    party_datasets_tens = torch.tensor(party_datasets[:, :train_test_split_idx], device=device, dtype=torch.float32)
    reference_dataset_tens = torch.tensor(reference_dataset, device=device, dtype=torch.float32)
    party_datasets_test = torch.tensor(party_datasets[:, train_test_split_idx:], device=device, dtype=torch.float32)

    # Get lower bound required for non-negativity of v(S)
    lb = nonneg_lb(n, S, 1)

    # Select num_val_points random points to check k(x_i, x_j) > lb
    val_points = torch.tensor(reference_dataset[np.random.permutation(np.arange(num_val_points))], device=device,
                              dtype=torch.float32)
    val_points_np = val_points.cpu().numpy()

    # Get Pareto frontier of squared differences
    squared_diffs = np.square(np.expand_dims(val_points_np, 1) - val_points_np)  # (m, m, d)
    squared_diffs = np.reshape(squared_diffs, [-1, d])
    squared_diff_idxs = \
    np.where((np.triu(np.ones((num_val_points, num_val_points))) - np.diag(np.ones(num_val_points))).flatten())[0]
    squared_diffs_reduced = squared_diffs[squared_diff_idxs]
    print("Calculating Pareto optimal differences")
    reduced_D = squared_diffs_reduced[is_pareto_efficient(-squared_diffs_reduced)]

    # Upper bound for linear program in Frank-Wolfe algorithm
    b = (-2 * np.log(lb)) * np.ones(len(reduced_D), dtype=np.float32)

    # Frank-Wolfe conditional gradient algorithm
    optimizer = torch.optim.SGD(k.parameters(), lr=0.1)
    t = 0
    patience = 20
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

        print("========= Kernel parameters ===========")
        print("inv lengthscale squared:")
        print(k.inv_ls_squared)
        print("lengthscale:")
        print(np.sqrt(1 / k.inv_ls_squared.cpu().detach().numpy()))
        print("k still valid (all above upper bound): {}".format(is_all_above_lb(k, val_points, lb)))

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

            # change gradients to argmin x \in C <grad, x>
            grad = k.inv_ls_squared.grad.cpu().numpy()
            #print("Actual grad: {}".format(grad))
            res = linprog(grad, A_ub=reduced_D, b_ub=b, method='interior-point')
            y_t = res['x']
            #print("y_t: {}".format(y_t))
            #print("inv_ls_squared: {}".format(k.inv_ls_squared))

            # original conditional gradient update method
            step_size = 2 / (t + 2)
            #print("Step size: {}".format(step_size))
            k.set_inv_ls_squared((1 - step_size) * k.inv_ls_squared.cpu().detach().numpy() + step_size * y_t)
            t += 1

        # Code for early termination if no improvement after patience number of epochs
        averages.append(avg)
        if avg <= averages[best_idx]:
            best_idx = epoch  # Low is better for this
        elif avg >= averages[best_idx] and epoch - best_idx >= patience:
            print("No improvement for {} epochs, terminating early".format(patience))
            break

    return k
