import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from core.mmd import mmd_neg_unbiased


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


def optimize_kernel(kernel, device, party_datasets, reference_dataset, num_epochs=50, batch_size=256):
    optimizer = torch.optim.Adam(kernel.parameters(), lr=0.1)

    party_ds_size = len(party_datasets[0])
    num_parties = len(party_datasets)
    party_datasets_tens = torch.tensor(party_datasets, device=device)
    reference_dataset_tens = torch.tensor(reference_dataset, device=device)

    for epoch in range(num_epochs):
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
                                         kernel)

            # Calc loss and backprop gradients
            loss.backward()
            optimizer.step()

    return kernel
