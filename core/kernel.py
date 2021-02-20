import numpy as np
import gpytorch
from tqdm import tqdm


def median_heuristic(data):
    """
    :param data: array of shape (n, d)
    """
    n, d = data.shape
    norms = np.zeros(n*(n-1)//2)
    idx = 0
    for i in tqdm(range(len(data))):
        current_norms = np.linalg.norm(data[i:i+1] - data[i+1:], axis=1)
        norms[idx:idx+len(current_norms)] = current_norms
        idx = idx + len(current_norms)
    return np.median(norms)


def get_kernel(kernel_name, d, lengthscale):
    if kernel_name == 'se':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel.base_kernel.lengthscale = [lengthscale for _ in range(d)]
        kernel.outputscale = 1

    elif kernel_name == 'rq':
        rq01 = gpytorch.kernels.RQKernel()
        rq01.alpha = 0.1
        rq1 = gpytorch.kernels.RQKernel()
        rq1.alpha = 1.
        rq10 = gpytorch.kernels.RQKernel()
        rq10.alpha = 10.
        kernel = rq01 + rq1
        kernel.kernels.append(rq10)
        for k in kernel.kernels:
            k.raw_alpha.requires_grad = False
            k.raw_lengthscale.requires_grad = False

    else:
        raise Exception("Kernel name must be 'se' or 'rq'")

    return kernel
