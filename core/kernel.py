import gpytorch


def get_kernel(dataset, d):
    if dataset == 'GMM':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel.base_kernel.lengthscale = [0.05, 0.05]
        kernel.outputscale = 1

    elif dataset == 'MNIST':
        kernel = None

    elif dataset == 'CIFAR':
        kernel = None

    else:
        raise Exception("Parameter dataset must be 'GMM', 'MNIST', or 'CIFAR'")

    return kernel
