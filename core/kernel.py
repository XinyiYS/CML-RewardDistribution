import gpytorch


def get_kernel(dataset, d):
    if dataset == 'GMM':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel.base_kernel.lengthscale = [0.05, 0.05]
        kernel.outputscale = 1

    elif dataset == 'MNIST' or 'CIFAR':
        rq01 = gpytorch.kernels.RQKernel()
        rq01.alpha = 0.1
        rq1 = gpytorch.kernels.RQKernel()
        rq1.alpha = 1.
        rq10 = gpytorch.kernels.RQKernel()
        rq10.alpha = 10.
        lin = gpytorch.kernels.LinearKernel()
        lin.variance = 1.
        kernel = rq01 + rq1 + rq10 + lin

    else:
        raise Exception("Parameter dataset must be 'GMM', 'MNIST', or 'CIFAR'")

    return kernel
