import gpytorch


def get_kernel(kernel_name, d):
    if kernel_name == 'se':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        kernel.base_kernel.lengthscale = [0.05, 0.05]
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
