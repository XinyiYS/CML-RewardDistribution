import torch
import numpy as np


def mmd_neg_biased(X, Y, k):
    """
    Calculates biased MMD^2 without the S_YY term, where S_X, S_XY and S_YY are the pairwise-XX, pairwise-XY, pairwise-YY
    summation terms respectively.
    :param X: array of shape (m, d)
    :param Y: array of shape (n, d)
    :param k: GPyTorch kernel
    :return: MMD^2, S_X, S_XY, S_Y
    """
    m = X.shape[0]
    n = Y.shape[0]
    X_tens = torch.tensor(X, dtype=torch.float32)
    Y_tens = torch.tensor(Y, dtype=torch.float32)

    S_X = (1 / (m ** 2)) * torch.sum(k(X_tens).evaluate())
    S_XY = (2 / (m * n)) * torch.sum(k(X_tens, Y_tens).evaluate())

    return (S_XY - S_X).item(), S_X.item(), S_XY.item()


def mmd_neg_biased_batched(X, Y, k, device, batch_size=128):
    """
    Calculates biased MMD^2 without the S_YY term, where S_X, S_XY and S_YY are the pairwise-XX, pairwise-XY, pairwise-YY
    summation terms respectively. Does so using the GPU in a batch-wise manner.
    :param X: array of shape (m, d)
    :param Y: array of shape (n, d)
    :param k: GPyTorch kernel
    :param device:
    :param batch_size:
    :return: MMD^2, S_X, S_XY
    """
    max_m = X.shape[0]
    n = Y.shape[0]

    X_tens = torch.tensor(X, device=device)
    Y_tens = torch.tensor(Y, device=device)

    with torch.no_grad():
        S_XY = 0
        S_X = 0
        for i in range((max_m // batch_size) + 1):
            idx = i + 1
            next_m = np.min([idx * batch_size, max_m])
            m = (idx - 1) * batch_size
            S_XY = (m * S_XY + (2 / n) * torch.sum(k(X_tens[m:next_m], Y_tens).evaluate())) / next_m
            S_X = ((m ** 2) * S_X + 2 * torch.sum(k(X_tens[m:next_m], X_tens[:m]).evaluate()) +
                   torch.sum(k(X_tens[m:next_m]).evaluate())) / (next_m ** 2)

    return (S_XY - S_X).item(), S_X.item(), S_XY.item()


def mmd_neg_unbiased(X, Y, k):
    """
    Used as loss function.
    :param X: Torch tensor
    :param Y: Torch tensor
    :param k: GPyTorch kernel
    :return: scalar
    """
    m = X.size(0)
    n = Y.size(0)

    S_X = (1 / (m * (m-1))) * (torch.sum(k(X).evaluate()) - torch.sum(torch.diag(k(X).evaluate())))
    S_XY = (2 / (m * n)) * torch.sum(k(X, Y).evaluate())
    S_Y = (1 / (n * (n-1))) * (torch.sum(k(Y).evaluate()) - torch.sum(torch.diag(k(Y).evaluate())))

    return S_XY - S_X - S_Y


def mmd_neg_unbiased_batched(X, Y, k, batch_size=128):
    max_m = X.size(0)
    max_n = Y.size(0)

    with torch.no_grad():
        S_XY = 0
        S_X = 0
        S_Y = 0
        for i in range((max_m // batch_size) + 1):
            idx = i + 1
            next_m = np.min([idx * batch_size, max_m])
            m = (idx - 1) * batch_size
            S_XY = (m * S_XY + (2 / max_n) * torch.sum(k(X[m:next_m], Y).evaluate())) / next_m
            S_X = ((m * (m-1)) * S_X + 2 * torch.sum(k(X[m:next_m], X[:m]).evaluate()) +
                   torch.sum(k(X[m:next_m]).evaluate()) - torch.sum(torch.diag(k(X[m:next_m]).evaluate()))) / (next_m * (next_m-1))

        for i in range((max_n // batch_size) + 1):
            idx = i + 1
            next_n = np.min([idx * batch_size, max_n])
            n = (idx - 1) * batch_size
            S_Y = ((n * (n-1)) * S_Y + 2 * torch.sum(k(Y[n:next_n], Y[:n]).evaluate()) +
                   torch.sum(k(Y[n:next_n]).evaluate()) - torch.sum(torch.diag(k(Y[n:next_n]).evaluate()))) / (next_n * (next_n-1))

    return S_XY - S_X - S_Y
