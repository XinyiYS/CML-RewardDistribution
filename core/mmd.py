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
    k.to(device)

    with torch.no_grad():
        # first batch
        S_XY = (2 / (batch_size * n)) * torch.sum(k(X_tens[:batch_size], Y_tens).evaluate())
        S_X = (1 / (batch_size ** 2)) * torch.sum(k(X_tens[:batch_size]).evaluate())

        for i in range(max_m // batch_size):
            idx = i + 2
            next_m = np.min([idx * batch_size, max_m])
            m = (idx - 1) * batch_size
            S_XY = (m * S_XY + (2 / n) * torch.sum(k(X_tens[m:next_m], Y_tens).evaluate())) / next_m
            S_X = ((m ** 2) * S_X + 2 * torch.sum(k(X_tens[m:next_m], X_tens[:m]).evaluate()) +
                   torch.sum(k(X_tens[m:next_m]).evaluate())) / (next_m ** 2)

    return (S_XY - S_X).item(), S_X.item(), S_XY.item()
