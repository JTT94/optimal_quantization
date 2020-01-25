import torch
import numpy as np


def uniform_weights(n, dtype=torch.float64):
    weights = np.repeat(1/n, n)
    return torch.tensor(weights, dtype=dtype)
