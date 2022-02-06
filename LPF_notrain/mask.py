import numpy as np


def secret_seed(seed, a, b):
    np.random.seed(seed)
    if b != 0:
        return np.random.rand(a, b)
    else:
        return np.random.rand(a,)
