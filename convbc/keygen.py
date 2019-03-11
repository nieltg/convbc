import numpy as np


def expand_key(key, n=24):
    if key.shape[-1] % 2 != 0 or key.dtype != np.uint8:
        raise Exception()
    return np.zeros((n, 16))
