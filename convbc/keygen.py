import numpy as np
import hashlib


def flatten2d(key):
    if len(key.shape) < 2:
        return key.reshape(1, *key.shape)
    else:
        return key.reshape(np.prod(key.shape[:-1]), key.shape[-1])


def expand_key(key, n=24):
    if key.shape[-1] % 2 != 0 or key.dtype != np.uint8:
        raise Exception()

    return np.zeros(key.shape[:-1] + (n, 16))
