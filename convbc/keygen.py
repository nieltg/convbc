import numpy as np


def expand_key(key):
    if key.shape[-1] % 2 != 0:
        raise Exception()
    return np.zeros((24, 16))
