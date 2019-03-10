import numpy as np


def expand_key(seed, length=24):
    len_dim0, = seed.shape
    if len_dim0 % 2 != 0 or seed.dtype != np.uint8:
        raise Exception()
    return [None] * length
