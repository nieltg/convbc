import numpy as np


def KeyGenerator(seed):
    len_dim0, = seed.shape
    if len_dim0 % 2 != 0 or seed.dtype != np.uint8:
        raise Exception()
    return object()
