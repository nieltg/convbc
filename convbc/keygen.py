import numpy as np


def KeyGenerator(seed):
    if seed.ndim != 1 or seed.dtype != np.uint8:
        raise Exception()
    return object()
