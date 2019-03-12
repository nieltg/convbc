import numpy as np
import hashlib

from joblib import Parallel


md4 = hashlib.new('md4')


def flatten2d(key):
    if len(key.shape) < 2:
        return key.reshape(1, *key.shape)
    else:
        return key.reshape(np.prod(key.shape[:-1]), key.shape[-1])


def hash(flat_key, mid):
    key_data = flat_key.tobytes()

    md4.update(key_data[:mid])
    hash1 = md4.digest()
    md4.update(key_data[mid:])
    hash2 = md4.digest()

    hash_data = hash1 + hash2[:9]
    hash_kernel = hash2[-4:]

    return (hash_data, hash_kernel)


def hash_all(flat_keys):
    mid = flat_keys.shape[-1] // 2

    joblib.Parallel()

    for flat_key in flat_keys:
        pass


def expand_key(key, n=24):
    if key.shape[-1] % 2 != 0 or key.dtype != np.uint8:
        raise Exception()

    return np.zeros(key.shape[:-1] + (n, 16))
