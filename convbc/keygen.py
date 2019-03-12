import numpy as np
import hashlib

from joblib import Parallel, delayed

md4 = hashlib.new('md4')


def flatten2d(key):
    if len(key.shape) < 2:
        return key.reshape(1, *key.shape)
    else:
        return key.reshape(np.prod(key.shape[:-1]), key.shape[-1])


def hash(flat_key):
    n_mid = len(flat_key) // 2
    key_data = flat_key.tobytes()

    md4.update(key_data[:n_mid])
    hash1 = md4.digest()
    md4.update(key_data[n_mid:])
    hash2 = md4.digest()

    hash_data = hash1 + hash2[:9]
    hash_kernel = hash2[-4:]

    return (hash_data, hash_kernel)


def hash_all(flat_keys):
    length = len(flat_keys)

    hashes_data = np.empty((length, 25))
    hashes_kernel = np.empty((length, 4))

    for i in range(0, length):
        hash_data, hash_kernel = hash(flat_keys[i])

        hashes_data[i] = np.frombuffer(hash_data, dtype=np.uint8)
        hashes_kernel[i] = np.frombuffer(hash_kernel, dtype=np.uint8)

    return (hashes_data, hashes_kernel)


convo_data_shape = (5, 5)
convo_kernel_shape = (2, 2)

convo_view_shape = tuple(
    np.subtract(convo_data_shape, convo_kernel_shape) + 1) + convo_kernel_shape


def build_window_blocks(hashes_data):
    length = len(hashes_data)
    data_blocks = hashes_data.reshape(length, *convo_data_shape)

    view_shape = (length, ) + convo_view_shape
    view_strides = data_blocks.strides + data_blocks.strides[1:]

    return np.lib.stride_tricks.as_strided(
        data_blocks, view_shape, view_strides, writeable=False)


def expand_key(key, n=24):
    if key.shape[-1] % 2 != 0 or key.dtype != np.uint8:
        raise Exception()

    return np.zeros(key.shape[:-1] + (n, 16))
