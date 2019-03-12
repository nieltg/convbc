import numpy as np
import hashlib

from joblib import Parallel, delayed


def flatten2d(key):
    if len(key.shape) < 2:
        return key.reshape(1, *key.shape)
    else:
        return key.reshape(np.prod(key.shape[:-1]), key.shape[-1])


def hash(flat_key):
    n_mid = len(flat_key) // 2
    key_data = flat_key.tobytes()

    hash1 = hashlib.new('md4', key_data[:n_mid]).digest()
    hash2 = hashlib.new('md4', key_data[n_mid:]).digest()

    hash_data = hash1 + hash2[:9]
    hash_kernel = hash2[-4:]

    return (hash_data, hash_kernel)


def hash_all(flat_keys):
    length = len(flat_keys)

    hashes_data = np.empty((length, 25), dtype=np.uint8)
    hashes_kernel = np.empty((length, 4), dtype=np.uint8)

    for i in range(0, length):
        hash_data, hash_kernel = hash(flat_keys[i])

        hashes_data[i] = np.frombuffer(hash_data, dtype=np.uint8)
        hashes_kernel[i] = np.frombuffer(hash_kernel, dtype=np.uint8)

    return (hashes_data, hashes_kernel)


convo_data_shape = (5, 5)
convo_kernel_shape = (2, 2)

convo_window_shape = np.subtract(convo_data_shape, convo_kernel_shape) + 1
convo_view_shape = tuple(convo_window_shape) + convo_kernel_shape


def build_window_blocks(hashes_data):
    data_blocks = hashes_data.reshape(-1, *convo_data_shape)

    # Reference: https://stackoverflow.com/a/43087507
    view_shape = (len(hashes_data), ) + convo_view_shape
    view_strides = data_blocks.strides + data_blocks.strides[1:]

    return np.lib.stride_tricks.as_strided(
        data_blocks, view_shape, view_strides, writeable=False)


def convo2d(hashes_data, hashes_kernel):
    kernel_blocks = hashes_kernel.reshape(-1, 1, 1, *convo_kernel_shape)
    data_window_blocks = build_window_blocks(hashes_data)

    out = np.sum(
        kernel_blocks * data_window_blocks, axis=(3, 4), dtype=np.uint8)
    return out.reshape(-1, 16)


def calculate_padding(n, n_max=7):
    trend = n // n_max
    unit = n % n_max

    if trend % 2 == 0:
        return unit
    else:
        return n_max - unit


padding_out_size = np.prod(convo_data_shape)
padding_n_max = np.prod(convo_data_shape) - np.prod(convo_window_shape)


def pad(convo_values, n):
    n_pad = calculate_padding(n, padding_n_max)
    n_pad_end = padding_out_size - convo_values.shape[1] - n_pad

    return np.pad(convo_values, ((0, 0), (n_pad, n_pad_end)), mode='constant')


def expand_key(key, n=24):
    if key.shape[-1] % 2 != 0 or key.dtype != np.uint8:
        raise Exception()

    side_shape = key.shape[:-1]

    data_blocks, kernel_blocks = hash_all(flatten2d(key))
    out = np.empty((n, np.prod(side_shape + (1, )), 16), dtype=np.uint8)

    for i in range(n):
        convo_values = convo2d(data_blocks, kernel_blocks)
        out[i] = convo_values

        data_blocks = data_blocks ^ pad(convo_values, n)

    return np.transpose(out, axes=(1, 0, 2)).reshape(side_shape + (n, 16))
