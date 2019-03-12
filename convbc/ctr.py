import numpy as np

from .keygen import expand_key
from .feistel_parallel import feistel_network, inverse_feistel_network
from .encrypt import split_blocks, f


def build_keys(key, blocks_len):
    counter = np.arange(blocks_len, dtype='<u8').tobytes()
    counter_blocks = np.frombuffer(counter, dtype=np.uint8).reshape(-1, 8)

    repeated_keys = np.tile(key, (blocks_len, 1))
    return np.concatenate((repeated_keys, counter_blocks), axis=1)


def encrypt(data, key):
    blocks = split_blocks(data)
    expanded_key = expand_key(build_keys(key, len(blocks)))

    data = feistel_network(f, blocks, expanded_key)
    return data.reshape(-1)


def decrypt(data, key):
    blocks = split_blocks(data)
    expanded_key = expand_key(build_keys(key, len(blocks)))

    data = inverse_feistel_network(f, split_blocks(data), expanded_key)
    return data.reshape(-1)
