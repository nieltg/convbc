import numpy as np

from .keygen import expand_key
from .feistel_parallel import feistel_network, inverse_feistel_network
from .encrypt import split_blocks, f


def build_data_blocks(key, blocks_len):
    counter = np.arange(blocks_len, dtype='<u8').tobytes()
    counter_blocks = np.frombuffer(counter, dtype=np.uint8).reshape(-1, 8)

    trimmed_key = key[:24]
    padded_key = np.pad(
        trimmed_key, (0, 24 - len(trimmed_key)), mode='constant')
    tiled_padded_key = np.tile(padded_key, (blocks_len, 1))

    return np.concatenate((tiled_padded_key, counter_blocks), axis=1)


def encrypt(data, key):
    expanded_key = expand_key(key)

    data_blocks = split_blocks(data)
    blocks = build_data_blocks(key, len(data_blocks))

    out = feistel_network(f, blocks, expanded_key)
    return (out ^ data_blocks).reshape(-1)


def decrypt(data, key):
    return encrypt(data, key)
