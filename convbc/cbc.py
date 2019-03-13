import numpy as np

from . import feistel
from . import feistel_parallel
from .keygen import expand_key
from .encrypt import split_blocks, f


def build_iv(key):
    trimmed_key = key[:32]
    return np.pad(trimmed_key, (0, 32 - len(trimmed_key)), mode='constant')


def encrypt(data, key):
    expanded_key = expand_key(key)

    blocks = np.copy(split_blocks(data))
    prev_block = build_iv(key)

    for block in blocks:
        block[:] = feistel_parallel.feistel_network(f, (block ^ prev_block)[np.newaxis,:], expanded_key)[0]
        prev_block = block
    return blocks.reshape(-1)


def decrypt(data, key):
    expanded_key = expand_key(key)

    blocks = split_blocks(data)
    xor_blocks = np.insert(blocks[:-1], 0, build_iv(key), axis=0)

    out = feistel_parallel.inverse_feistel_network(f, blocks, expanded_key)
    return (out ^ xor_blocks).reshape(-1)
