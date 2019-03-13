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

    orig_blocks = split_blocks(data)
    blocks = np.copy(orig_blocks)
    next_feed = build_iv(key)

    for block in blocks:
        block[:] = feistel_parallel.feistel_network(f, (next_feed)[np.newaxis,:], expanded_key)[0]
        next_feed = block
    return (blocks ^ orig_blocks).reshape(-1)


def decrypt(data, key):
    return encrypt(data, key)
