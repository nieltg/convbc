import numpy as np

from .keygen import expand_key
from .feistel_parallel import feistel_network, inverse_feistel_network
from .encrypt import split_blocks, f


def encrypt(data, key):
    expanded_key = expand_key(key)

    data = feistel_network(f, split_blocks(data), expanded_key)
    return data.reshape(-1)


def decrypt(data, key):
    expanded_key = expand_key(key)

    data = inverse_feistel_network(f, split_blocks(data), expanded_key)
    return data.reshape(-1)
