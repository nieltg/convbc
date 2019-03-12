import numpy as np


def split_plain_block(plain_block):
    n = plain_block.shape[1]
    return plain_block[:, :n // 2], plain_block[:, -n // 2:]


def combine_plain_block(plain_L, plain_R):
    return np.concatenate((plain_L, plain_R), axis=1)


def feistel_network(func, plain_block, round_keys):
    n = round_keys.shape[0]
    plain_L, plain_R = split_plain_block(plain_block)
    for i in range(n):
        tmp = plain_R
        plain_R = plain_L ^ func(round_keys[i], plain_R)
        plain_L = tmp
    cipher_block = combine_plain_block(plain_R, plain_L)
    return cipher_block


def inverse_feistel_network(func, plain_block, round_keys):
    n = round_keys.shape[0]
    plain_R, plain_L = split_plain_block(plain_block)
    for i in range(n):
        tmp = plain_L
        plain_L = plain_R ^ func(round_keys[n - i - 1], plain_L)
        plain_R = tmp
    cipher_block = combine_plain_block(plain_L, plain_R)
    return cipher_block
