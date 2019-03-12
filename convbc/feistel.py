import numpy as np

def split_plain_block(plain_block):
    n = len(plain_block)
    return plain_block[:n//2], plain_block[-n//2:]


def combine_plain_block(plain_L, plain_R):
    return np.concatenate((plain_L, plain_R), axis=0)


def feistel_network(func, plain_block, round_keys):
    n = round_keys.shape[0]
    plain_L, plain_R = split_plain_block(plain_block)
    for i in range(n):
        tmp = plain_R
        plain_R = plain_L ^ func(round_keys[i], plain_R)
        plain_L = tmp
    cipher_block = combine_plain_block(plain_R, plain_L)
    return cipher_block
