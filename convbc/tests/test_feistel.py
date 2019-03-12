import numpy as np
import pytest

from .. import feistel


def test_split_plain_block_with_n_len_plain_block():
    n = 16
    plain_block = np.arange(1, n+1, 1, dtype=np.uint8)
    plain_L, plain_R = feistel.split_plain_block(plain_block)
    assert np.array_equal(plain_L,np.arange(1, 9, 1, dtype=np.uint8))
    assert np.array_equal(plain_R, np.arange(9, 17, 1, dtype=np.uint8))
    assert len(plain_L) == 8
    assert len(plain_R) == 8


def test_combine_plain_block_with_n_len_plain_block():
    plain_L = np.zeros((8,), dtype=np.uint8)
    plain_R = np.zeros((8,), dtype=np.uint8)
    plain_block = feistel.combine_plain_block(plain_L, plain_R)
    assert len(plain_block) == 16


def test_feistel_with_key_zeros():
    n = 3
    plain_block = np.arange(1, 17, 1, dtype=np.uint8)
    round_keys = np.zeros((n, 8), dtype=np.uint8)
    cipher_block = feistel.feistel_network(lambda k, r: k, plain_block, round_keys)
    assert np.array_equal(plain_block, cipher_block)
    assert len(cipher_block) == 16


def test_feistel_with_key_not_zeros():
    n = 3
    plain_block = np.arange(1, 17, 1, dtype=np.uint8)
    round_keys = np.random.randint(1,100,(8,), dtype=np.uint8)
    cipher_block = feistel.feistel_network(
        lambda k, r: k, plain_block, round_keys)
    assert not np.array_equal(plain_block, cipher_block)
    assert len(cipher_block) == 16
