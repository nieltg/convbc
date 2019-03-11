import numpy as np
import pytest

from .. import keygen


def test_flatten2d_without_key():
    with pytest.raises(TypeError):
        keygen.flatten2d()


def test_expand_key_without_seed():
    with pytest.raises(TypeError):
        keygen.expand_key()


def test_expand_key_with_key_odd_last_dim_1d_array():
    key = np.zeros((1, ), dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.expand_key(key)


def test_expand_key_with_key_odd_last_dim_2d_array():
    key = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.expand_key(key)


def test_expand_key_with_key_even_last_dim_1d_array():
    key = np.zeros((2, ), dtype=np.uint8)
    assert keygen.expand_key(key).shape == (24, 16)


def test_expand_key_with_key_non_uint8_1d_array():
    key = np.zeros((2, ), dtype=np.float)
    with pytest.raises(Exception):
        keygen.expand_key(key)


def test_expand_key_with_n_neg_1_number():
    key = np.zeros((2, ), dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.expand_key(key, -1)


def test_expand_key_with_n_neg_2_number():
    key = np.zeros((2, ), dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.expand_key(key, -2)


def test_expand_key_with_n_zero():
    key = np.zeros((2, ), dtype=np.uint8)
    assert keygen.expand_key(key, 0).shape == (0, 16)


def test_expand_key_with_key_2d_array_and_n_1():
    key = np.zeros((
        1,
        2,
    ), dtype=np.uint8)
    assert keygen.expand_key(key, 1).shape == (1, 1, 16)
