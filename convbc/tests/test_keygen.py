import numpy as np
import pytest

from .. import keygen


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


def test_expand_key_with_n_negative_number():
    key = np.zeros((2, ), dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.expand_key(key, -1)
