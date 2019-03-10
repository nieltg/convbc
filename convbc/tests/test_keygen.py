import numpy as np
import pytest

from .. import keygen


def test_key_generator_without_seed():
    with pytest.raises(TypeError):
        keygen.KeyGenerator()

def test_key_generator_with_odd_lengthed_ndarray_as_seed():
    seed = np.arange(1, dtype=np.uint8)
    with pytest.raises(Exception):
        keygen.KeyGenerator(seed)

def test_key_generator_with_multidimensional_ndarray_as_seed():
    seed = np.arange(2, dtype=np.uint8).reshape(-1, 1)
    with pytest.raises(Exception):
        keygen.KeyGenerator(seed)

def test_key_generator_with_non_uint8_dtyped_ndarray_as_seed():
    seed = np.arange(2, dtype=np.float)
    with pytest.raises(Exception):
        keygen.KeyGenerator(seed)

def test_key_generator_with_seed():
    seed = np.arange(2, dtype=np.uint8)
    assert keygen.KeyGenerator(seed) is not None

def test_key_generator_with_seed_and_negative_length():
    seed = np.arange(2, dtype=np.uint8)
    length = -1

    i, i_max = 0, max(0, length)
    for key in keygen.KeyGenerator(seed, length):
        i = i + 1
        assert i <= i_max
    assert i == i_max
