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

def _test_iterator_produces_n_values(iter, n):
    i = 0
    for key in iter:
        i = i + 1
        assert i <= n
    assert i == n

def test_key_generator_with_seed():
    seed = np.arange(2, dtype=np.uint8)
    assert keygen.KeyGenerator(seed) is not None

def _test_key_generator_with_seed_and_length(length, n):
    seed = np.arange(2, dtype=np.uint8)
    _test_iterator_produces_n_values(keygen.KeyGenerator(seed, length), n)

def test_key_generator_with_seed_and_negative_length():
    _test_key_generator_with_seed_and_length(-1, n=0)

def test_key_generator_with_seed_and_length_0():
    _test_key_generator_with_seed_and_length(0, n=0)

def test_key_generator_with_seed_and_length_1():
    _test_key_generator_with_seed_and_length(1, n=1)
