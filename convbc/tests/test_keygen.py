import numpy as np
import pytest

from .. import keygen


def test_key_generator_without_seed():
    with pytest.raises(TypeError):
        keygen.KeyGenerator()

def test_key_generator_with_multidimensional_ndarray_as_seed():
    seed = np.arange(2, dtype=np.uint8).reshape(-1, 1)
    with pytest.raises(Exception):
        keygen.KeyGenerator(seed)
