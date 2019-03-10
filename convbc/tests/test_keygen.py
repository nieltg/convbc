import pytest

from .. import keygen

def test_key_generator_without_seed():
    with pytest.raises(TypeError):
        keygen.KeyGenerator()
