import pytest

from .. import keygen


def test_expand_key_without_seed():
    with pytest.raises(TypeError):
        keygen.expand_key()
