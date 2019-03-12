import numpy as np
import pytest

from .. import keygen


def test_flatten2d_without_key():
    with pytest.raises(TypeError):
        keygen.flatten2d()


def test_flatten2d_with_key_1d_array():
    key = np.zeros((1, ))
    assert np.array_equal(keygen.flatten2d(key), key.reshape(1, 1))


def test_flatten2d_with_key_1d_array_shape_2():
    key = np.zeros((2, ))
    assert np.array_equal(keygen.flatten2d(key), key.reshape(1, 2))


def test_flatten2d_with_key_2d_array():
    key = np.zeros((1, 2))
    assert np.array_equal(keygen.flatten2d(key), key)


def test_flatten2d_with_key_3d_array():
    key = np.zeros((1, 2, 3))
    assert keygen.flatten2d(key).shape == (2, 3)


def test_hash_returns_correct_array_lengths():
    flat_key = np.zeros((4, ), dtype=np.uint8)
    hash_data, hash_kernel = keygen.hash(flat_key)

    assert len(hash_data) == 25
    assert len(hash_kernel) == 4


def test_hash_all_returns_correct_array_lengths():
    flat_keys = np.zeros((2, 4), dtype=np.uint8)
    hashes_data, hashes_kernel = keygen.hash_all(flat_keys)

    assert hashes_data.shape == (2, 25)
    assert hashes_kernel.shape == (2, 4)


def test_build_window_blocks_returns_correct_array_shape():
    hashes_data = np.zeros((2, 5, 5), dtype=np.uint8)
    blocks = keygen.build_window_blocks(hashes_data)

    assert blocks.shape == (2, 4, 4, 2, 2)


def test_convo2d_returns_correct_array_shape():
    hashes_data = np.zeros((2, 25), dtype=np.uint8)
    hashes_kernel = np.zeros((2, 4), dtype=np.uint8)

    result = keygen.convo2d(hashes_data, hashes_kernel)
    assert result.shape == (2, 16)


def test_calculate_padding_returns_0_if_n_0():
    assert keygen.calculate_padding(0) == 0


def test_calculate_padding_returns_1_if_n_1():
    assert keygen.calculate_padding(1) == 1


def test_calculate_padding_returns_6_if_n_8():
    assert keygen.calculate_padding(8) == 6


def test_calculate_padding_returns_5_if_n_9():
    assert keygen.calculate_padding(9) == 5


def test_calculate_padding_returns_0_if_n_14():
    assert keygen.calculate_padding(14) == 0


def test_calculate_padding_returns_1_if_n_15():
    assert keygen.calculate_padding(15) == 1


def test_pad_returns_correct_array_shape():
    convo_values = np.zeros((1, 16), dtype=np.uint8)
    assert keygen.pad(convo_values, 0).shape == (1, 25)


def test_pad_returns_correct_array_shape_with_n_9():
    convo_values = np.zeros((1, 16), dtype=np.uint8)
    assert keygen.pad(convo_values, 9).shape == (1, 25)


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
    key = np.zeros((1, 2), dtype=np.uint8)
    assert keygen.expand_key(key, 1).shape == (1, 1, 16)


def test_expand_key_with_key_3d_array_and_n_2():
    key = np.zeros((1, 2, 4), dtype=np.uint8)
    assert keygen.expand_key(key, 2).shape == (2, 1, 2, 16)
