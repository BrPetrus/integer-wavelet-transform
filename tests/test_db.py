from typing import Any

import pytest
from PIL import Image

from wavelets import *
from wavelets.db import *


def db2_ls_single(array: NDArray[int]) -> Any:
    # TODO: remove later
    # Divide
    f_even = array[::2]
    f_odd = array[1::2]

    # Run LS: Predict, Update, Predict
    f_odd = f_odd + np.floor(-1.7321 * f_even + 0.5)
    f_even = f_even + np.floor(
        0.5 + 0.433 * f_odd - 0.067 * np.pad(f_odd[1:], (0, 1), 'constant',
            constant_values=0))
    f_odd = f_odd + np.floor(
        0.5 + np.pad(f_even[:-1], (1, 0), 'constant', constant_values=0))

    return f_even, f_odd


def db4_ls_single(array: NDArray[int]) -> Any:
    # TODO: remove later....
    # Divide
    f_even = array[::2]
    f_odd = array[1::2]

    # Run LS: Predict, Update, Predict
    f_odd = f_odd + np.floor(0.5 - 0.3223 * np.pad(f_even[1:], (0, 1)))
    print(f_even, f_odd)
    f_even = f_even + np.floor(
        0.5 - 1.1171 * f_odd - 0.3001 * np.pad(f_odd[:-1], (1, 0)))
    print(f_even, f_odd)
    f_odd = f_odd + np.floor(
        0.5 - 0.0188 * np.pad(f_odd[2:], (0, 2)) + 0.1176 * np.pad(f_odd[1:],
                                                                   (0, 1)))
    print(f_even, f_odd)
    f_even = f_even + np.floor(
        0.5 + 2.1318 * f_odd + 0.6364 * np.pad(f_odd[:-1], (1, 0)))
    print(f_even, f_odd)
    f_odd = f_odd + np.floor(0.5 - 0.4691 * f_even + 0.14 * np.pad(f_even[:-1],
                                                                   (1,
                                                                    0)) - 0.4691 * np.pad(
        f_even[:-2], (2, 0)))
    return f_even, f_odd


def test_db2_cmp() -> None:
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6],
                     dtype=np.int32)

    print("old")
    result = np.zeros_like(array)
    cols = result.shape[-1]
    result[:cols // 2], result[cols // 2:] = db2_ls_single(array)
    print("---")

    assert np.all(result == wt_1d(array, db2_wavelet()))


def test_db2_short_4() -> None:
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    result = wt_1d(array, db2_wavelet())
    assert np.all(result == np.array([1, 3, 0, 0]))


def test_db2_short_2() -> None:
    array = np.array([1, 2], dtype=np.int32)
    result = wt_1d(array, db2_wavelet())
    assert np.all(result == np.array([1, 0]))


def test_db2_inv() -> None:
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6],
                     dtype=np.int32)

    transformed = wt_1d(array.copy(), db2_wavelet()).copy()

    cols = transformed.shape[-1]
    approx, diff = transformed[:cols // 2], transformed[cols // 2:]

    assert np.all(approx == np.array([2, 3, 4, 6, 8, 1, 3, 4]))
    assert np.all(diff == np.array([-7, 1, 0, 0, 1, 8, 0, 0]))

    result = wt_1d_inv(transformed.copy(), db2_wavelet())

    assert np.all(result == array)


def test_db4_cmp() -> None:
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6],
                     dtype=np.int32)
    result = wt_1d(array.copy(), db4_wavelet())

    assert np.all(result == np.array(
        [6, 5, 10, 11, 23, 6, 5, 12, -2, 0, 0, 0, 3, 1, 0,
         1]))  # assert np.all(result == np.array([6,-2,5,0,10,0,11,0,23,3,6,1,5,0,12,1]))


def test_db8() -> None:
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6],
                     dtype=np.int32)
    result = wt_1d(array.copy(), db8_wavelet())

    assert np.all(result == np.array(
        [1, 3, 2, 5, 2, 1, 2, 1, -27, 2, -1, 3, 0, 17, 6, -2]))


@pytest.mark.xfail(reason="Signal too short.")
def test_db8_short() -> None:
    array = np.array([5, 2, 3, 4], dtype=np.int32)
    result = wt_1d(array.copy(), db8_wavelet())

    assert np.all(result == np.array([1, -26, 1, -3]))


def test_db8_inv() -> None:
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6],
                     dtype=np.int32)
    first = wt_1d(array.copy(), db8_wavelet())
    result = wt_1d_inv(first, db8_wavelet())
    assert np.all(result == array)


def test_db8_inv_short() -> None:
    array = np.array([5, 2, 3, 4], dtype=np.int32)
    transformed = wt_1d(array.copy(), db8_wavelet())
    result = wt_1d_inv(transformed.copy(), db8_wavelet())
    assert (np.all(result == array))


