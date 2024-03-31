import numpy as np
from wavelets import wt_1d, inv_wt_1d
from wavelets.haar import haar_wavelet

def test_short():
    signal = np.array([1, 2, 6, 7])
    result = wt_1d(signal, haar_wavelet())

    assert np.all(result == np.array([2, 1, 7, 1]))


def test_long():
    signal = np.array([1, 2, 6, 7, 6, 2, 3, 1, 5, 6, 7, 7])
    result = wt_1d(signal, haar_wavelet())

    assert np.all(result == np.array([2, 1, 7, 1, 4, -4, 2, -2, 6, 1, 7, 0]))


def test_inv():
    signal = np.array([1, 2, 6, 7, 6, 2, 3, 1, 5, 6, 7, 7])
    transformed = wt_1d(signal, haar_wavelet())
    result = inv_wt_1d(transformed, haar_wavelet())
    assert np.all(result == signal)