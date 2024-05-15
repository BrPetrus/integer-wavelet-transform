import numpy as np

from wavelets.transform import wt_1d, wt_1d_inv
from wavelets.haar import haar_wavelet
from PIL import Image


def test_short():
    signal = np.array([1, 2, 6, 7], dtype=np.int32)
    result = wt_1d(signal, haar_wavelet())

    assert np.all(result == np.array([2,7,1,1]))


def test_long():
    signal = np.array([1, 2, 6, 7, 6, 2, 3, 1, 5, 6, 7, 7], dtype=np.int32)
    result = wt_1d(signal, haar_wavelet())

    assert np.all(result == np.array([2,7,4,2,6,7,1,1,-4,-2,1,0]))


def test_inv():
    signal = np.array([1, 2, 6, 7, 6, 2, 3, 1, 5, 6, 7, 7], dtype=np.int32)
    transformed = wt_1d(signal, haar_wavelet())
    result = wt_1d_inv(transformed, haar_wavelet())
    assert np.all(result == signal)

def test_inv_long():
    signal = np.arange(128, 0,-1, dtype=np.int32)
    trans = wt_1d(signal, haar_wavelet())
    inv = wt_1d_inv(trans, haar_wavelet())
    assert np.all(inv == signal)


def test_2d_single_row():
    path = './resources/maly_rozsutec_2023_grayscale_square.jpg'
    with Image.open(path, 'r') as img_pil:
        img = np.array(img_pil, dtype=np.int32)

    rows, cols = img.shape
    for row in range(rows):
        print(f"Testing for {row}")
        signal = img[row, :].copy()
        trans = wt_1d(signal, haar_wavelet())
        inv = wt_1d_inv(trans.copy(), haar_wavelet())
        assert np.all(inv == signal)
        print(f"Holds for {row}/{rows}")
