import numpy as np
import pytest

from wavelets import wt_1d, wt_1d_inv, wt_2d_inv, wt_2d
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


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
def test_2d_square_img(level: int):
    path = './resources/maly_rozsutec_2023_grayscale_square.jpg'
    with Image.open(path, 'r') as img_pil:
        img = np.array(img_pil, np.int32)

        assert img.ndim == 2
        assert img.shape[0] == img.shape[1]
        assert 2 ** np.floor(np.log2(img.shape[0])) == img.shape[0]

        transformed, config = wt_2d(img, haar_wavelet(), level)
        assert transformed[0].dtype == img.dtype
        assert len(transformed) == level * 3 + 1

        result = wt_2d_inv((transformed, config), haar_wavelet())
        assert result.dtype == img.dtype

        result_pil = Image.fromarray(result)
        result_pil.save("./artifacts/test_2d_haar_inv.tif")

        assert np.all(np.isclose(result, img))
        assert np.all(result == img)


