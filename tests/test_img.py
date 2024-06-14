import pytest
from PIL import Image
import numpy as np
from wavelets.haar import haar_wavelet
from wavelets.db import db4_wavelet, db8_wavelet, db2_wavelet
from wavelets.symlets import symlets2, symlets4, symlets8
from wavelets.coiflet import coiflet2, coiflet1
from wavelets.io import save_decomposed_img, read_decomposed_img
from wavelets.transform import wt_2d, wt_2d_inv
import os

@pytest.mark.parametrize("wavelet",
                         [db8_wavelet(), db4_wavelet(), db2_wavelet(),
                          haar_wavelet()])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_2d_square_img(wavelet, level):
    path = './resources/maly_rozsutec_2023_grayscale_square.jpg'
    with Image.open(path, 'r') as img_pil:
        img = np.array(img_pil, np.int32)

        assert img.ndim == 2
        assert img.shape[0] == img.shape[1]
        assert 2 ** np.floor(np.log2(img.shape[0])) == img.shape[0]

        transformed, config = wt_2d(img, wavelet, level)
        assert transformed[0].dtype == img.dtype
        assert len(transformed) == level * 3 + 1

        inverse = wt_2d_inv((transformed, config), wavelet)
        assert inverse.dtype == img.dtype

        assert np.all(np.isclose(inverse, img))
        assert np.all(inverse == img)

@pytest.mark.parametrize("wavelet",
                         [db8_wavelet(), db4_wavelet(), db2_wavelet(),
                          haar_wavelet(),
                          symlets2(), symlets4(), symlets8(),
                          coiflet1(), coiflet2()])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_2d_rectangular_img(wavelet, level):
    path = './resources/maly_rozsutec_2023_grayscale.jpg'
    with Image.open(path, 'r') as img_pil:
        img = np.array(img_pil, np.int32)

        assert img.ndim == 2
        transformed, config = wt_2d(img, wavelet, level)
        assert transformed[0].dtype == img.dtype
        assert len(transformed) == level * 3 + 1

        inverse = wt_2d_inv((transformed, config), wavelet)
        assert inverse.dtype == img.dtype

        assert np.all(np.isclose(inverse, img))
        assert np.all(inverse == img)


@pytest.mark.parametrize("path_to_image", [
     "./resources/maly_rozsutec_2023_grayscale.jpg",
     "./resources/maly_rozsutec_2023_grayscale_square.jpg"
])
@pytest.mark.parametrize("level", [1, 3])
@pytest.mark.parametrize("wavelet", [db8_wavelet(), haar_wavelet(), symlets8(),
                                     coiflet2()])
def test_saving_and_loading(path_to_image, level, wavelet):
    with Image.open(path_to_image, 'r') as img_pil:
        image = np.array(img_pil).astype(np.int32)

    decomposition = wt_2d(image, wavelet, level)
    path = os.path.join('.', 'artifacts', 'saved.tif')
    save_decomposed_img(decomposition, path)
    decomposition_found = read_decomposed_img(path)
    inv_img = wt_2d_inv(decomposition_found, wavelet)
    assert inv_img.dtype == image.dtype
    assert np.all(np.isclose(inv_img, image))
    assert np.all(image == inv_img)


