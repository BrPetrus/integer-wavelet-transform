import pytest
from PIL import Image
import numpy as np
from wavelets import wt_2d, wt_2d_inv
from wavelets.haar import haar_wavelet
from wavelets.db import db4_wavelet, db8_wavelet, db2_wavelet


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
                          haar_wavelet()])
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
