import tifffile
import numpy as np
import json
from numpy.typing import NDArray

from typing import *

from wavelets.main import Decomposition, Wavelet

def save_decomposed_img(decomposition: Decomposition, path: str) -> None:
    coefficients, config = decomposition
    config = config.copy()

    full_r, full_c = coefficients[0].shape  # Final shape
    full_r *= 2
    full_c *= 2

    data = []  # To save as multipage TIFF
    approx = coefficients.pop()
    img = np.zeros((full_r, full_c), dtype=coefficients[0].dtype)
    img[:approx.shape[0], :approx.shape[1]] = approx
    data.append(img)

    for _ in reversed(config):
        img = np.zeros((full_r, full_c), dtype=coefficients[0].dtype)
        diff_h = coefficients.pop()
        diff_diag = coefficients.pop()
        diff_vert = coefficients.pop()

        r, c = diff_h.shape
        img[:r, full_c // 2:full_c//2+c] = diff_h
        img[full_r // 2:full_r//2+r, full_c // 2:full_c//2+c] = diff_diag
        img[full_r // 2:full_r//2+r, :c] = diff_vert
        data.append(img)

    metadata = json.dumps(config)
    tifffile.imwrite(path, data, description=metadata)


def read_decomposed_img(path: str) -> Decomposition:
    raise NotImplementedError



if __name__ == "__main__":
    from wavelets.haar import haar_wavelet
    from wavelets import wt_2d
    from PIL import Image

    img = Image.open('../tests/resources/maly_rozsutec_2023_grayscale.jpg')
    img = np.array(img).astype(np.int32)

    decomposition = wt_2d(img, haar_wavelet(), 3)

    save_decomposed_img(decomposition, "decomposed_img.tif")

    loaded_img = read_decomposed_img("decomposed_img.tif")

    assert np.all(loaded_img == img)
