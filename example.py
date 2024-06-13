import numpy as np
from PIL import Image
import os

from wavelets.haar import haar_wavelet
from wavelets.db import db8_wavelet
from wavelets.transform import wt_2d, wt_2d_inv
from wavelets.io import save_decomposed_img, read_decomposed_img


def main():
    # Read an image
    with Image.open('tests/resources/maly_rozsutec_2023_grayscale.jpg') as img:
        data = np.array(img).astype(np.int32)

    # Use Haar wavelets
    wavelets = haar_wavelet()

    # Use Db8 wavelet
    wavelet_db8 = db8_wavelet()

    # Decompose
    decomposition = wt_2d(data, wavelets, level=3)

    # Save decomposition
    path = "outputs/decomposition.tif"
    if not os.path.exists("outputs"):
        os.mkdir(os.path.dirname(path))

    save_decomposed_img(decomposition, path)

    # Read decomposed image
    decomposition = read_decomposed_img(path)

    # Inverse transform
    data_reconstructed = wt_2d_inv(decomposition, wavelets)

    # Compare
    np.all(data_reconstructed == data)


if __name__ == "__main__":
    main()
