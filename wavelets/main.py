from PIL import Image
from dataclasses import dataclass
import argparse
import numpy as np

from wavelets.lifting_step import Wavelet
from wavelets.db import db4_wavelet, db8_wavelet, db2_wavelet
from wavelets.haar import haar_wavelet
from wavelets.io import save_decomposed_img, read_decomposed_img
from wavelets.transform import wt_2d, wt_2d_inv


@dataclass(frozen=True)
class ParsedConfig:
    """Class for storing parsed configuration."""
    wavelet: Wavelet
    path_in: str
    path_out: str
    level: int
    operation_decompose: bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Integer wavelet decomposition')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--decomposition', action='store_true', help="Decompose the image.")
    group.add_argument('-r', '--reconstruction', action='store_true', help="Reconstruct the image.")

    parser.add_argument('-i', '--input', required=True,
                        help='Path to the mandatory input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-w', '--wavelet',
                        choices=[
                            'Haar', 'Daubechie2', 'Daubechie4', 'Daubechie8',
                        ],
                        required=True, help='Wavelet type', default='Haar')
    parser.add_argument('-l', '--level', type=int, required=True,
                        help='Level of decomposition (positive integer)')
    args = parser.parse_args()

    if args.level <= 0:
        parser.error("Level of decomposition must be a positive integer.")

    match args.wavelet:
        case 'Haar':
            wavelet = haar_wavelet()
        case 'Daubechie2':
            wavelet = db2_wavelet()
        case 'Daubechie4':
            wavelet = db4_wavelet()
        case 'Daubechie8':
            wavelet = db8_wavelet()
        case _:
            # Note: This should not happen, since argparse checks validity of wavelets
            raise RuntimeError("Unknown wavelet")
    configuration = ParsedConfig(
        wavelet=wavelet,
        path_in=args.input,
        path_out=args.output,
        level=args.level,
        operation_decompose=True if args.decomposition else False
    )

    if ParsedConfig.operation_decompose:
        with Image.open(configuration.path_in, 'r') as img_pil:
            image = np.array(img_pil).astype(int)
        decomposition = wt_2d(image, configuration.wavelet, configuration.level)
        save_decomposed_img(decomposition, configuration.path_out)
    else:
        decomposition = read_decomposed_img(configuration.path_in)
        image_numpy = wt_2d_inv(decomposition, configuration.wavelet)
        image = Image.fromarray(image_numpy)
        image.save(configuration.path_out)

