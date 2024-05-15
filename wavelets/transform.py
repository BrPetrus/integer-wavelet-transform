from typing import TypeVar, List, Tuple

import numpy as np
from numpy.typing import NDArray

from wavelets.lifting_step import Wavelet
from wavelets.misc import TransformError, Decomposition

T = TypeVar("T", bound=np.generic, covariant=True)


def wt_1d(signal: NDArray[T], lifting_scheme: Wavelet) -> NDArray[T]:
    if not np.issubdtype(signal.dtype, np.integer):
        raise TransformError("Input signal does not use integer type!")

    orig_type = signal.dtype
    signal = signal.astype(int)
    approx, diff = signal[::2].copy(), signal[1::2].copy()
    for step in lifting_scheme:
        approx, diff = step.evaluate(approx, diff)
    result = np.zeros_like(signal)
    cols = signal.shape[-1]
    result[:cols // 2] = approx
    result[cols // 2:] = diff

    if result.max() >= np.iinfo(orig_type).max or result.min() <= np.iinfo(
            orig_type).min:
        raise TransformError(
            "Original type of the image cannot hold the transformed signal!")

    result = result.astype(orig_type)

    assert result.dtype == orig_type
    assert np.issubdtype(result.dtype, np.integer)
    return result


def wt_1d_inv(signal: NDArray[T], lifting_scheme: Wavelet) -> NDArray[T]:
    if not np.issubdtype(signal.dtype, np.integer):
        raise TransformError("Input signal does not use a numpy integer type!")
    orig_type = signal.dtype
    signal = signal.astype(int)
    cols = signal.shape[-1]
    approx, diff = signal[:cols // 2].copy(), signal[cols // 2:].copy()
    for step in reversed(lifting_scheme):
        approx, diff = step.evaluate(approx, diff, inverse=True)
    result = np.zeros_like(signal)
    result[::2] = approx
    result[1::2] = diff

    if result.max() >= np.iinfo(orig_type).max or result.min() <= np.iinfo(
            orig_type).min:
        raise TransformError(
            "Original type of the image cannot hold the transformed signal!")

    result = result.astype(orig_type)
    assert result.dtype == orig_type
    assert np.issubdtype(result.dtype, np.integer)
    return result


def _wt_2d(img: NDArray[T], lifting_scheme: Wavelet, coords: Tuple[int, int]) \
        -> NDArray[T]:
    rows, cols = coords

    result = np.zeros(coords, dtype=img.dtype)
    # TODO: Do this without for cycle for efficiency
    for row in range(rows):
        result[row] = wt_1d(img[row, :].copy(), lifting_scheme)
    for col in range(cols):
        result[:, col] = wt_1d(result[:, col], lifting_scheme)

    return result



def wt_2d(image: NDArray[T], lifting_scheme: Wavelet, level: int = 1) -> Decomposition:
    configurations: Config = []
    coefficients: List[NDArray[T]] = []

    rows, cols = image.shape
    approx = image
    # TODO: Limit the value of the var. level.
    for curr_lvl in range(level):
        if rows == 1 and cols == 1:
            raise TransformError(f"Cannot decompose the image more than {curr_lvl} times.")

        configurations.append((rows, cols))

        # Pad the image
        pad_rows, pad_cols = 0, 0
        if rows % 2 == 1:
            pad_rows = 1
            rows += 1
        if cols % 2 == 1:
            pad_cols = 1
            cols += 1
        image = np.pad(image, ((0, pad_rows), (0, pad_cols)))

        # Transform
        image = _wt_2d(image, lifting_scheme, (rows, cols))

        # Split
        approx = image[:rows // 2, :cols // 2]
        # TODO: Write into docs, which of the RGB channels corres. to which difference.
        coefficients.append(image[rows // 2:, :cols // 2])  # Diff. vert.
        coefficients.append(image[rows // 2:, cols // 2:])  # Diff. diag.
        coefficients.append(image[:rows // 2, cols // 2:])  # Diff. hor.

        # TODO: sanity check
        image = approx.copy()

        rows //= 2
        cols //= 2

    coefficients.append(approx)

    return coefficients, configurations


def _wt_2d_inv(image: NDArray[T], lifting_scheme: Wavelet) -> NDArray[T]:
    rows, cols = image.shape
    result = image.copy()
    for col in range(cols):
        result[:, col] = wt_1d_inv(result[:, col], lifting_scheme)
    for row in reversed(range(rows)):
        result[row, :] = wt_1d_inv(result[row, :].copy(), lifting_scheme)
    return result


def wt_2d_inv(decomposition: Decomposition, lifting_scheme: Wavelet) -> NDArray[T]:
    coefficients, config = decomposition

    approx = coefficients.pop()
    for shape in reversed(config):
        target_rows, target_cols = shape

        # Get coefficients
        diff_h = coefficients.pop()
        diff_diag = coefficients.pop()
        diff_vert = coefficients.pop()

        # Construct image
        new_rows, new_cols = approx.shape[0] * 2, approx.shape[1] * 2
        img = np.zeros((new_rows, new_cols), dtype=approx.dtype)
        img[:new_rows // 2, :new_cols // 2] = approx
        img[:new_rows // 2, new_cols // 2:] = diff_h
        img[new_rows // 2:, new_cols // 2:] = diff_diag
        img[new_rows // 2:, :new_cols // 2] = diff_vert

        # Inverse transform
        img = _wt_2d_inv(img, lifting_scheme)

        # Remove padding
        if new_rows != target_rows:
            img = img[:-1, :]
        if new_cols != target_cols:
            img = img[:, :-1]

        approx = img

    return approx
