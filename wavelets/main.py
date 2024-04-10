from typing import TypeVar, List, Tuple

import numpy as np
from numpy.typing import NDArray

from wavelets.lifting_step import Wavelet

T = TypeVar("T", bound=np.generic, covariant=True)

Config = List[Tuple[int, int]]


def wt_1d(signal: NDArray[T], lifting_scheme: Wavelet) -> NDArray[T]:
    if not np.issubdtype(signal.dtype, np.integer):
        raise RuntimeWarning("Input signal does not use integer type!")

    orig_type = signal.dtype
    signal = signal.astype(int)
    approx, diff = signal[::2].copy(), signal[1::2].copy()
    for step in lifting_scheme:
        approx, diff = step.evaluate(approx, diff)
        print(approx, diff)
    result = np.zeros_like(signal)
    cols = signal.shape[-1]
    result[:cols // 2] = approx
    result[cols // 2:] = diff

    if result.max() >= np.iinfo(orig_type).max or result.min() <= np.iinfo(
            orig_type).min:
        raise RuntimeError(
            "Original type of the image cannot hold the transformed signal!")

    result = result.astype(orig_type)

    assert result.dtype == orig_type
    assert np.issubdtype(result.dtype, np.integer)
    return result


def wt_1d_inv(signal: NDArray[T], lifting_scheme: Wavelet) -> NDArray[T]:
    if not np.issubdtype(signal.dtype, np.integer):
        raise RuntimeWarning("Input signal does not use a numpy integer type!")
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
        raise RuntimeError(
            "Original type of the image cannot hold the transformed signal!")

    result = result.astype(orig_type)
    assert result.dtype == orig_type
    assert np.issubdtype(result.dtype, np.integer)
    return result


def _wt_2d(img: NDArray[T], lifting_scheme: Wavelet, coords: Tuple[int, int]) \
        -> None:
    rows, cols = coords

    result = np.zeros_like(img, dtype=img.dtype)
    # TODO: Do this without for cycle for efficiency
    for row in range(rows):
        result[row] = wt_1d(img[row, :].copy(), lifting_scheme)
    for col in range(cols):
        result[:, col] = wt_1d(result[:, col], lifting_scheme)

    img[:rows, :cols] = result


def wt_2d(image: NDArray[T], lifting_scheme: Wavelet, level: int = 1) \
        -> Tuple[NDArray[T], Config]:
    rows, cols = image.shape
    out = image.copy()
    configuration: Config = []
    for current_level in range(level):
        configuration.append((rows, cols))
        _wt_2d(out[:rows, :cols], lifting_scheme, (rows, cols))
        rows //= 2
        cols //= 2
    return out, configuration


def _wt_2d_inv(image: NDArray[T], lifting_scheme: Wavelet) -> None:
    rows, cols = image.shape
    result = image.copy()
    for col in range(cols):
        result[:, col] = wt_1d_inv(result[:, col], lifting_scheme)
    for row in reversed(range(rows)):
        result[row, :] = wt_1d_inv(result[row, :].copy(), lifting_scheme)
    image[:rows, :cols] = result


def wt_2d_inv(image: NDArray[T], lifting_scheme: Wavelet, config: Config) \
        -> NDArray[T]:
    out = image.copy()
    for shape in reversed(config):
        rows, cols = shape
        _wt_2d_inv(out[:rows, :cols], lifting_scheme)
    return out
