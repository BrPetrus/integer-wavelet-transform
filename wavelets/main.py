import numpy as np
from numpy.typing import NDArray
from .lifting_step import Wavelet


def wt_1d(signal: NDArray[int], lifting_scheme: Wavelet) -> NDArray[int]:
    approx, diff = signal[::2], signal[1::2]
    for step in lifting_scheme:
        approx, diff = step.evaluate(approx, diff)
        print(approx, diff)
    result = np.zeros_like(signal)
    result[::2] = approx
    result[1::2] = diff
    return result


def inv_wt_1d(signal: NDArray[int], lifting_scheme: Wavelet) -> NDArray[int]:
    approx, diff = signal[::2], signal[1::2]
    for step in reversed(lifting_scheme):
        approx, diff = step.evaluate(approx, diff, inverse=True)
    result = np.zeros_like(signal)
    result[::2] = approx
    result[1::2] = diff
    return result


def wt_2d(signal: NDArray[int], lifting_scheme: Wavelet) -> NDArray[int]:
    raise NotImplementedError
    rows, col = signal.shape

    result = np.zeros_like(signal)
    for row in range(rows):
        signal[row] = wt_1d(signal[row], lifting_scheme)

    signal = signal.T
    for col in range(col):
        signal[col] = wt_1d(signal[col], lifting_scheme)


