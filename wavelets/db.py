from LiftingStep import Wavelet, LSStep,  LSType, LSBoundaryCondition
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def db2_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-1.7321], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [0.433, -0.067], 1, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [1], -1, LSBoundaryCondition.ZERO_PADDING),
    ]


def db4_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-0.3223], 1, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [-0.3001, -1.1171], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [-0.0188, 0.1176], 2, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [2.1318, 0.6364], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [-0.4691, 0.14, -0.0248], 0, LSBoundaryCondition.ZERO_PADDING),
    ]


def db2_ls_single(array: NDArray[int]) -> Any:
    #rows, cols = array.shape

    # TODO: boundary conditions

    # Divide
    f_even = array[::2]
    f_odd = array[1::2]

    # Run LS: Predict, Update, Predict
    f_odd = f_odd + np.floor(-1.7321*f_even + 0.5)
    print(f_even, f_odd)
    f_even = f_even + np.floor(
        0.5 + 0.433*f_odd - 0.067*np.pad(
            f_odd[1:], (0, 1), 'constant', constant_values=0))
    print(f_even, f_odd)
    f_odd = f_odd + np.floor(
        0.5 + np.pad(f_even[:-1], (1, 0), 'constant', constant_values=0))
    print(f_even, f_odd)

    return f_even, f_odd


if __name__ == "__main__":
    array = np.array([1, 2, 3, 4, 5, 6])
    print("Original")
    print(db2_ls_single(array))

    print("Daubechies 2 wavelet integer transform")
    print("======================================")
    lsw = db2_wavelet()
    approx, diff = array[::2], array[1::2]
    for step in lsw:
        approx, diff = step.evaluate(approx, diff)
        print(f"approx = {approx}, diff = {diff}")
    print(approx)
    print(diff)

    # print("\n\n\n\n")
    # print("Daubechies 4 wavelet integer transform")
    # print("======================================")
    #
    # approx, diff = array[::2], array[1::2]
    # db4 = db4_wavelet()
    # for step in db4:
    #     approx, diff = step.evaluate(approx, diff)
    # print(approx)
    # print(diff)
