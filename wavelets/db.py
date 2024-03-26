from LiftingStep import Wavelet, LSStep,  LSType, LSBoundaryCondition
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def db2_wavelet(inverse: bool = False) -> Wavelet:
    c = -1 if inverse else 1
    return [
        LSStep(LSType.PREDICT, [c*(-1.7321)], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [c*0.433, c*(-0.067)], 1, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [c*1], -1, LSBoundaryCondition.ZERO_PADDING),
    ]


def db4_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-0.3223], 1, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [-0.3001, -1.1171], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [0.1176, -0.0188], 2, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [0.6364, 2.1318], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.PREDICT, [-0.0248, 0.1400, -0.4691], 0, LSBoundaryCondition.ZERO_PADDING),
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

"""
LSStep(LSType.PREDICT, [-0.3223], 1, LSBoundaryCondition.ZERO_PADDING),
LSStep(LSType.UPDATE, [-0.3001, -1.1171], 0, LSBoundaryCondition.ZERO_PADDING),
LSStep(LSType.PREDICT, [0.1176, -0.0188], 2, LSBoundaryCondition.ZERO_PADDING),
LSStep(LSType.UPDATE, [0.6364, 2.1318], 0, LSBoundaryCondition.ZERO_PADDING),
LSStep(LSType.PREDICT, [-0.0248, 0.1400, -0.4691], 0, LSBoundaryCondition.ZERO_PADDING),
"""

def db4_ls_single(array: NDArray[int]) -> Any:
    # TODO: remove later....
    # Divide
    f_even = array[::2]
    f_odd = array[1::2]

    # Run LS: Predict, Update, Predict
    f_odd = f_odd + np.floor(0.5 - 0.3223*np.pad(f_even[1:], (0, 1)))
    print(f_even, f_odd)
    f_even = f_even + np.floor(0.5 - 1.1171*f_odd - 0.3001*np.pad(f_odd[:-1], (1, 0)))
    print(f_even, f_odd)
    f_odd = f_odd + np.floor(0.5 - 0.0188*np.pad(f_odd[2:], (0, 2)) +
                             0.1176 * np.pad(f_odd[1:], (0, 1)))
    print(f_even, f_odd)
    f_even = f_even + np.floor(0.5 + 2.1318*f_odd + 0.6364*np.pad(f_odd[:-1], (1, 0)))
    print(f_even, f_odd)
    f_odd = f_odd + np.floor(0.5 - 0.4691*f_even + 0.14*np.pad(f_even[:-1], (1, 0)) - 0.4691 *np.pad(f_even[:-2], (2, 0)))
    return f_even, f_odd


if __name__ == "__main__":
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8])
    # print("Original")
    # print(db2_ls_single(array))
    #
    # print("Daubechies 2 wavelet integer transform")
    # print("======================================")
    # lsw = db2_wavelet()
    # approx, diff = array[::2], array[1::2]
    # for step in lsw:
    #     approx, diff = step.evaluate(approx, diff)
    #     print(f"approx = {approx}, diff = {diff}")
    # print(approx)
    # print(diff)
    #
    # print("Inverse")
    # lsw = db2_wavelet()
    # cp_approx, cp_diff = approx, diff
    # for step in reversed(lsw):
    #     approx, diff = step.evaluate(approx, diff, inverse=True)
    #     print(f"approx = {approx}, diff = {diff}")
    # result = np.zeros(array.shape)
    # result[::2] = approx
    # result[1::2] = diff
    # print(result)


    print("\n\n\n\n")

    print(db4_ls_single(array))
    print("--- .... ----")

    print("Daubechies 4 wavelet integer transform")
    print("======================================")

    approx, diff = array[::2], array[1::2]
    db4 = db4_wavelet()
    for step in db4:
        approx, diff = step.evaluate(approx, diff)
        print(approx, diff)
    print(approx)
    print(diff)

    print("Inverse")
    print("=====================================")
    db4 = db4_wavelet()
    for step in reversed(db4):
        approx, diff = step.evaluate(approx, diff, inverse=True)
    result = np.zeros(array.shape)
    result[::2] = approx
    result[1::2] = diff
    print(result)
