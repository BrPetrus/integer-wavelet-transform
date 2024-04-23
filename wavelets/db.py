import numpy as np

from wavelets.lifting_step import Wavelet, LSStep, LSType


def db2_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-1.7321], 0),
        LSStep(LSType.UPDATE, [0.433, -0.067], 1),
        LSStep(LSType.PREDICT, [1], -1),
    ]


def db4_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-0.3223], 1),
        LSStep(LSType.UPDATE, [-0.3001, -1.1171], 0),
        LSStep(LSType.PREDICT, [0.1176, -0.0188], 2),
        LSStep(LSType.UPDATE, [0.6364, 2.1318], 0),
        LSStep(LSType.PREDICT, [-0.0248, 0.1400, -0.4691], 0),
    ]


def db8_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-5.7496], 0),
        LSStep(LSType.UPDATE, [0.1688, -0.0523], 1),
        LSStep(LSType.PREDICT, [-7.4021, 14.5428], -1),
        LSStep(LSType.UPDATE, [0.0609, -0.0324], 3),
        LSStep(LSType.PREDICT, [-2.7557, 5.8187], -3),
        LSStep(LSType.UPDATE, [0.2420, 0.9453], 5),
        LSStep(LSType.PREDICT, [-0.0018, 1.8884e-04], -3),
        LSStep(LSType.UPDATE, [-0.2241, -0.9526], 5),
        LSStep(LSType.PREDICT, [0.0272, -0.2470, 1.0497], -5),
    ]


if __name__ == "__main__":
    array = np.array([5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1, 2, 3, 4, 5, 6])

    print("Daubechies 2 wavelet integer transform")
    print("======================================")
    lsw = db2_wavelet()
    approx, diff = array[::2], array[1::2]
    for step in lsw:
        approx, diff = step.evaluate(approx, diff)
        print(f"approx = {approx}, diff = {diff}")
    print(approx)
    print(diff)

    print("Inverse")
    lsw = db2_wavelet()
    cp_approx, cp_diff = approx, diff
    for step in reversed(lsw):
        approx, diff = step.evaluate(approx, diff, inverse=True)
        print(f"approx = {approx}, diff = {diff}")
    result = np.zeros(array.shape)
    result[::2] = approx
    result[1::2] = diff
    print(result)
