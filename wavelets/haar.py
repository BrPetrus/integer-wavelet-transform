from wavelets.lifting_step import Wavelet, LSStep, LSType, LSBoundaryCondition


def haar_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-1], 0),
        LSStep(LSType.UPDATE, [0.5], 0)
    ]

